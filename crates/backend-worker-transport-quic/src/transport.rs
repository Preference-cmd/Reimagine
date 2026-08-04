use std::net::SocketAddr;

use async_trait::async_trait;
use reimagine_backend_worker_protocol::{
    TransportDescription, TransportError, TransportKind, WorkerTransport,
};

use crate::tls::SelfSignedCert;

/// QUIC-based transport for remote worker communication.
///
/// Wraps a `quinn::Connection` and provides bidirectional streams
/// for the worker protocol. The `FrameCodec` (length-prefixed JSON)
/// works over QUIC streams without modification.
pub struct QuicTransport {
    connection: quinn::Connection,
    server_name: String,
}

impl QuicTransport {
    /// Connect to a remote worker via QUIC.
    ///
    /// `cert` is used to configure TLS trust. `server_name` must
    /// match the hostname in the server's certificate.
    pub async fn connect(
        bind_addr: SocketAddr,
        server_addr: SocketAddr,
        server_name: &str,
        cert: &SelfSignedCert,
    ) -> Result<Self, TransportError> {
        let endpoint = crate::tls::client_endpoint(bind_addr, cert)
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;
        let connection = endpoint
            .connect(server_addr, server_name)
            .map_err(|e| TransportError::ConnectionFailed(format!("connect failed: {e}")))?
            .await
            .map_err(|e| TransportError::ConnectionFailed(format!("handshake failed: {e}")))?;
        Ok(Self {
            connection,
            server_name: server_name.to_owned(),
        })
    }

    /// Wrap an existing quinn connection (e.g. from an accepted connection).
    pub fn from_connection(connection: quinn::Connection, server_name: String) -> Self {
        Self {
            connection,
            server_name,
        }
    }

    /// Open a bidirectional stream for sending and receiving.
    pub async fn open_bi(
        &self,
    ) -> Result<(quinn::SendStream, quinn::RecvStream), TransportError> {
        self.connection
            .open_bi()
            .await
            .map_err(|e| TransportError::Io(format!("open_bi failed: {e}")))
    }

    /// Get the remote address of the connection.
    pub fn remote_addr(&self) -> SocketAddr {
        self.connection.remote_address()
    }
}

#[async_trait]
impl WorkerTransport for QuicTransport {
    fn description(&self) -> TransportDescription {
        TransportDescription {
            kind: TransportKind::Quic,
            endpoint: format!("quic://{}:{}", self.server_name, self.remote_addr()),
        }
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        self.connection
            .close(0u32.into(), b"shutdown");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tls::SelfSignedCert;
    use std::net::Ipv4Addr;

    #[test]
    fn self_signed_cert_generation() {
        let cert = SelfSignedCert::generate("localhost").unwrap();
        // Should be able to build both server and client configs
        let _server = cert.server_config().unwrap();
        let _client = cert.client_config().unwrap();
    }

    #[tokio::test]
    async fn quic_transport_connect_and_shutdown() {
        let cert = SelfSignedCert::generate("localhost").unwrap();
        let listen_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0);
        let endpoint = crate::tls::server_endpoint(listen_addr, &cert).unwrap();
        let listen_addr = endpoint.local_addr().unwrap();

        // Spawn a task that accepts one connection and echoes
        let server = tokio::spawn(async move {
            let incoming = endpoint.accept().await.unwrap();
            let conn = incoming.await.unwrap();
            let (mut send, mut recv) = conn.accept_bi().await.unwrap();
            // Echo: read what client sends, write it back
            let mut buf = vec![0u8; 1024];
            let n = recv.read(&mut buf).await.unwrap().unwrap();
            send.write_all(&buf[..n]).await.unwrap();
            send.finish().unwrap();
            // Keep the connection alive until client is done
            // by waiting for the connection to close naturally
            let _ = conn.closed().await;
        });

        // Client trusts the server's certificate
        let client_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0);
        let transport =
            QuicTransport::connect(client_addr, listen_addr, "localhost", &cert)
                .await
                .unwrap();

        // Verify description
        assert_eq!(transport.description().kind, TransportKind::Quic);
        assert!(transport.description().endpoint.contains("quic://"));

        // Open a bidirectional stream and do a round-trip
        let (mut send, mut recv) = transport.open_bi().await.unwrap();
        send.write_all(b"hello quic").await.unwrap();
        send.finish().unwrap();

        let response = recv.read_to_end(1024).await.unwrap();
        assert_eq!(response, b"hello quic");

        // Shutdown
        transport.shutdown().await.unwrap();
        server.await.unwrap();
    }
}
