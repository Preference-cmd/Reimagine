use std::net::SocketAddr;

use quinn::Endpoint;
use reimagine_backend_worker_protocol::{
    HostHello, WorkerHello, WorkerIdentity, WorkerIncarnationId,
    WorkerInstallationId, WorkerInstanceProfile, WorkerProfile, WireMessage,
    negotiate_protocol, ProtocolRange,
};

use crate::tls::SelfSignedCert;
use crate::Error;

/// A QUIC listener that accepts connections and performs the worker handshake.
pub struct QuicWorkerListener {
    endpoint: Endpoint,
}

impl QuicWorkerListener {
    /// Create and start a QUIC worker listener on the given address.
    pub fn start(listen_addr: SocketAddr, cert: &SelfSignedCert) -> Result<Self, Error> {
        let endpoint = crate::tls::server_endpoint(listen_addr, cert)?;
        Ok(Self { endpoint })
    }

    /// Get the local address this listener is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, Error> {
        self.endpoint
            .local_addr()
            .map_err(|e| Error::ConnectionFailed(format!("failed to get local addr: {e}")))
    }

    /// Accept the next incoming connection.
    ///
    /// Returns the connection, bidirectional stream pair, and the
    /// `WorkerHello` after completing the handshake.
    pub async fn accept(
        &self,
    ) -> Result<(quinn::Connection, quinn::SendStream, quinn::RecvStream, WorkerHello), Error> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| Error::ConnectionFailed("endpoint closed".into()))?;
        let connection = incoming
            .await
            .map_err(|e| Error::ConnectionFailed(format!("connection failed: {e}")))?;
        let (mut send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|e| Error::ConnectionFailed(format!("accept_bi failed: {e}")))?;

        // Read HostHello: u32 length prefix + JSON payload
        let mut prefix = [0u8; 4];
        recv.read_exact(&mut prefix)
            .await
            .map_err(|e| Error::Io(format!("read host_hello prefix: {e}")))?;
        let len = u32::from_be_bytes(prefix) as usize;
        let mut payload = vec![0u8; len];
        recv.read_exact(&mut payload)
            .await
            .map_err(|e| Error::Io(format!("read host_hello payload: {e}")))?;
        let host_hello: HostHello = match serde_json::from_slice::<WireMessage>(&payload)
            .map_err(|e| Error::Io(format!("deserialize host_hello: {e}")))?
        {
            WireMessage::HostHello(h) => h,
            other => return Err(Error::Io(format!("expected HostHello, got {:?}", other.kind()))),
        };

        let selected = negotiate_protocol(
            host_hello.supported_protocols,
            ProtocolRange::new(1, 1),
        )
        .map_err(|e| Error::Protocol(e.to_string()))?;

        let worker_hello = WorkerHello {
            selected_protocol: selected,
            identity: WorkerIdentity {
                backend_instance_id: reimagine_backend_worker_protocol::BackendInstanceId::from("fake:cpu:default"),
                installation_id: WorkerInstallationId::from("fake-installation"),
                incarnation_id: WorkerIncarnationId(format!("fake-quic-{}", std::process::id())),
                worker_version: env!("CARGO_PKG_VERSION").to_owned(),
                backend_kind: "fake".to_owned(),
                target: std::env::consts::ARCH.to_owned(),
                manifest_digest: "test-manifest".to_owned(),
            },
            profile: WorkerProfile {
                instances: vec![WorkerInstanceProfile {
                    backend_instance_id: reimagine_backend_worker_protocol::BackendInstanceId::from("fake:cpu:default"),
                    device_label: "cpu".to_owned(),
                    capabilities: vec![
                        "echo".to_owned(),
                        "delay".to_owned(),
                        "progress".to_owned(),
                    ],
                    operation_options: serde_json::json!({}),
                }],
            },
        };

        // Send WorkerHello: u32 length prefix + JSON payload
        let hello_json = serde_json::to_vec(&WireMessage::WorkerHello(worker_hello.clone()))
            .map_err(|e| Error::Io(format!("serialize worker_hello: {e}")))?;
        send.write_all(&(hello_json.len() as u32).to_be_bytes())
            .await
            .map_err(|e| Error::Io(format!("write hello length: {e}")))?;
        send.write_all(&hello_json)
            .await
            .map_err(|e| Error::Io(format!("write hello payload: {e}")))?;

        Ok((connection, send, recv, worker_hello))
    }
}
