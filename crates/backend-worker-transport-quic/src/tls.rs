use std::sync::Arc;

use quinn::crypto::rustls::QuicClientConfig;
use rustls::ClientConfig;
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};

use crate::Error;

/// Self-signed certificate for LAN development.
pub struct SelfSignedCert {
    pub cert_der: CertificateDer<'static>,
    pub key_der: PrivatePkcs8KeyDer<'static>,
}

impl SelfSignedCert {
    /// Generate a new self-signed certificate for the given hostname.
    pub fn generate(hostname: &str) -> Result<Self, Error> {
        let cert = rcgen::generate_simple_self_signed(vec![hostname.to_owned()])
            .map_err(|e| Error::Certificate(e.to_string()))?;
        let cert_der = CertificateDer::from(cert.cert);
        let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());
        Ok(Self { cert_der, key_der })
    }

    /// Build a rustls `ServerConfig` for use with quinn.
    pub fn server_config(&self) -> Result<rustls::ServerConfig, Error> {
        let mut server_config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(|e| Error::Tls(e.to_string()))?
        .with_no_client_auth()
        .with_single_cert(
            vec![self.cert_der.clone()],
            rustls::pki_types::PrivateKeyDer::Pkcs8(self.key_der.clone_key()),
        )
        .map_err(|e| Error::Tls(e.to_string()))?;
        server_config.alpn_protocols = vec![b"reimagine-worker-v1".to_vec()];
        Ok(server_config)
    }

    /// Build a rustls `ClientConfig` that trusts this certificate.
    pub fn client_config(&self) -> Result<rustls::ClientConfig, Error> {
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(self.cert_der.clone())
            .map_err(|e| Error::Tls(e.to_string()))?;
        let mut client_config =
            ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                .with_safe_default_protocol_versions()
                .map_err(|e| Error::Tls(e.to_string()))?
                .with_root_certificates(roots)
                .with_no_client_auth();
        client_config.alpn_protocols = vec![b"reimagine-worker-v1".to_vec()];
        Ok(client_config)
    }
}

/// Build a quinn `Endpoint` configured as a server.
pub fn server_endpoint(
    listen_addr: std::net::SocketAddr,
    cert: &SelfSignedCert,
) -> Result<quinn::Endpoint, Error> {
    let server_config = cert.server_config()?;
    let quic_server_config = quinn::crypto::rustls::QuicServerConfig::try_from(server_config)
        .map_err(|e| Error::Tls(e.to_string()))?;
    let server_config = quinn::ServerConfig::with_crypto(Arc::new(quic_server_config));
    quinn::Endpoint::server(server_config, listen_addr)
        .map_err(|e| Error::ConnectionFailed(format!("failed to bind: {e}")))
}

/// Build a quinn `Endpoint` configured as a client.
pub fn client_endpoint(
    bind_addr: std::net::SocketAddr,
    cert: &SelfSignedCert,
) -> Result<quinn::Endpoint, Error> {
    let client_config = cert.client_config()?;
    let quic_client_config =
        QuicClientConfig::try_from(client_config).map_err(|e| Error::Tls(e.to_string()))?;
    let mut endpoint = quinn::Endpoint::client(bind_addr)
        .map_err(|e| Error::ConnectionFailed(format!("failed to bind: {e}")))?;
    endpoint.set_default_client_config(quinn::ClientConfig::new(Arc::new(quic_client_config)));
    Ok(endpoint)
}
