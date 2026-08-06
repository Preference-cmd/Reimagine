//! TLS helpers for the gRPC worker transport: self-signed server
//! identities (dev/LAN) and a dev-only accept-any client verifier.

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::WebPkiSupportedAlgorithms;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};

/// Ensure the process-level rustls crypto provider is installed.
///
/// rustls 0.23 requires an explicit process-wide provider before any
/// connection is built; the `ring` feature supplies one but does not
/// install it automatically. Idempotent: a second call returns `Err`
/// (already installed) which we ignore.
pub fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// A self-signed certificate identity in PEM form, for
/// `tonic::transport::Identity::from_pem`.
#[derive(Debug, Clone)]
pub struct SelfSignedIdentity {
    /// PEM-encoded certificate chain.
    pub cert_pem: String,
    /// PEM-encoded PKCS#8 private key.
    pub key_pem: String,
}

/// Generate a fresh self-signed identity for the given hostname.
///
/// Clients can either pin the certificate (compare against the worker's
/// advertised fingerprint) or trust it directly via its PEM form.
pub fn generate_self_signed_identity(hostname: &str) -> Result<SelfSignedIdentity, String> {
    ensure_crypto_provider();
    let certified = rcgen::generate_simple_self_signed(vec![hostname.to_owned()])
        .map_err(|e| format!("failed to generate self-signed certificate: {e}"))?;
    Ok(SelfSignedIdentity {
        cert_pem: certified.cert.pem(),
        key_pem: certified.key_pair.serialize_pem(),
    })
}

/// Server-cert verifier that accepts any certificate (dev only).
///
/// Handshake signatures are still verified cryptographically; only the
/// certificate chain/fingerprint check is skipped.
#[derive(Debug)]
pub struct AcceptAnyServerVerifier {
    algorithms: WebPkiSupportedAlgorithms,
}

impl Default for AcceptAnyServerVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl AcceptAnyServerVerifier {
    /// Create an accept-any verifier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithms: rustls::crypto::ring::default_provider().signature_verification_algorithms,
        }
    }
}

impl ServerCertVerifier for AcceptAnyServerVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algorithms.supported_schemes()
    }
}
