//! Trust On First Use (TOFU) store and certificate pinning for QUIC workers.
//!
//! A worker presents a self-signed certificate. The host derives a stable
//! fingerprint (SHA-256 over the DER bytes, hex-encoded) and pins it in a
//! small JSON store. The first connect to a worker records the presented
//! fingerprint; every later connect rejects a certificate whose fingerprint
//! differs from the recorded pin.
//!
//! Store layout (host side):
//! ```text
//! {app_data_root}/workers/trusted-keys.json
//! ```
//! JSON shape: `{ "version": 1, "keys": { "<worker-id>": "<hex-fingerprint>" } }`

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::WebPkiSupportedAlgorithms;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::Error;

/// Environment variable that disables pinning (dev/test escape hatch).
///
/// Set to `"1"` or `"true"` to accept any presented certificate without
/// recording a pin. Defaults to off; never enable in production.
pub const INSECURE_SKIP_PINNING_ENV: &str = "REIMAGINE_INSECURE_SKIP_PINNING";

/// On-disk store format version.
const STORE_VERSION: u32 = 1;

/// SHA-256 fingerprint of a DER-encoded certificate, hex-encoded
/// (64 lowercase hex characters).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CertFingerprint(pub String);

impl std::fmt::Display for CertFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl CertFingerprint {
    /// Derive the SHA-256 fingerprint of a DER-encoded certificate.
    #[must_use]
    pub fn of_der(der: &CertificateDer<'_>) -> Self {
        let digest = sha2::Sha256::digest(der.as_ref());
        Self(hex::encode(digest))
    }

    /// The hex string form of this fingerprint.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// `true` when [`INSECURE_SKIP_PINNING_ENV`] is set to a truthy value.
#[must_use]
pub fn insecure_skip_pinning_from_env() -> bool {
    match std::env::var(INSECURE_SKIP_PINNING_ENV) {
        Ok(value) => value == "1" || value.eq_ignore_ascii_case("true"),
        Err(_) => false,
    }
}

/// On-disk representation of the trusted-key store.
#[derive(Debug, Serialize, Deserialize)]
struct TrustedKeysFile {
    version: u32,
    keys: BTreeMap<String, CertFingerprint>,
}

/// Persistent store mapping worker identities to pinned certificate
/// fingerprints.
///
/// All mutations persist atomically (write temp file, rename). A missing
/// file is treated as an empty store.
#[derive(Debug)]
pub struct TrustedKeyStore {
    path: PathBuf,
    keys: BTreeMap<String, CertFingerprint>,
}

impl TrustedKeyStore {
    /// Load the store from `path`; a missing file yields an empty store.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, Error> {
        let path = path.into();
        let keys = match std::fs::read(&path) {
            Ok(bytes) => {
                let file: TrustedKeysFile = serde_json::from_slice(&bytes).map_err(|e| {
                    Error::Store(format!("corrupt trusted-keys file {}: {e}", path.display()))
                })?;
                if file.version != STORE_VERSION {
                    return Err(Error::Store(format!(
                        "unsupported trusted-keys format version {} in {}",
                        file.version,
                        path.display()
                    )));
                }
                file.keys
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => {
                return Err(Error::Store(format!(
                    "failed to read trusted-keys file {}: {e}",
                    path.display()
                )));
            }
        };
        Ok(Self { path, keys })
    }

    /// The file path backing this store.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The fingerprint pinned for `worker_id`, if any.
    #[must_use]
    pub fn pinned(&self, worker_id: &str) -> Option<&CertFingerprint> {
        self.keys.get(worker_id)
    }

    /// Number of pinned keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the store holds no pins.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Record or replace the pin for `worker_id` and persist.
    ///
    /// Callers that raced a first-connect must re-check [`Self::pinned`]
    /// under the same lock before calling this; a concurrent first-trust
    /// that recorded a different fingerprint for the same id would
    /// otherwise be silently overwritten (last-writer-wins).
    pub fn trust(&mut self, worker_id: &str, fingerprint: &CertFingerprint) -> Result<(), Error> {
        self.keys.insert(worker_id.to_owned(), fingerprint.clone());
        self.persist()
    }

    /// Remove the pin for `worker_id` and persist.
    ///
    /// Returns whether a pin existed for the id.
    pub fn forget(&mut self, worker_id: &str) -> Result<bool, Error> {
        let removed = self.keys.remove(worker_id).is_some();
        if removed {
            self.persist()?;
        }
        Ok(removed)
    }

    /// Remove every pin and persist. Returns the number of pins removed.
    pub fn forget_all(&mut self) -> Result<usize, Error> {
        let count = self.keys.len();
        if count > 0 {
            self.keys.clear();
            self.persist()?;
        }
        Ok(count)
    }

    /// Atomically persist the store to disk.
    fn persist(&self) -> Result<(), Error> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Store(format!("failed to create {}: {e}", parent.display())))?;
        }
        let file = TrustedKeysFile {
            version: STORE_VERSION,
            keys: self.keys.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&file)
            .map_err(|e| Error::Store(format!("serialize trusted-keys: {e}")))?;
        // Unique temp name + O_EXCL: a fixed name would let a second
        // principal race a symlink into the store directory; the parent
        // dir is user-owned today, but harden the window anyway.
        let tmp = self.path.with_file_name(format!(
            "{}.{}.tmp",
            self.path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "trusted-keys".to_owned()),
            std::process::id(),
        ));
        {
            let mut handle = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| Error::Store(format!("failed to create {}: {e}", tmp.display())))?;
            handle
                .write_all(&bytes)
                .map_err(|e| Error::Store(format!("failed to write {}: {e}", tmp.display())))?;
            handle
                .sync_all()
                .map_err(|e| Error::Store(format!("failed to sync {}: {e}", tmp.display())))?;
        }
        std::fs::rename(&tmp, &self.path)
            .map_err(|e| Error::Store(format!("failed to rename {}: {e}", tmp.display())))?;
        if let Some(parent) = self.path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
        Ok(())
    }
}

/// rustls server-cert verifier that pins the peer by fingerprint.
///
/// `expected == None` is the TOFU mode: any certificate is accepted and its
/// fingerprint is recorded in `captured`. `expected == Some(pin)` rejects a
/// certificate whose fingerprint differs from the pin.
#[derive(Debug)]
pub struct FingerprintServerVerifier {
    algorithms: WebPkiSupportedAlgorithms,
    expected: Option<CertFingerprint>,
    captured: Arc<Mutex<Option<CertFingerprint>>>,
}

impl FingerprintServerVerifier {
    /// Create a verifier expecting `expected` (or recording when `None`).
    #[must_use]
    pub fn new(
        expected: Option<CertFingerprint>,
        captured: Arc<Mutex<Option<CertFingerprint>>>,
    ) -> Self {
        Self {
            algorithms: rustls::crypto::ring::default_provider().signature_verification_algorithms,
            expected,
            captured,
        }
    }

    /// The fingerprint recorded by the most recent verification, if any.
    #[must_use]
    pub fn captured(&self) -> Option<CertFingerprint> {
        self.captured.lock().ok().and_then(|guard| guard.clone())
    }
}

/// A `std::error::Error` wrapper for a fingerprint mismatch.
#[derive(Debug)]
struct FingerprintMismatch(String);

impl std::fmt::Display for FingerprintMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FingerprintMismatch {}

impl ServerCertVerifier for FingerprintServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let fingerprint = CertFingerprint::of_der(end_entity);
        if let Ok(mut guard) = self.captured.lock() {
            *guard = Some(fingerprint.clone());
        }
        if let Some(expected) = &self.expected
            && expected != &fingerprint
        {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::Other(rustls::OtherError(Arc::new(FingerprintMismatch(
                    format!(
                        "certificate fingerprint {fingerprint} does not match pinned {expected}"
                    ),
                )))),
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use rustls::pki_types::PrivateKeyDer;

    fn sample_cert() -> (CertificateDer<'static>, PrivateKeyDer<'static>) {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
        (
            CertificateDer::from(cert.cert),
            PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into()),
        )
    }

    #[test]
    fn fingerprint_is_stable_and_distinct() {
        let (der, _key) = sample_cert();
        let fp1 = CertFingerprint::of_der(&der);
        let fp2 = CertFingerprint::of_der(&der);
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.0.len(), 64);
        assert!(fp1.0.chars().all(|c| c.is_ascii_hexdigit()));

        let (der2, _key) = sample_cert();
        let fp3 = CertFingerprint::of_der(&der2);
        assert_ne!(fp1, fp3);
    }

    #[test]
    fn store_roundtrip_and_forget() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trusted-keys.json");
        let (der, _key) = sample_cert();
        let fp = CertFingerprint::of_der(&der);

        let mut store = TrustedKeyStore::load(&path).unwrap();
        assert!(store.is_empty());
        store.trust("worker-a", &fp).unwrap();

        let reloaded = TrustedKeyStore::load(&path).unwrap();
        assert_eq!(reloaded.pinned("worker-a"), Some(&fp));
        assert_eq!(reloaded.len(), 1);
        assert!(path.exists());

        let mut reloaded = reloaded;
        assert!(reloaded.forget("worker-a").unwrap());
        assert!(!reloaded.forget("worker-a").unwrap());
        assert!(TrustedKeyStore::load(&path).unwrap().is_empty());
    }

    #[test]
    fn store_forget_all_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trusted-keys.json");
        let (der, _key) = sample_cert();
        let fp = CertFingerprint::of_der(&der);

        let mut store = TrustedKeyStore::load(&path).unwrap();
        store.trust("a", &fp).unwrap();
        store.trust("b", &fp).unwrap();
        assert_eq!(store.forget_all().unwrap(), 2);
        assert!(TrustedKeyStore::load(&path).unwrap().is_empty());
    }

    #[test]
    fn corrupt_store_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trusted-keys.json");
        std::fs::write(&path, b"not json").unwrap();
        assert!(matches!(TrustedKeyStore::load(&path), Err(Error::Store(_))));
    }

    #[test]
    fn insecure_skip_env_parsing() {
        unsafe {
            std::env::set_var(INSECURE_SKIP_PINNING_ENV, "1");
        }
        assert!(insecure_skip_pinning_from_env());
        unsafe {
            std::env::set_var(INSECURE_SKIP_PINNING_ENV, "true");
        }
        assert!(insecure_skip_pinning_from_env());
        unsafe {
            std::env::set_var(INSECURE_SKIP_PINNING_ENV, "0");
        }
        assert!(!insecure_skip_pinning_from_env());
        unsafe {
            std::env::remove_var(INSECURE_SKIP_PINNING_ENV);
        }
        assert!(!insecure_skip_pinning_from_env());
    }

    #[test]
    fn verifier_records_and_pins() {
        let (der, _key) = sample_cert();
        let fp = CertFingerprint::of_der(&der);

        // TOFU mode records the presented fingerprint.
        let captured = Arc::new(Mutex::new(None));
        let verifier = FingerprintServerVerifier::new(None, Arc::clone(&captured));
        verifier
            .verify_server_cert(
                &der,
                &[],
                &"localhost".try_into().unwrap(),
                &[],
                UnixTime::now(),
            )
            .unwrap();
        assert_eq!(verifier.captured(), Some(fp.clone()));

        // Pin mode accepts a matching certificate.
        let captured = Arc::new(Mutex::new(None));
        let verifier = FingerprintServerVerifier::new(Some(fp.clone()), captured);
        verifier
            .verify_server_cert(
                &der,
                &[],
                &"localhost".try_into().unwrap(),
                &[],
                UnixTime::now(),
            )
            .unwrap();

        // Pin mode rejects a different certificate.
        let (der2, _key) = sample_cert();
        let captured = Arc::new(Mutex::new(None));
        let verifier = FingerprintServerVerifier::new(Some(fp), captured);
        assert!(
            verifier
                .verify_server_cert(
                    &der2,
                    &[],
                    &"localhost".try_into().unwrap(),
                    &[],
                    UnixTime::now()
                )
                .is_err()
        );
    }
}
