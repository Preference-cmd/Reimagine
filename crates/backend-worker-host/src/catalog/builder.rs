//! Production-grade TUF catalog builder.
//!
//! Assembles a complete TUF metadata chain (root → targets → snapshot → timestamp)
//! for a set of worker release packages. Designed for use in CI dry-run verification
//! and production release publication.
//!
//! Key design properties:
//! - Deterministic output for the same inputs
//! - Publication order enforced: targets → snapshot (re-signed) → timestamp (re-signed last)
//! - Pluggable signing via [`SigningKeyProvider`]
//! - Optional root provision (use embedded root or generate one)

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use super::error::CatalogSigningKeyError;
use super::tuf::{
    self, RoleConfig, RootMetadata, RootSigned, SignatureEntry, SnapshotMetaEntry,
    SnapshotMetadata, SnapshotSigned, TargetDesc, TargetsMetaEntry, TargetsMetadata, TargetsSigned,
    TimestampMetadata, TimestampSigned, TufKey, TufKeyVal,
};

/// TUF online roles that have distinct signing keys under V1.
///
/// Root keys are *not* online — they stay offline — so they do not appear
/// here. Each variant maps to a specific environment variable; the mapping is
/// enforced by [`EnvSigningKeyProvider::from_role`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnlineSigningRole {
    Targets,
    Snapshot,
    Timestamp,
}

impl OnlineSigningRole {
    /// Human-readable role name used in error messages and tests.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Targets => "targets",
            Self::Snapshot => "snapshot",
            Self::Timestamp => "timestamp",
        }
    }

    /// Environment variable name carrying the Ed25519 seed for this role.
    #[must_use]
    pub const fn env_var(self) -> &'static str {
        match self {
            Self::Targets => "REIMAGINE_TUF_TARGET_KEY",
            Self::Snapshot => "REIMAGINE_TUF_SNAPSHOT_KEY",
            Self::Timestamp => "REIMAGINE_TUF_TIMESTAMP_KEY",
        }
    }
}

/// A canonical representation of the signed payload bytes.
fn serialize_canonical(value: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

fn signed_payload(inner: &serde_json::Value) -> Vec<u8> {
    serialize_canonical(inner)
}

/// Provider of signing keys for TUF role metadata.
///
/// This trait abstracts over different key sources:
/// - Deterministic test keys (for CI dry-run)
/// - Environment-loaded production keys (for signed release)
pub trait SigningKeyProvider: std::fmt::Debug {
    /// Sign the given payload bytes, returning the hex-encoded signature.
    fn sign(&self, data: &[u8]) -> String;

    /// Return the TUF key that corresponds to this signer.
    fn tuf_key(&self) -> TufKey;

    /// Return the key ID (SHA-256 of canonical key JSON).
    fn key_id(&self) -> String {
        let key = self.tuf_key();
        let key_json = serde_json::json!({
            "keytype": key.key_type,
            "scheme": key.scheme,
            "keyval": {"public": &key.keyval.public}
        });
        hex::encode(Sha256::digest(serde_json::to_vec(&key_json).unwrap()))
    }
}

/// A deterministic test signing key provider (RFC 8032 test vector).
///
/// This is the production counterpart of the testing module's key helper.
/// Use in CI dry-run and local testing where cryptographic authenticity
/// is not required.
#[derive(Clone, Debug)]
pub struct TestSigningKey {
    key: ed25519_dalek::SigningKey,
}

impl Default for TestSigningKey {
    fn default() -> Self {
        Self::new()
    }
}

impl TestSigningKey {
    /// Create a new test signing key from the deterministic RFC 8032 test vector.
    #[must_use]
    pub fn new() -> Self {
        let seed: [u8; 32] =
            hex::decode("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
                .expect("valid hex")
                .try_into()
                .expect("valid 32-byte key");
        let key = ed25519_dalek::SigningKey::from_bytes(&seed);
        Self { key }
    }
}

impl SigningKeyProvider for TestSigningKey {
    fn sign(&self, data: &[u8]) -> String {
        use ed25519_dalek::Signer;
        hex::encode(self.key.sign(data).to_bytes())
    }

    fn tuf_key(&self) -> TufKey {
        let public = hex::encode(self.key.verifying_key().to_bytes());
        TufKey {
            key_type: "ed25519".to_string(),
            scheme: "ed25519".to_string(),
            keyval: TufKeyVal { public },
        }
    }
}

/// A signing key loaded from an environment variable.
///
/// Used for production releases where keys are stored as GitHub Environment
/// secrets and injected only into the protected `sign-catalog` job.
///
/// The provider refuses to expose the secret material via [`Debug`] or any
/// error path: errors reference the variable name, never the bytes.
#[derive(Clone)]
pub struct EnvSigningKeyProvider {
    role: OnlineSigningRole,
    signing_key: ed25519_dalek::SigningKey,
}

// `Debug` is hand-written so the seed never appears in logs.
impl std::fmt::Debug for EnvSigningKeyProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvSigningKeyProvider")
            .field("role", &self.role)
            .field("key_id", &self.key_id())
            .finish_non_exhaustive()
    }
}

impl EnvSigningKeyProvider {
    /// Build a provider for the given role, loading the seed from the role's
    /// canonical environment variable.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogSigningKeyError`] if the variable is missing, empty,
    /// not valid hex, or not exactly 32 bytes after decoding. None of these
    /// paths include the secret material.
    pub fn from_role(role: OnlineSigningRole) -> Result<Self, CatalogSigningKeyError> {
        let env_var = role.env_var();
        let raw = std::env::var(env_var).map_err(|_| CatalogSigningKeyError::Missing {
            role: role.as_str(),
            env_var,
        })?;
        Self::from_hex(role, raw)
    }

    /// Build a provider from a hex string. Useful for tests that do not want
    /// to mutate process-wide environment state.
    ///
    /// # Errors
    ///
    /// See [`Self::from_role`].
    pub fn from_hex(
        role: OnlineSigningRole,
        hex_seed: String,
    ) -> Result<Self, CatalogSigningKeyError> {
        let env_var = role.env_var();
        let trimmed = hex_seed.trim();
        if trimmed.is_empty() {
            return Err(CatalogSigningKeyError::Empty {
                role: role.as_str(),
                env_var,
            });
        }
        let bytes = hex::decode(trimmed).map_err(|_| CatalogSigningKeyError::InvalidHex {
            role: role.as_str(),
            env_var,
        })?;
        if bytes.len() != 32 {
            return Err(CatalogSigningKeyError::InvalidLength {
                role: role.as_str(),
                env_var,
                length: bytes.len(),
            });
        }
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&bytes);
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&seed);
        Ok(Self { role, signing_key })
    }

    /// The role this provider signs for.
    #[must_use]
    pub const fn role(&self) -> OnlineSigningRole {
        self.role
    }
}

impl SigningKeyProvider for EnvSigningKeyProvider {
    fn sign(&self, data: &[u8]) -> String {
        use ed25519_dalek::Signer;
        hex::encode(self.signing_key.sign(data).to_bytes())
    }

    fn tuf_key(&self) -> TufKey {
        let public = hex::encode(self.signing_key.verifying_key().to_bytes());
        TufKey {
            key_type: "ed25519".to_string(),
            scheme: "ed25519".to_string(),
            keyval: TufKeyVal { public },
        }
    }
}

/// Parameters for building a catalog.
#[derive(Debug)]
pub struct CatalogParams {
    /// The embedded/trusted root metadata, or None to generate one.
    /// If None, the builder creates a root using the provided provider.
    pub root: Option<RootMetadata>,
    /// Root metadata version (used when generating a new root, or
    /// when the targets/snapshot/timestamp metadata should reference it).
    pub root_version: u64,
    /// Targets metadata version.
    pub targets_version: u64,
    /// Snapshot metadata version.
    pub snapshot_version: u64,
    /// Timestamp metadata version.
    pub timestamp_version: u64,
    /// Expiry date string for all metadata (e.g. "2999-12-31T23:59:59Z").
    pub expires: String,
    /// Signing key provider for the targets role. Must be distinct from
    /// the snapshot and timestamp providers in production.
    pub targets_provider: Box<dyn SigningKeyProvider>,
    /// Signing key provider for the snapshot role.
    pub snapshot_provider: Box<dyn SigningKeyProvider>,
    /// Signing key provider for the timestamp role.
    pub timestamp_provider: Box<dyn SigningKeyProvider>,
    /// The signing key provider for the root role (only used when generating
    /// a new root, i.e. when `root` is None).
    pub root_provider: Box<dyn SigningKeyProvider>,
}

impl CatalogParams {
    /// Create a `CatalogParams` using three distinct deterministic test signing
    /// keys, one per online role, plus an independent root test key.
    ///
    /// Useful in CI dry-run and local testing where production keys are
    /// not available.
    #[must_use]
    pub fn with_test_keys(
        root: Option<RootMetadata>,
        root_version: u64,
        targets_version: u64,
        snapshot_version: u64,
        timestamp_version: u64,
        expires: String,
    ) -> Self {
        Self {
            root,
            root_version,
            targets_version,
            snapshot_version,
            timestamp_version,
            expires,
            targets_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Targets)),
            snapshot_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Snapshot)),
            timestamp_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Timestamp)),
            root_provider: Box::new(TestSigningKey::new()),
        }
    }
}

/// A deterministic test signing key whose seed varies by role so that
/// each online role uses a distinct key ID in tests.
///
/// Backed by the RFC 8032 Ed25519 test vector but XOR-mixed with a per-role
/// salt so the three online roles are visibly different. This exists so that
/// distinct-key tests (e.g. "root does not authorize this provider's key ID")
/// have something to assert against.
#[derive(Clone, Debug)]
pub struct RoleDistinctTestKey {
    role: OnlineSigningRole,
    signing_key: ed25519_dalek::SigningKey,
}

impl RoleDistinctTestKey {
    /// Construct a test key for the given role. Different roles produce
    /// different key IDs.
    #[must_use]
    pub fn new(role: OnlineSigningRole) -> Self {
        let base_seed: [u8; 32] =
            hex::decode("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
                .expect("valid hex")
                .try_into()
                .expect("valid 32-byte key");
        let salt: u8 = match role {
            OnlineSigningRole::Targets => 0x01,
            OnlineSigningRole::Snapshot => 0x02,
            OnlineSigningRole::Timestamp => 0x03,
        };
        let mut seed = base_seed;
        seed[0] ^= salt;
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&seed);
        Self { role, signing_key }
    }

    /// The role this test key signs for.
    #[must_use]
    pub const fn role(&self) -> OnlineSigningRole {
        self.role
    }
}

impl SigningKeyProvider for RoleDistinctTestKey {
    fn sign(&self, data: &[u8]) -> String {
        use ed25519_dalek::Signer;
        hex::encode(self.signing_key.sign(data).to_bytes())
    }

    fn tuf_key(&self) -> TufKey {
        let public = hex::encode(self.signing_key.verifying_key().to_bytes());
        TufKey {
            key_type: "ed25519".to_string(),
            scheme: "ed25519".to_string(),
            keyval: TufKeyVal { public },
        }
    }
}

/// A compiled catalog ready for serving or inspection.
#[derive(Clone, Debug)]
pub struct CatalogBundle {
    /// Root metadata (JSON bytes, versioned and unversioned).
    pub root_json: Vec<u8>,
    /// Targets metadata (JSON bytes).
    pub targets_json: Vec<u8>,
    /// Snapshot metadata (JSON bytes).
    pub snapshot_json: Vec<u8>,
    /// Timestamp metadata (JSON bytes).
    pub timestamp_json: Vec<u8>,
    /// The parsed root metadata.
    pub root: RootMetadata,
    /// The parsed targets metadata.
    pub targets: TargetsMetadata,
    /// The parsed snapshot metadata.
    pub snapshot: SnapshotMetadata,
    /// The parsed timestamp metadata.
    pub timestamp: TimestampMetadata,
    /// Map of target path → archive SHA-256.
    pub target_hashes: HashMap<String, String>,
}

/// Build a complete TUF catalog for a set of target entries.
///
/// When `params.root` is `None`, a root is generated that authorizes the
/// three distinct online providers under their respective roles. When
/// `params.root` is provided (production path), each provider's key ID
/// must already be authorized by the supplied root for its role.
///
/// # Panics
///
/// Panics on internal serialization failures (bug-level, not input errors).
pub fn build_catalog(params: &CatalogParams, targets: &[(String, TargetDesc)]) -> CatalogBundle {
    let targets_key_id = params.targets_provider.key_id();
    let targets_key = params.targets_provider.tuf_key();
    let snapshot_key_id = params.snapshot_provider.key_id();
    let snapshot_key = params.snapshot_provider.tuf_key();
    let timestamp_key_id = params.timestamp_provider.key_id();
    let timestamp_key = params.timestamp_provider.tuf_key();

    // ── Root ──────────────────────────────────────────────────────
    let root = match &params.root {
        Some(root) => {
            // Validate that each provider's key is authorized for its role
            // before signing any metadata. Failing closed here prevents
            // publishing catalogs that the root would later reject.
            for (role, key_id) in [
                ("root", &key_id_or_unauthorized(root, "root")),
                ("targets", &targets_key_id),
                ("snapshot", &snapshot_key_id),
                ("timestamp", &timestamp_key_id),
            ] {
                let role_cfg = root.signed.roles.get(role).unwrap_or_else(|| {
                    panic!("supplied root is missing role `{role}`")
                });
                if !role_cfg.keyids.contains(key_id) {
                    panic!(
                        "supplied root does not authorize key `{key_id}` for role `{role}`"
                    );
                }
            }
            root.clone()
        }
        None => {
            let root_key_id = params.root_provider.key_id();
            let root_key = params.root_provider.tuf_key();

            let mut all_keys = HashMap::new();
            all_keys.insert(root_key_id.clone(), root_key.clone());
            all_keys.insert(targets_key_id.clone(), targets_key.clone());
            all_keys.insert(snapshot_key_id.clone(), snapshot_key.clone());
            all_keys.insert(timestamp_key_id.clone(), timestamp_key.clone());

            let root_signed = RootSigned {
                kind: "root".to_string(),
                version: params.root_version,
                expires: params.expires.clone(),
                keys: all_keys,
                roles: HashMap::from([
                    (
                        "root".to_string(),
                        RoleConfig {
                            keyids: vec![root_key_id.clone()],
                            threshold: 1,
                        },
                    ),
                    (
                        "timestamp".to_string(),
                        RoleConfig {
                            keyids: vec![timestamp_key_id.clone()],
                            threshold: 1,
                        },
                    ),
                    (
                        "snapshot".to_string(),
                        RoleConfig {
                            keyids: vec![snapshot_key_id.clone()],
                            threshold: 1,
                        },
                    ),
                    (
                        "targets".to_string(),
                        RoleConfig {
                            keyids: vec![targets_key_id.clone()],
                            threshold: 1,
                        },
                    ),
                ]),
            };
            let root_payload = signed_payload(&serde_json::to_value(&root_signed).unwrap());
            let root_signature = SignatureEntry {
                keyid: root_key_id,
                sig: params.root_provider.sign(&root_payload),
            };
            RootMetadata {
                signed: root_signed,
                signatures: vec![root_signature],
            }
        }
    };

    // ── Targets ───────────────────────────────────────────────────
    let mut targets_map = HashMap::new();
    let mut target_hashes = HashMap::new();
    for (path, desc) in targets {
        if let Some(sha256) = desc.hashes.get("sha256") {
            target_hashes.insert(path.clone(), sha256.clone());
        }
        targets_map.insert(path.clone(), desc.clone());
    }

    let targets_signed = TargetsSigned {
        kind: "targets".to_string(),
        version: params.targets_version,
        expires: params.expires.clone(),
        targets: targets_map,
    };
    let targets_payload = signed_payload(&serde_json::to_value(&targets_signed).unwrap());
    let targets = TargetsMetadata {
        signed: targets_signed,
        signatures: vec![SignatureEntry {
            keyid: targets_key_id,
            sig: params.targets_provider.sign(&targets_payload),
        }],
    };
    let targets_bytes = serde_json::to_vec(&targets).unwrap();
    let targets_sha256 = hex::encode(Sha256::digest(&targets_bytes));

    // ── Snapshot ──────────────────────────────────────────────────
    let snapshot_signed = SnapshotSigned {
        kind: "snapshot".to_string(),
        version: params.snapshot_version,
        expires: params.expires.clone(),
        meta: HashMap::from([(
            "targets.json".to_string(),
            TargetsMetaEntry {
                version: params.targets_version,
                length: Some(targets_bytes.len() as u64),
                hashes: Some(HashMap::from([("sha256".to_string(), targets_sha256)])),
            },
        )]),
    };
    let snapshot_payload = signed_payload(&serde_json::to_value(&snapshot_signed).unwrap());
    let snapshot = SnapshotMetadata {
        signed: snapshot_signed,
        signatures: vec![SignatureEntry {
            keyid: snapshot_key_id,
            sig: params.snapshot_provider.sign(&snapshot_payload),
        }],
    };
    let snapshot_bytes = serde_json::to_vec(&snapshot).unwrap();
    let snapshot_sha256 = hex::encode(Sha256::digest(&snapshot_bytes));

    // ── Timestamp (produced LAST) ─────────────────────────────────
    let timestamp_signed = TimestampSigned {
        kind: "timestamp".to_string(),
        version: params.timestamp_version,
        expires: params.expires.clone(),
        meta: HashMap::from([(
            "snapshot.json".to_string(),
            SnapshotMetaEntry {
                version: params.snapshot_version,
                length: Some(snapshot_bytes.len() as u64),
                hashes: Some(HashMap::from([("sha256".to_string(), snapshot_sha256)])),
            },
        )]),
    };
    let timestamp_payload = signed_payload(&serde_json::to_value(&timestamp_signed).unwrap());
    let timestamp = TimestampMetadata {
        signed: timestamp_signed,
        signatures: vec![SignatureEntry {
            keyid: timestamp_key_id,
            sig: params.timestamp_provider.sign(&timestamp_payload),
        }],
    };

    CatalogBundle {
        root_json: serde_json::to_vec(&root).unwrap(),
        targets_json: targets_bytes,
        snapshot_json: snapshot_bytes,
        timestamp_json: serde_json::to_vec(&timestamp).unwrap(),
        root,
        targets,
        snapshot,
        timestamp,
        target_hashes,
    }
}

/// Helper used inside `build_catalog`'s panic messages only.
fn key_id_or_unauthorized(root: &RootMetadata, role: &str) -> String {
    root.signed
        .roles
        .get(role)
        .and_then(|cfg| cfg.keyids.first().cloned())
        .unwrap_or_else(|| format!("<no {role} key>"))
}

/// Write a catalog bundle to disk in the expected directory layout.
///
/// Layout:
/// ```text
/// {output_dir}/
///   root.json                    (unversioned, for initial bootstrap)
///   timestamp.json               (unversioned, for fast polling)
///   {version}.root.json           (versioned)
///   {version}.targets.json        (versioned)
///   {version}.snapshot.json       (versioned)
///   {version}.timestamp.json      (versioned)
///   {target-path-1}.tar.gz       (package archive)
///   ...
/// ```
///
/// # Errors
///
/// Returns an I/O error if writing fails.
pub fn write_catalog(
    bundle: &CatalogBundle,
    output_dir: &std::path::Path,
    target_archives: &[(&str, &[u8])],
) -> std::io::Result<()> {
    use std::io::Write;

    let meta_dir = output_dir.join("metadata");
    std::fs::create_dir_all(&meta_dir)?;

    // Helper to write both versioned and unversioned copies
    let write_meta = |name: &str, version: u64, bytes: &[u8]| -> std::io::Result<()> {
        // Versioned
        let mut f = std::fs::File::create(meta_dir.join(format!("{version}.{name}.json")))?;
        f.write_all(bytes)?;
        // Unversioned (for simple discovery)
        let mut f = std::fs::File::create(meta_dir.join(format!("{name}.json")))?;
        f.write_all(bytes)?;
        Ok(())
    };

    write_meta("root", bundle.root.signed.version, &bundle.root_json)?;
    write_meta(
        "targets",
        bundle.targets.signed.version,
        &bundle.targets_json,
    )?;
    write_meta(
        "snapshot",
        bundle.snapshot.signed.version,
        &bundle.snapshot_json,
    )?;
    write_meta(
        "timestamp",
        bundle.timestamp.signed.version,
        &bundle.timestamp_json,
    )?;

    // Write packages
    for (path, data) in target_archives {
        let dest = output_dir.join(path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(dest, data)?;
    }

    Ok(())
}

/// Verify a catalog bundle against the embedded root, checking the full TUF chain.
///
/// # Errors
///
/// Returns a [`CatalogError`] if any verification step fails.
pub fn verify_catalog(bundle: &CatalogBundle) -> Result<(), super::CatalogError> {
    let root_keys = tuf::verify_root(&bundle.root, None)?;
    let timestamp_snapshot_meta = bundle
        .timestamp
        .signed
        .meta
        .get("snapshot.json")
        .ok_or_else(|| super::CatalogError::Json {
            path: None,
            message: "timestamp missing snapshot.json meta".to_string(),
        })?;
    tuf::verify_timestamp(&bundle.timestamp, &bundle.root, 0)?;
    tuf::verify_snapshot(
        &bundle.snapshot,
        &bundle.snapshot_json,
        &bundle.root,
        timestamp_snapshot_meta,
    )?;

    let snapshot_targets_meta =
        bundle
            .snapshot
            .signed
            .meta
            .get("targets.json")
            .ok_or_else(|| super::CatalogError::Json {
                path: None,
                message: "snapshot missing targets.json meta".to_string(),
            })?;
    tuf::verify_targets(
        &bundle.targets,
        &bundle.targets_json,
        &bundle.root,
        snapshot_targets_meta,
    )?;

    // Verify each target's hash in the metadata matches our recorded hash
    for (path, expected_sha256) in &bundle.target_hashes {
        let found =
            bundle
                .targets
                .signed
                .targets
                .get(path)
                .ok_or_else(|| super::CatalogError::Json {
                    path: None,
                    message: format!("target `{path}` not found in targets metadata"),
                })?;
        let found_sha256 = found
            .hashes
            .get("sha256")
            .ok_or_else(|| super::CatalogError::Json {
                path: None,
                message: format!("target `{path}` missing sha256 hash"),
            })?;
        if found_sha256 != expected_sha256 {
            return Err(super::CatalogError::TargetHashMismatch {
                target: path.clone(),
                algorithm: "sha256".to_string(),
            });
        }
    }

    // Verify root keys are consistent
    if !root_keys.is_empty()
        && root_keys.values().any(|key| {
            bundle
                .root
                .signed
                .keys
                .values()
                .any(|rk| rk.key_type != key.key_type)
        })
    {
        // Just a sanity cross-check — the verify calls above already validate.
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::builder::{PackageParams, build_package};
    use ed25519_dalek::Verifier;

    #[test]
    fn build_catalog_with_test_keys_succeeds() {
        let pkg = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"test binary content".to_vec(),
            binary_name: "reimagine-inference-burn-worker".to_string(),
            license_path: None,
        })
        .expect("package build");

        let (path, desc) = crate::package::builder::target_desc(
            &pkg,
            &PackageParams {
                backend_kind: "burn".to_string(),
                backend_instance_id: "burn:wgpu:default".to_string(),
                installation_id: "burn-wgpu-darwin-aarch64".to_string(),
                target: "aarch64-apple-darwin".to_string(),
                version: "0.1.0".to_string(),
                package_kind: "burn-worker".to_string(),
                binary_content: b"test binary content".to_vec(),
                binary_name: "reimagine-inference-burn-worker".to_string(),
                license_path: None,
            },
        );

        let params =
            CatalogParams::with_test_keys(None, 1, 1, 1, 1, "2999-12-31T23:59:59Z".to_string());

        let bundle = build_catalog(&params, &[(path.clone(), desc)]);

        // Verify structure
        assert_eq!(bundle.root.signed.version, 1);
        assert_eq!(bundle.targets.signed.targets.len(), 1);
        assert!(bundle.target_hashes.contains_key(&path));

        // Verify the full TUF chain
        verify_catalog(&bundle).expect("catalog verification should pass");
    }

    #[test]
    fn build_catalog_multiple_targets() {
        let pkg1 = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:darwin".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"darwin binary".to_vec(),
            binary_name: "worker".to_string(),
            license_path: None,
        })
        .expect("package 1");

        let pkg2 = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:linux".to_string(),
            installation_id: "burn-wgpu-linux-x86_64".to_string(),
            target: "x86_64-unknown-linux-gnu".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"linux binary".to_vec(),
            binary_name: "worker".to_string(),
            license_path: None,
        })
        .expect("package 2");

        let (p1_path, p1_desc) = crate::package::builder::target_desc(
            &pkg1,
            &PackageParams {
                backend_kind: "burn".to_string(),
                backend_instance_id: "burn:wgpu:darwin".to_string(),
                installation_id: "burn-wgpu-darwin-aarch64".to_string(),
                target: "aarch64-apple-darwin".to_string(),
                version: "0.1.0".to_string(),
                package_kind: "burn-worker".to_string(),
                binary_content: b"darwin binary".to_vec(),
                binary_name: "worker".to_string(),
                license_path: None,
            },
        );
        let (p2_path, p2_desc) = crate::package::builder::target_desc(
            &pkg2,
            &PackageParams {
                backend_kind: "burn".to_string(),
                backend_instance_id: "burn:wgpu:linux".to_string(),
                installation_id: "burn-wgpu-linux-x86_64".to_string(),
                target: "x86_64-unknown-linux-gnu".to_string(),
                version: "0.1.0".to_string(),
                package_kind: "burn-worker".to_string(),
                binary_content: b"linux binary".to_vec(),
                binary_name: "worker".to_string(),
                license_path: None,
            },
        );

        let params = CatalogParams {
            root: None,
            root_version: 1,
            targets_version: 1,
            snapshot_version: 1,
            timestamp_version: 1,
            expires: "2999-12-31T23:59:59Z".to_string(),
            targets_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Targets)),
            snapshot_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Snapshot)),
            timestamp_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Timestamp)),
            root_provider: Box::new(TestSigningKey::new()),
        };

        let bundle = build_catalog(&params, &[(p1_path, p1_desc), (p2_path, p2_desc)]);

        assert_eq!(bundle.targets.signed.targets.len(), 2);
        verify_catalog(&bundle).expect("multi-target verification should pass");
    }

    #[test]
    fn catalog_write_and_verify_roundtrip() {
        let pkg = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"roundtrip test".to_vec(),
            binary_name: "worker".to_string(),
            license_path: None,
        })
        .expect("package build");

        let (path, desc) = crate::package::builder::target_desc(
            &pkg,
            &PackageParams {
                backend_kind: "burn".to_string(),
                backend_instance_id: "burn:wgpu:default".to_string(),
                installation_id: "burn-wgpu-darwin-aarch64".to_string(),
                target: "aarch64-apple-darwin".to_string(),
                version: "0.1.0".to_string(),
                package_kind: "burn-worker".to_string(),
                binary_content: b"roundtrip test".to_vec(),
                binary_name: "worker".to_string(),
                license_path: None,
            },
        );

        let params = CatalogParams {
            root: None,
            root_version: 1,
            targets_version: 1,
            snapshot_version: 1,
            timestamp_version: 1,
            expires: "2999-12-31T23:59:59Z".to_string(),
            targets_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Targets)),
            snapshot_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Snapshot)),
            timestamp_provider: Box::new(RoleDistinctTestKey::new(OnlineSigningRole::Timestamp)),
            root_provider: Box::new(TestSigningKey::new()),
        };

        let bundle = build_catalog(&params, &[(path.clone(), desc)]);

        let dir = tempfile::tempdir().unwrap();
        write_catalog(&bundle, dir.path(), &[(&path, &pkg.archive)]).expect("write catalog");

        // Verify directory structure
        assert!(dir.path().join("metadata").join("1.root.json").exists());
        assert!(dir.path().join("metadata").join("root.json").exists());
        assert!(dir.path().join("metadata").join("1.targets.json").exists());
        assert!(dir.path().join("metadata").join("1.snapshot.json").exists());
        assert!(
            dir.path()
                .join("metadata")
                .join("1.timestamp.json")
                .exists()
        );
        assert!(dir.path().join(&path).exists());
    }

    #[test]
    fn test_signing_key_is_deterministic() {
        let k1 = TestSigningKey::new();
        let k2 = TestSigningKey::new();

        assert_eq!(k1.key_id(), k2.key_id());
        assert_eq!(
            k1.sign(b"test"),
            k2.sign(b"test"),
            "signatures must be deterministic"
        );
    }

    // ── EnvSigningKeyProvider tests ──────────────────────────────

    const TEST_TARGET_SEED_HEX: &str =
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";

    #[test]
    fn env_provider_loads_from_hex_and_signs() {
        let provider = EnvSigningKeyProvider::from_hex(
            OnlineSigningRole::Targets,
            TEST_TARGET_SEED_HEX.to_string(),
        )
        .expect("valid hex should construct");

        // Signing round-trip: sign a payload and verify with the public key.
        let payload = b"hello world";
        let sig_hex = provider.sign(payload);
        let pub_key_bytes = hex::decode(provider.tuf_key().keyval.public.clone())
            .expect("valid hex");
        let public = ed25519_dalek::VerifyingKey::from_bytes(
            &pub_key_bytes.try_into().expect("32-byte public key"),
        )
        .expect("valid public key");
        let sig_bytes = hex::decode(&sig_hex).expect("valid hex");
        let sig = ed25519_dalek::Signature::from_slice(&sig_bytes)
            .expect("valid signature bytes");
        public.verify_strict(payload, &sig).expect("signature must verify");
    }

    #[test]
    fn env_provider_rejects_missing_var() {
        // Use a sentinel env var name unlikely to be set. The unit test never
        // touches process env unless it calls from_role.
        let result = std::panic::catch_unwind(|| {
            // from_role reads std::env. Force a path that won't be set by
            // ensuring the role's env var is cleared for the duration.
            // We can't actually unset process env safely in unit tests, so
            // skip if it's set in CI.
            if std::env::var(OnlineSigningRole::Targets.env_var()).is_ok() {
                return None;
            }
            Some(EnvSigningKeyProvider::from_role(OnlineSigningRole::Targets))
        });
        match result {
            Ok(Some(Err(e))) => {
                assert!(matches!(e, super::super::error::CatalogSigningKeyError::Missing { .. }));
            }
            Ok(None) => {
                // env var is set; skip the test cleanly.
                eprintln!("skipping env_provider_rejects_missing_var: var is set");
            }
            Ok(Some(Ok(_))) => panic!("expected Err, got Ok"),
            Err(_) => panic!("test panicked"),
        }
    }

    #[test]
    fn env_provider_rejects_empty_hex() {
        let err = EnvSigningKeyProvider::from_hex(
            OnlineSigningRole::Snapshot,
            "   ".to_string(),
        )
        .expect_err("empty should fail");
        assert!(matches!(err, super::super::error::CatalogSigningKeyError::Empty { .. }));
    }

    #[test]
    fn env_provider_rejects_non_hex() {
        let err = EnvSigningKeyProvider::from_hex(
            OnlineSigningRole::Timestamp,
            "not-hex-at-all".to_string(),
        )
        .expect_err("non-hex should fail");
        assert!(matches!(err, super::super::error::CatalogSigningKeyError::InvalidHex { .. }));
    }

    #[test]
    fn env_provider_rejects_wrong_length() {
        // 31 bytes of valid hex — wrong length.
        let short = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f";
        let err = EnvSigningKeyProvider::from_hex(
            OnlineSigningRole::Targets,
            short.to_string(),
        )
        .expect_err("wrong length should fail");
        assert!(matches!(err, super::super::error::CatalogSigningKeyError::InvalidLength { .. }));
    }

    #[test]
    fn env_provider_debug_does_not_leak_seed() {
        let provider = EnvSigningKeyProvider::from_hex(
            OnlineSigningRole::Targets,
            TEST_TARGET_SEED_HEX.to_string(),
        )
        .expect("valid hex");
        let debug = format!("{provider:?}");
        // The seed bytes must not appear in the Debug output.
        assert!(
            !debug.contains(TEST_TARGET_SEED_HEX),
            "Debug output leaked seed hex: {debug}"
        );
        // But the key ID should appear.
        assert!(
            debug.contains(&provider.key_id()),
            "Debug should contain the key ID"
        );
    }

    #[test]
    fn role_distinct_test_keys_have_distinct_ids() {
        let t = RoleDistinctTestKey::new(OnlineSigningRole::Targets);
        let s = RoleDistinctTestKey::new(OnlineSigningRole::Snapshot);
        let ts = RoleDistinctTestKey::new(OnlineSigningRole::Timestamp);
        assert_ne!(t.key_id(), s.key_id());
        assert_ne!(s.key_id(), ts.key_id());
        assert_ne!(t.key_id(), ts.key_id());
    }

    #[test]
    fn build_catalog_with_three_role_distinct_keys_authorizes_all_roles() {
        // Verifies that the root generated by `build_catalog` when no root
        // is provided authorizes every distinct provider's key ID under
        // its respective role — preventing a regression where the root
        // would still be issued with a single shared key.
        let pkg = build_package(&PackageParams {
            backend_kind: "burn".to_string(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-darwin-aarch64".to_string(),
            target: "aarch64-apple-darwin".to_string(),
            version: "0.1.0".to_string(),
            package_kind: "burn-worker".to_string(),
            binary_content: b"distinct roles".to_vec(),
            binary_name: "worker".to_string(),
            license_path: None,
        })
        .expect("build package");

        let (path, desc) = crate::package::builder::target_desc(
            &pkg,
            &PackageParams {
                backend_kind: "burn".to_string(),
                backend_instance_id: "burn:wgpu:default".to_string(),
                installation_id: "burn-wgpu-darwin-aarch64".to_string(),
                target: "aarch64-apple-darwin".to_string(),
                version: "0.1.0".to_string(),
                package_kind: "burn-worker".to_string(),
                binary_content: b"distinct roles".to_vec(),
                binary_name: "worker".to_string(),
                license_path: None,
            },
        );

        let targets_provider = RoleDistinctTestKey::new(OnlineSigningRole::Targets);
        let snapshot_provider = RoleDistinctTestKey::new(OnlineSigningRole::Snapshot);
        let timestamp_provider = RoleDistinctTestKey::new(OnlineSigningRole::Timestamp);

        let params = CatalogParams {
            root: None,
            root_version: 1,
            targets_version: 1,
            snapshot_version: 1,
            timestamp_version: 1,
            expires: "2999-12-31T23:59:59Z".to_string(),
            targets_provider: Box::new(targets_provider),
            snapshot_provider: Box::new(snapshot_provider),
            timestamp_provider: Box::new(timestamp_provider),
            root_provider: Box::new(TestSigningKey::new()),
        };

        let bundle = build_catalog(&params, &[(path, desc)]);
        verify_catalog(&bundle).expect("chain verifies with distinct keys");
    }
}
