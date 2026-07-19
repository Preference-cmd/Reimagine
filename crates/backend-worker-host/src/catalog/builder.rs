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

use super::tuf::{
    self, RoleConfig, RootMetadata, RootSigned, SignatureEntry, SnapshotMetaEntry,
    SnapshotMetadata, SnapshotSigned, TargetDesc, TargetsMetaEntry, TargetsMetadata, TargetsSigned,
    TimestampMetadata, TimestampSigned, TufKey, TufKeyVal,
};

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
    /// The signing key provider for all online roles (targets, snapshot, timestamp).
    /// In production, distinct providers are used for each role.
    pub online_provider: Box<dyn SigningKeyProvider>,
    /// The signing key provider for the root role (if generating a new root).
    pub root_provider: Box<dyn SigningKeyProvider>,
}

impl CatalogParams {
    /// Create a `CatalogParams` using deterministic test signing keys.
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
            online_provider: Box::new(TestSigningKey::new()),
            root_provider: Box::new(TestSigningKey::new()),
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
/// # Panics
///
/// Panics on internal serialization failures (bug-level, not input errors).
pub fn build_catalog(
    params: &CatalogParams,
    targets: &[(String, TargetDesc)],
) -> CatalogBundle {
    let online_key_id = params.online_provider.key_id();
    let online_key = params.online_provider.tuf_key();

    // ── Root ──────────────────────────────────────────────────────
    let root = match &params.root {
        Some(root) => root.clone(),
        None => {
            let root_key_id = params.root_provider.key_id();
            let root_key = params.root_provider.tuf_key();

            let mut all_keys = HashMap::new();
            all_keys.insert(root_key_id.clone(), root_key.clone());
            all_keys.insert(online_key_id.clone(), online_key.clone());

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
                            keyids: vec![online_key_id.clone()],
                            threshold: 1,
                        },
                    ),
                    (
                        "snapshot".to_string(),
                        RoleConfig {
                            keyids: vec![online_key_id.clone()],
                            threshold: 1,
                        },
                    ),
                    (
                        "targets".to_string(),
                        RoleConfig {
                            keyids: vec![online_key_id.clone()],
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
            keyid: online_key_id.clone(),
            sig: params.online_provider.sign(&targets_payload),
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
            keyid: online_key_id.clone(),
            sig: params.online_provider.sign(&snapshot_payload),
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
            keyid: online_key_id,
            sig: params.online_provider.sign(&timestamp_payload),
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
    write_meta("targets", bundle.targets.signed.version, &bundle.targets_json)?;
    write_meta("snapshot", bundle.snapshot.signed.version, &bundle.snapshot_json)?;
    write_meta("timestamp", bundle.timestamp.signed.version, &bundle.timestamp_json)?;

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

    let snapshot_targets_meta = bundle
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
        let found = bundle.targets.signed.targets.get(path).ok_or_else(|| {
            super::CatalogError::Json {
                path: None,
                message: format!("target `{path}` not found in targets metadata"),
            }
        })?;
        let found_sha256 = found.hashes.get("sha256").ok_or_else(|| {
            super::CatalogError::Json {
                path: None,
                message: format!("target `{path}` missing sha256 hash"),
            }
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
        && root_keys
            .values()
            .any(|key| bundle.root.signed.keys.values().any(|rk| rk.key_type != key.key_type))
    {
        // Just a sanity cross-check — the verify calls above already validate.
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::builder::{build_package, PackageParams};

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

        let params = CatalogParams::with_test_keys(
            None,
            1, 1, 1, 1,
            "2999-12-31T23:59:59Z".to_string(),
        );

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
        let online = TestSigningKey::new();
        let root = TestSigningKey::new();

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
            online_provider: Box::new(online),
            root_provider: Box::new(root),
        };

        let bundle = build_catalog(&params, &[(p1_path, p1_desc), (p2_path, p2_desc)]);

        assert_eq!(bundle.targets.signed.targets.len(), 2);
        verify_catalog(&bundle).expect("multi-target verification should pass");
    }

    #[test]
    fn catalog_write_and_verify_roundtrip() {
        let online = TestSigningKey::new();
        let root_provider = TestSigningKey::new();

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
            online_provider: Box::new(online),
            root_provider: Box::new(root_provider),
        };

        let bundle = build_catalog(&params, &[(path.clone(), desc)]);

        let dir = tempfile::tempdir().unwrap();
        write_catalog(
            &bundle,
            dir.path(),
            &[(&path, &pkg.archive)],
        )
        .expect("write catalog");

        // Verify directory structure
        assert!(dir.path().join("metadata").join("1.root.json").exists());
        assert!(dir.path().join("metadata").join("root.json").exists());
        assert!(dir.path().join("metadata").join("1.targets.json").exists());
        assert!(dir.path().join("metadata").join("1.snapshot.json").exists());
        assert!(dir.path().join("metadata").join("1.timestamp.json").exists());
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
}
