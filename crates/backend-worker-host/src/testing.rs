//! Deterministic fixtures for worker release pipeline testing.
//!
//! Provides:
//! - [`generate_package`] — build a deterministic tar.gz worker package
//! - [`generate_tuf_metadata`] — build a complete TUF root/targets/snapshot/timestamp chain
//! - [`generate_full_catalog`] — assemble both into a tempdir ready for local HTTP hosting
//!
//! All keys are deterministic (RFC 8032 test vector). Output is byte-identical
//! for the same input parameters.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ed25519_dalek::Signer;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

use crate::catalog::tuf::{
    self, RoleConfig, RootMetadata, RootSigned, SignatureEntry, SnapshotMetaEntry,
    SnapshotMetadata, SnapshotSigned, TargetDesc, TargetsMetaEntry, TargetsMetadata, TargetsSigned,
    TimestampMetadata, TimestampSigned, TufKey, TufKeyVal,
};
use crate::package::{ExtractionLimits, PackageExtractor, PackageFileEntry, PackageManifest};
use crate::{BackendInstanceId, ExpectedWorkerIdentity, WorkerInstallationId};

/// Deterministic test signing key (RFC 8032 test vector secret key).
pub fn test_signing_key() -> ed25519_dalek::SigningKey {
    let seed: [u8; 32] =
        hex::decode("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
            .expect("valid hex")
            .try_into()
            .expect("valid 32-byte key");
    ed25519_dalek::SigningKey::from_bytes(&seed)
}

/// The deterministic public key hex corresponding to `test_signing_key`.
pub fn test_public_key_hex() -> String {
    hex::encode(test_signing_key().verifying_key().to_bytes())
}

/// The deterministic key ID derived from the test key.
pub fn test_key_id() -> String {
    let key = tuf::TufKey {
        key_type: "ed25519".to_string(),
        scheme: "ed25519".to_string(),
        keyval: tuf::TufKeyVal {
            public: test_public_key_hex(),
        },
    };
    key_id(&key)
}

// ── Key helpers ──────────────────────────────────────────────────────

fn key_id(key: &TufKey) -> String {
    let key_json = serde_json::json!({
        "keytype": key.key_type,
        "scheme": key.scheme,
        "keyval": {"public": &key.keyval.public}
    });
    hex::encode(Sha256::digest(serde_json::to_vec(&key_json).unwrap()))
}

fn sign_hex(data: &[u8]) -> String {
    let key = test_signing_key();
    hex::encode(key.sign(data).to_bytes())
}

fn serialize_canonical(value: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

fn signed_payload(inner: &serde_json::Value) -> Vec<u8> {
    // The canonical form for signing is the `signed` object alone, per TUF spec.
    serialize_canonical(inner)
}

// ── Package fixture ──────────────────────────────────────────────────

/// Parameters for a deterministic worker package fixture.
#[derive(Clone, Debug)]
pub struct PackageFixtureParams {
    pub package_kind: String,
    pub binary_name: String,
    pub binary_content: Vec<u8>,
    pub backend_instance_id: String,
    pub installation_id: String,
    pub backend_kind: String,
    pub target: String,
    pub manifest_digest: String,
}

impl Default for PackageFixtureParams {
    fn default() -> Self {
        Self {
            package_kind: "burn-worker".to_string(),
            binary_name: "reimagine-burn-worker".to_string(),
            binary_content: b"#!/bin/sh\necho deterministic worker binary\0".to_vec(),
            backend_instance_id: "burn:wgpu:default".to_string(),
            installation_id: "burn-wgpu-v1".to_string(),
            backend_kind: "burn".to_string(),
            target: std::env::consts::ARCH.to_string(),
            manifest_digest: "test-manifest-0000".to_string(),
        }
    }
}

/// Generate a deterministic tar.gz worker package.
///
/// Returns the compressed archive bytes. The package contains:
/// - The binary at the root
/// - `package.json` manifest
/// - `LICENSE` placeholder
pub fn generate_package(params: &PackageFixtureParams) -> Vec<u8> {
    let binary_hash = hex::encode(Sha256::digest(&params.binary_content));
    let license_content = b"MIT License Placeholder\n";
    let license_hash = hex::encode(Sha256::digest(license_content));

    // Build file entries first so we can compute the serialized manifest size
    let mut file_entries = vec![
        PackageFileEntry {
            path: params.binary_name.clone(),
            sha256: binary_hash,
            size: params.binary_content.len() as u64,
            mode: 0o755,
            executable: true,
        },
        PackageFileEntry {
            path: "LICENSE".to_string(),
            sha256: license_hash,
            size: license_content.len() as u64,
            mode: 0o644,
            executable: false,
        },
    ];

    // Build a provisional manifest to compute the serialized size
    // NOTE: package.json is NOT included in files[] — the extractor
    // reads the manifest before extraction and skips it during extract.
    let mut manifest = PackageManifest {
        schema_version: 1,
        package_kind: params.package_kind.clone(),
        identity: ExpectedWorkerIdentity {
            backend_instance_id: BackendInstanceId(params.backend_instance_id.clone()),
            installation_id: WorkerInstallationId(params.installation_id.clone()),
            backend_kind: params.backend_kind.clone(),
            target: params.target.clone(),
            manifest_digest: params.manifest_digest.clone(),
        },
        files: file_entries.clone(),
        required_size: 0, // placeholder
        required_entries: 2, // binary + LICENSE
    };

    let manifest_json = serde_json::to_vec(&manifest).unwrap();

    // Rebuild manifest with correct required_size
    let total_size: u64 = file_entries.iter().map(|f| f.size).sum();
    manifest.required_size = total_size;

    // Re-serialize with final values
    let manifest_json = serde_json::to_vec(&manifest).unwrap();

    let mut buf = Vec::new();
    {
        let gz = GzEncoder::new(&mut buf, flate2::Compression::fast());
        let mut tar = tar::Builder::new(gz);

        // Add the binary
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o755);
        header.set_size(params.binary_content.len() as u64);
        tar.append_data(&mut header, &params.binary_name, &params.binary_content[..])
            .expect("append binary");

        // Add LICENSE
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o644);
        header.set_size(license_content.len() as u64);
        tar.append_data(&mut header, "LICENSE", &license_content[..])
            .expect("append LICENSE");

        // Add package.json manifest
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o644);
        header.set_size(manifest_json.len() as u64);
        tar.append_data(&mut header, "package.json", &manifest_json[..])
            .expect("append package.json");
    }

    buf
}

// ── TUF metadata fixture ─────────────────────────────────────────────

/// Parameters for a TUF metadata fixture.
#[derive(Clone, Debug)]
pub struct TufMetadataParams {
    pub root_version: u64,
    pub targets_version: u64,
    pub snapshot_version: u64,
    pub timestamp_version: u64,
    pub expires: String,
}

impl Default for TufMetadataParams {
    fn default() -> Self {
        Self {
            root_version: 1,
            targets_version: 1,
            snapshot_version: 1,
            timestamp_version: 1,
            expires: "2999-12-31T23:59:59Z".to_string(),
        }
    }
}

/// Generated TUF metadata fixture bundle.
#[derive(Debug)]
pub struct TufMetadataBundle {
    pub root: RootMetadata,
    pub targets: TargetsMetadata,
    pub snapshot: SnapshotMetadata,
    pub timestamp: TimestampMetadata,
    /// The key ID used for all roles.
    pub key_id: String,
    /// The TUF key.
    pub key: TufKey,
}

/// Generate a complete deterministic TUF metadata chain.
pub fn generate_tuf_metadata(params: &TufMetadataParams) -> TufMetadataBundle {
    let signing_key = test_signing_key();
    let verifying_key = signing_key.verifying_key();
    let key = TufKey {
        key_type: "ed25519".to_string(),
        scheme: "ed25519".to_string(),
        keyval: TufKeyVal {
            public: hex::encode(verifying_key.to_bytes()),
        },
    };
    let key_id = key_id(&key);

    // ── Root ──────────────────────────────────────────────────────
    let root_signed = RootSigned {
        kind: "root".to_string(),
        version: params.root_version,
        expires: params.expires.clone(),
        keys: HashMap::from([(key_id.clone(), key.clone())]),
        roles: HashMap::from([
            (
                "root".to_string(),
                RoleConfig {
                    keyids: vec![key_id.clone()],
                    threshold: 1,
                },
            ),
            (
                "timestamp".to_string(),
                RoleConfig {
                    keyids: vec![key_id.clone()],
                    threshold: 1,
                },
            ),
            (
                "snapshot".to_string(),
                RoleConfig {
                    keyids: vec![key_id.clone()],
                    threshold: 1,
                },
            ),
            (
                "targets".to_string(),
                RoleConfig {
                    keyids: vec![key_id.clone()],
                    threshold: 1,
                },
            ),
        ]),
    };
    let root_payload = signed_payload(&serde_json::to_value(&root_signed).unwrap());
    let root = RootMetadata {
        signed: root_signed,
        signatures: vec![SignatureEntry {
            keyid: key_id.clone(),
            sig: sign_hex(&root_payload),
        }],
    };

    // ── Targets ───────────────────────────────────────────────────
    let targets_signed = TargetsSigned {
        kind: "targets".to_string(),
        version: params.targets_version,
        expires: params.expires.clone(),
        targets: HashMap::new(), // caller fills in
    };
    let targets_payload = signed_payload(&serde_json::to_value(&targets_signed).unwrap());
    let targets = TargetsMetadata {
        signed: targets_signed,
        signatures: vec![SignatureEntry {
            keyid: key_id.clone(),
            sig: sign_hex(&targets_payload),
        }],
    };

    // ── Snapshot ──────────────────────────────────────────────────
    let snapshot_signed = SnapshotSigned {
        kind: "snapshot".to_string(),
        version: params.snapshot_version,
        expires: params.expires.clone(),
        meta: HashMap::from([(
            "targets.json".to_string(),
            TargetsMetaEntry {
                version: params.targets_version,
                length: None,
                hashes: None,
            },
        )]),
    };
    let snapshot_payload = signed_payload(&serde_json::to_value(&snapshot_signed).unwrap());
    let snapshot = SnapshotMetadata {
        signed: snapshot_signed,
        signatures: vec![SignatureEntry {
            keyid: key_id.clone(),
            sig: sign_hex(&snapshot_payload),
        }],
    };

    // ── Timestamp ─────────────────────────────────────────────────
    let timestamp_signed = TimestampSigned {
        kind: "timestamp".to_string(),
        version: params.timestamp_version,
        expires: params.expires.clone(),
        meta: HashMap::from([(
            "snapshot.json".to_string(),
            SnapshotMetaEntry {
                version: params.snapshot_version,
                length: None,
                hashes: None,
            },
        )]),
    };
    let timestamp_payload = signed_payload(&serde_json::to_value(&timestamp_signed).unwrap());
    let timestamp = TimestampMetadata {
        signed: timestamp_signed,
        signatures: vec![SignatureEntry {
            keyid: key_id.clone(),
            sig: sign_hex(&timestamp_payload),
        }],
    };

    TufMetadataBundle {
        root,
        targets,
        snapshot,
        timestamp,
        key_id,
        key,
    }
}

/// Add a target entry to `TargetsMetadata` targets map and re-sign.
pub fn add_target(
    mut targets: TargetsMetadata,
    path: &str,
    data: &[u8],
    custom: serde_json::Value,
) -> TargetsMetadata {
    let sha256 = hex::encode(Sha256::digest(data));
    targets
        .signed
        .targets
        .entry(path.to_string())
        .or_insert_with(|| TargetDesc {
            length: data.len() as u64,
            hashes: HashMap::from([("sha256".to_string(), sha256)]),
            custom: Some(custom),
        });
    // Re-sign
    let payload = signed_payload(&serde_json::to_value(&targets.signed).unwrap());
    let key_id_local = targets
        .signatures
        .first()
        .map(|s| s.keyid.clone())
        .unwrap_or_default();
    targets.signatures = vec![SignatureEntry {
        keyid: key_id_local,
        sig: sign_hex(&payload),
    }];
    targets
}

// ── Full catalog assembly ────────────────────────────────────────────

/// Assembled catalog ready for local serving or testing.
#[derive(Debug)]
pub struct CatalogFixture {
    /// Directory containing all catalog files (root.json, targets.json, etc.)
    pub catalog_dir: PathBuf,
    /// The generated TUF metadata bundle.
    pub metadata: TufMetadataBundle,
    /// Package paths stored in this catalog.
    pub package_paths: Vec<String>,
}

/// Generate a full catalog directory with TUF metadata and worker packages.
///
/// Returns the path to the catalog directory and the generated metadata.
pub fn generate_full_catalog(
    base_dir: &Path,
    tuf_params: &TufMetadataParams,
    package_params: &[PackageFixtureParams],
) -> CatalogFixture {
    let mut metadata = generate_tuf_metadata(tuf_params);
    let mut package_paths = Vec::new();

    for pkg in package_params {
        let archive = generate_package(pkg);
        let filename = format!(
            "reimagine-worker-{}-{}.tar.gz",
            pkg.backend_kind, pkg.target
        );

        // Write package file
        std::fs::write(base_dir.join(&filename), &archive).expect("write package");

        // Add to targets
        let custom = serde_json::json!({
            "version": "1.0.0-test",
            "installation_id": pkg.installation_id,
            "backend_instance_id": pkg.backend_instance_id,
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "worker_kind": pkg.backend_kind,
            "protocol_version_min": 1,
            "protocol_version_max": 1,
            "package_format": "tar.gz",
            "min_runtime_version": null,
            "target": pkg.target,
            "manifest_digest": pkg.manifest_digest,
        });
        metadata.targets = add_target(metadata.targets, &filename, &archive, custom);
        package_paths.push(filename);
    }

    // Re-sign snapshot and timestamp after targets change
    let targets_bytes = serde_json::to_vec(&metadata.targets).unwrap();
    let targets_sha256 = hex::encode(Sha256::digest(&targets_bytes));
    let targets_version = metadata.targets.signed.version;

    let snapshot_version = tuf_params.snapshot_version;
    metadata.snapshot.signed.meta.insert(
        "targets.json".to_string(),
        TargetsMetaEntry {
            version: targets_version,
            length: Some(targets_bytes.len() as u64),
            hashes: Some(HashMap::from([("sha256".to_string(), targets_sha256)])),
        },
    );
    let snapshot_payload = signed_payload(&serde_json::to_value(&metadata.snapshot.signed).unwrap());
    metadata.snapshot.signatures = vec![SignatureEntry {
        keyid: metadata.key_id.clone(),
        sig: sign_hex(&snapshot_payload),
    }];

    let snapshot_bytes = serde_json::to_vec(&metadata.snapshot).unwrap();
    let snapshot_sha256 = hex::encode(Sha256::digest(&snapshot_bytes));
    metadata.timestamp.signed.meta.insert(
        "snapshot.json".to_string(),
        SnapshotMetaEntry {
            version: snapshot_version,
            length: Some(snapshot_bytes.len() as u64),
            hashes: Some(HashMap::from([("sha256".to_string(), snapshot_sha256)])),
        },
    );
    let timestamp_payload =
        signed_payload(&serde_json::to_value(&metadata.timestamp.signed).unwrap());
    metadata.timestamp.signatures = vec![SignatureEntry {
        keyid: metadata.key_id.clone(),
        sig: sign_hex(&timestamp_payload),
    }];

    // Write metadata files
    let meta_dir = base_dir.join("metadata");
    std::fs::create_dir_all(&meta_dir).expect("create metadata dir");

    write_versioned_metadata(&meta_dir, "root", tuf_params.root_version, &metadata.root);
    write_versioned_metadata(
        &meta_dir,
        "targets",
        targets_version,
        &metadata.targets,
    );
    write_versioned_metadata(
        &meta_dir,
        "snapshot",
        tuf_params.snapshot_version,
        &metadata.snapshot,
    );
    write_versioned_metadata(
        &meta_dir,
        "timestamp",
        tuf_params.timestamp_version,
        &metadata.timestamp,
    );

    CatalogFixture {
        catalog_dir: base_dir.to_path_buf(),
        metadata,
        package_paths,
    }
}

fn write_versioned_metadata(
    dir: &Path,
    name: &str,
    version: u64,
    metadata: &impl serde::Serialize,
) {
    let bytes = serde_json::to_vec_pretty(metadata).expect("serialize metadata");
    // Versioned form: {version}.{name}.json  e.g. "1.snapshot.json"
    // (matching CatalogClient::meta_url which produces base/{version}.{name})
    std::fs::write(dir.join(format!("{version}.{name}.json")), &bytes)
        .expect("write versioned metadata file");
    // Also write unversioned copy for simple discovery (timestamp.json, root.json)
    std::fs::write(dir.join(format!("{name}.json")), &bytes)
        .expect("write unversioned metadata");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_catalog_assembles_without_error() {
        let dir = tempfile::tempdir().unwrap();
        let pkgs = vec![
            PackageFixtureParams {
                target: "aarch64-apple-darwin".to_string(),
                ..PackageFixtureParams::default()
            },
            PackageFixtureParams {
                backend_instance_id: "burn:flex:cpu".to_string(),
                installation_id: "burn-flex-cpu-v1".to_string(),
                backend_kind: "burn-flex".to_string(),
                target: "x86_64-unknown-linux-gnu".to_string(),
                ..PackageFixtureParams::default()
            },
        ];
        let catalog = generate_full_catalog(dir.path(), &TufMetadataParams::default(), &pkgs);

        assert!(catalog.catalog_dir.join("metadata/1.root.json").exists());
        assert!(catalog.catalog_dir.join("metadata/1.targets.json").exists());
        assert!(catalog.catalog_dir.join("metadata/1.snapshot.json").exists());
        assert!(catalog.catalog_dir.join("metadata/1.timestamp.json").exists());
        assert_eq!(catalog.package_paths.len(), 2);
        assert_eq!(catalog.metadata.targets.signed.targets.len(), 2);
    }

    #[test]
    fn package_extracts_and_matches_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let staging = dir.path().join("staging");
        let archive = generate_package(&PackageFixtureParams::default());
        let extractor = PackageExtractor::new(ExtractionLimits::default());

        let manifest = extractor
            .extract(&archive, &staging, None)
            .expect("extraction should succeed");

        assert_eq!(manifest.package_kind, "burn-worker");
        assert!(staging.join("reimagine-burn-worker").exists());
        assert!(staging.join("LICENSE").exists());
    }

    #[test]
    fn different_manifest_digest_produces_different_package() {
        let mut p = PackageFixtureParams::default();
        p.manifest_digest = "different-digest".to_string();
        let p1 = generate_package(&PackageFixtureParams::default());
        let p2 = generate_package(&p);
        assert_ne!(p1, p2);
    }
}
