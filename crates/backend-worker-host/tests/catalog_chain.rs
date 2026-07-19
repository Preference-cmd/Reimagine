//! Integration test for the complete worker release pipeline chain.
//!
//! Covers: package generation → TUF signing → catalog assembly →
//! TUF chain verification → package download hash verification →
//! package extraction.
//!
//! The TUF verification step uses the deterministic `testing::` fixtures
//! and exercises the same `catalog::tuf` verification functions that
//! `CatalogClient::fetch_catalog` calls at runtime.

use reimagine_backend_worker_host::catalog::tuf;
use reimagine_backend_worker_host::testing::{self, PackageFixtureParams, TufMetadataParams};
use reimagine_backend_worker_host::{CatalogTarget, ExtractionLimits, PackageExtractor};
use sha2::{Digest, Sha256};

/// Verify the full TUF chain from root → timestamp → snapshot → targets.
fn verify_tuf_chain(
    metadata: &testing::TufMetadataBundle,
    stored_timestamp_version: u64,
) -> Vec<CatalogTarget> {
    let root = &metadata.root;
    let _root_keys = tuf::verify_root(root, None).expect("root verification");
    tuf::verify_timestamp(&metadata.timestamp, root, stored_timestamp_version)
        .expect("timestamp verification");
    let snapshot_meta = metadata
        .timestamp
        .signed
        .meta
        .get("snapshot.json")
        .expect("timestamp has snapshot.json meta");
    let snapshot_bytes = serde_json::to_vec(&metadata.snapshot).unwrap();
    tuf::verify_snapshot(&metadata.snapshot, &snapshot_bytes, root, snapshot_meta)
        .expect("snapshot verification");
    let targets_meta = metadata
        .snapshot
        .signed
        .meta
        .get("targets.json")
        .expect("snapshot has targets.json meta");
    let targets_bytes = serde_json::to_vec(&metadata.targets).unwrap();
    tuf::verify_targets(&metadata.targets, &targets_bytes, root, targets_meta)
        .expect("targets verification");

    let mut catalog_targets = Vec::new();
    for (path, desc) in &metadata.targets.signed.targets {
        let sha256 = desc
            .hashes
            .get("sha256")
            .cloned()
            .expect("target has valid sha256");
        let custom_deser: reimagine_backend_worker_host::TargetCustomMetadata =
            serde_json::from_value(desc.custom.clone().expect("target has custom metadata"))
                .expect("custom metadata");
        catalog_targets.push(CatalogTarget {
            path: path.clone(),
            sha256,
            length: desc.length,
            custom: custom_deser,
            download_url: format!("http://localhost/{path}"),
        });
    }
    catalog_targets
}

// ── Test cases ───────────────────────────────────────────────────────

#[test]
fn tuf_chain_verification_from_deterministic_fixture() {
    let metadata = testing::generate_tuf_metadata(&TufMetadataParams::default());
    assert!(verify_tuf_chain(&metadata, 0).is_empty());
}

#[test]
fn full_chain_from_fixture_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let pkgs = vec![PackageFixtureParams::default()];
    let catalog = testing::generate_full_catalog(dir.path(), &TufMetadataParams::default(), &pkgs);
    let targets = verify_tuf_chain(&catalog.metadata, 0);
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].custom.worker_kind, "burn");
    let archive_bytes = std::fs::read(dir.path().join(&catalog.package_paths[0])).unwrap();
    let staging = dir.path().join("installed");
    let extractor = PackageExtractor::new(ExtractionLimits::default());
    let manifest = extractor.extract(&archive_bytes, &staging, None).unwrap();
    assert_eq!(manifest.package_kind, "burn-worker");
    assert!(staging.join("reimagine-burn-worker").exists());
    assert!(staging.join("LICENSE").exists());
}

#[test]
fn multiple_worker_kinds_in_one_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let pkg_wgpu = PackageFixtureParams {
        backend_instance_id: "burn:wgpu:default".to_string(),
        installation_id: "burn-wgpu-v1".to_string(),
        ..PackageFixtureParams::default()
    };
    let pkg_flex = PackageFixtureParams {
        binary_name: "reimagine-burn-flex-worker".to_string(),
        backend_instance_id: "burn:flex:cpu".to_string(),
        installation_id: "burn-flex-cpu-v1".to_string(),
        backend_kind: "burn-flex".to_string(),
        target: "x86_64-unknown-linux-gnu".to_string(),
        manifest_digest: "test-manifest-flex".to_string(),
        ..PackageFixtureParams::default()
    };
    let catalog = testing::generate_full_catalog(
        dir.path(),
        &TufMetadataParams::default(),
        &[pkg_wgpu.clone(), pkg_flex.clone()],
    );
    let targets = verify_tuf_chain(&catalog.metadata, 0);
    assert_eq!(targets.len(), 2);
    for pkg in &[pkg_wgpu, pkg_flex] {
        let target = targets
            .iter()
            .find(|t| t.custom.installation_id == pkg.installation_id)
            .unwrap();
        let data = std::fs::read(dir.path().join(&target.path)).unwrap();
        let staging = dir
            .path()
            .join(format!("installed-{}", pkg.installation_id));
        let extractor = PackageExtractor::new(ExtractionLimits::default());
        let manifest = extractor.extract(&data, &staging, None).unwrap();
        assert_eq!(manifest.package_kind, pkg.package_kind);
        assert!(staging.join(&pkg.binary_name).exists());
    }
}

#[test]
fn chain_catches_target_hash_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = testing::generate_full_catalog(
        dir.path(),
        &TufMetadataParams::default(),
        &[PackageFixtureParams::default()],
    );
    let targets = verify_tuf_chain(&catalog.metadata, 0);
    assert_eq!(targets.len(), 1);
    let mut corrupted = std::fs::read(dir.path().join(&catalog.package_paths[0])).unwrap();
    corrupted[10] = 0xFF;
    assert_ne!(hex::encode(Sha256::digest(&corrupted)), targets[0].sha256);
}
