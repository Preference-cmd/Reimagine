//! Dry-run release verification: package a real compiled worker binary,
//! sign TUF metadata with deterministic test keys, and verify the complete
//! release chain.
//!
//! This test is gated by the `REIMAGINE_DRY_RUN_BINARY` environment variable.
//! When unset, the test skips (useful for CI where the release binary is
//! compiled separately). When set to the path of a compiled
//! `reimagine-inference-burn-worker` binary, it verifies the full release
//! pipeline end-to-end.
//!
//! Usage in CI:
//!   REIMAGINE_DRY_RUN_BINARY=target/release/reimagine-inference-burn-worker \
//!     cargo test --package reimagine-backend-worker-host --test dry_run_release

use std::path::Path;

use reimagine_backend_worker_host::catalog::tuf;
use reimagine_backend_worker_host::testing::{
    self, PackageFixtureParams, TufMetadataParams,
};
use reimagine_backend_worker_host::{
    ExtractionLimits, PackageExtractor,
};

const ENV_BINARY_PATH: &str = "REIMAGINE_DRY_RUN_BINARY";

/// Read a real compiled worker binary from disk, if available.
fn maybe_worker_binary() -> Option<Vec<u8>> {
    let path = std::env::var(ENV_BINARY_PATH).ok()?;
    std::fs::read(&path).ok()
}

/// Read binary name from the env var path, or use a default.
fn binary_name_from_env() -> String {
    std::env::var(ENV_BINARY_PATH)
        .ok()
        .and_then(|p| {
            Path::new(&p)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| "reimagine-inference-burn-worker".to_string())
}

/// Verify that the generated TUF metadata chain is valid.
fn verify_tuf_chain(metadata: &testing::TufMetadataBundle) -> Vec<String> {
    let root = &metadata.root;
    let _root_keys = tuf::verify_root(root, None).expect("root verification");
    tuf::verify_timestamp(&metadata.timestamp, root, 0).expect("timestamp verification");

    let snapshot_meta = metadata
        .timestamp
        .signed
        .meta
        .get("snapshot.json")
        .expect("snapshot.json meta");
    let snapshot_bytes = serde_json::to_vec(&metadata.snapshot).unwrap();
    tuf::verify_snapshot(&metadata.snapshot, &snapshot_bytes, root, snapshot_meta)
        .expect("snapshot verification");

    let targets_meta = metadata
        .snapshot
        .signed
        .meta
        .get("targets.json")
        .expect("targets.json meta");
    let targets_bytes = serde_json::to_vec(&metadata.targets).unwrap();
    tuf::verify_targets(&metadata.targets, &targets_bytes, root, targets_meta)
        .expect("targets verification");

    metadata.targets.signed.targets.keys().cloned().collect()
}

#[test]
fn dry_run_release_chain_from_real_binary() {
    let binary = match maybe_worker_binary() {
        Some(b) => b,
        None => {
            eprintln!(
                "skipping dry-run: set {ENV_BINARY_PATH}=/path/to/reimagine-inference-burn-worker"
            );
            return;
        }
    };

    let dir = tempfile::tempdir().expect("temp dir");
    let binary_name = binary_name_from_env();
    let target = std::env::consts::ARCH.to_string();
    let os = std::env::consts::OS.to_string();

    // ── Step 1: Build a package from the real binary ──────────────
    let pkg_params = PackageFixtureParams {
        package_kind: "burn-worker".to_string(),
        binary_name: binary_name.clone(),
        binary_content: binary,
        backend_instance_id: "burn:wgpu:default".to_string(),
        installation_id: format!("burn-wgpu-{}-{}", os, target),
        backend_kind: "burn".to_string(),
        target: target.clone(),
        manifest_digest: "dry-run-release".to_string(),
    };

    let package_archive = testing::generate_package(&pkg_params);
    assert!(
        !package_archive.is_empty(),
        "generated package archive must not be empty"
    );

    // ── Step 2: Write package to disk ─────────────────────────────
    let package_filename = format!("reimagine-worker-burn-{target}.tar.gz");
    std::fs::write(dir.path().join(&package_filename), &package_archive)
        .expect("write package file");

    // ── Step 3: Generate TUF metadata with deterministic test keys ─
    let tuf_params = TufMetadataParams {
        root_version: 1,
        targets_version: 1,
        snapshot_version: 1,
        timestamp_version: 1,
        expires: "2999-12-31T23:59:59Z".to_string(),
    };

    let catalog = testing::generate_full_catalog(
        dir.path(),
        &tuf_params,
        &[pkg_params],
    );

    // ── Step 4: Verify TUF chain ──────────────────────────────────
    let target_names = verify_tuf_chain(&catalog.metadata);
    assert_eq!(target_names.len(), 1, "should have 1 target");
    assert!(
        target_names[0].contains(&target),
        "target path should contain arch: {}",
        target_names[0]
    );

    // ── Step 5: Verify the TUF root matches embedded format ───────
    let root_json = serde_json::to_value(&catalog.metadata.root).unwrap();
    // The CatalogClient expects a signed root. Verify shape matches.
    assert!(
        root_json.get("signed").is_some(),
        "root must have signed field"
    );
    assert!(
        root_json.get("signatures").is_some(),
        "root must have signatures field"
    );

    // ── Step 6: Extract and verify the package ────────────────────
    let staging = dir.path().join("extracted");
    let extractor = PackageExtractor::new(ExtractionLimits::default());
    let manifest = extractor
        .extract(&package_archive, &staging, None)
        .expect("package extraction should succeed");

    assert_eq!(manifest.package_kind, "burn-worker");
    assert!(
        staging.join(&binary_name).exists(),
        "extracted binary must exist: {binary_name}"
    );
    assert!(staging.join("LICENSE").exists());

    // ── Step 7: Verify metadata hashes match after extraction ─────
    for file_entry in &manifest.files {
        let file_path = staging.join(&file_entry.path);
        let actual_hash = {
            use sha2::{Digest, Sha256};
            let data = std::fs::read(&file_path).expect("read file for hash");
            hex::encode(Sha256::digest(&data))
        };
        assert_eq!(
            actual_hash, file_entry.sha256,
            "hash mismatch for {}",
            file_entry.path
        );
    }

    eprintln!(
        "dry-run release chain passed: {} ({}, {} bytes, {} files)",
        binary_name,
        target,
        manifest.files.iter().map(|f| f.size).sum::<u64>(),
        manifest.files.len(),
    );
}
