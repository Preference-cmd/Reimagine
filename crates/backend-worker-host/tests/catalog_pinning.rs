//! Integration tests for CatalogClient redirect pinning.
//!
//! These tests spin up a local axum server that simulates GitHub's
//! latest-Release redirect, serve TUF metadata and package archives,
//! and verify that the client pins the concrete tag and rejects
//! unexpected redirects and malformed tags.

use std::collections::HashMap;

use reimagine_backend_worker_host::catalog::client::CatalogClient;
use reimagine_backend_worker_host::catalog::compatibility::{CompatibilityFilter, HostInfo};
use reimagine_backend_worker_host::catalog::tuf;
use reimagine_backend_worker_host::testing;

mod fixtures;
use fixtures::server;

use ed25519_dalek::Signer;

/// Build a test catalog with one package target, register its files
/// on the server (flat naming, no metadata/ prefix), and return the
/// metadata bundle so callers can extract the trusted root.
async fn setup_test_catalog(
    config: server::TestCatalogConfig,
) -> (testing::TufMetadataBundle, String) {
    let tag = config.redirect_tag.clone();
    let (server_base, state) = server::start_server(config).await;

    // Use `generate_full_catalog` which handles all metadata linkage,
    // signing, and re-signing internally.
    let dir = tempfile::tempdir().expect("temp dir");
    let params = testing::PackageFixtureParams {
        target: testing::TufMetadataParams::default().targets_version.to_string(),
        ..testing::PackageFixtureParams::default()
    };
    let catalog = testing::generate_full_catalog(
        dir.path(),
        &testing::TufMetadataParams::default(),
        &[params],
    );

    // Register all metadata files at the flat paths.  We serve the compact
    // in-memory version (the link hashes were computed from `serde_json::to_vec`,
    // not pretty-printed).
    let mut meta_files: HashMap<String, Vec<u8>> = HashMap::new();
    let meta_dir = dir.path().join("metadata");
    for entry in std::fs::read_dir(&meta_dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().to_string();
        let data = match () {
            _ if name == "root.json" || name == "1.root.json" => {
                serde_json::to_vec(&catalog.metadata.root).unwrap()
            }
            _ if name == "targets.json" || name == "1.targets.json" => {
                serde_json::to_vec(&catalog.metadata.targets).unwrap()
            }
            _ if name == "snapshot.json" || name == "1.snapshot.json" => {
                serde_json::to_vec(&catalog.metadata.snapshot).unwrap()
            }
            _ if name == "timestamp.json" || name == "1.timestamp.json" => {
                serde_json::to_vec(&catalog.metadata.timestamp).unwrap()
            }
            _ => continue,
        };
        meta_files.insert(name, data);
    }

    for (name, data) in &meta_files {
        server::register_file(&*state, &format!("{tag}/{name}"), data.clone());
    }

    // Also register target package files.
    for pkg_path in &catalog.package_paths {
        let data = std::fs::read(dir.path().join(pkg_path)).unwrap();
        server::register_file(&*state, &format!("{tag}/{pkg_path}"), data);
    }

    (catalog.metadata, server_base)
}

#[tokio::test]
async fn redirect_latest_to_concrete_tag() {
    let config = server::TestCatalogConfig {
        redirect_tag: "worker-catalog-v12".to_string(),
        ..server::TestCatalogConfig::default()
    };
    let (metadata, server_base) = setup_test_catalog(config).await;
    let filter = CompatibilityFilter::new(HostInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        supported_protocol_range: (1, 3),
    });

    let client = CatalogClient::new(format!("{server_base}/latest"), filter);

    let result = client
        .fetch_catalog(&metadata.root, &HashMap::new(), 0)
        .await
        .expect("fetch_catalog should succeed");

    assert!(
        result.pinned_asset_base.contains("worker-catalog-v12"),
        "pinned base should contain the concrete tag, got: {}",
        result.pinned_asset_base,
    );

    assert_eq!(
        result.targets.len(),
        1,
        "expected exactly one compatible target"
    );

    let target = &result.targets[0];
    assert!(
        target.download_url.contains("worker-catalog-v12"),
        "target download URL should use pinned tag, got: {}",
        target.download_url,
    );
}

#[tokio::test]
async fn concrete_url_skips_discovery() {
    let config = server::TestCatalogConfig {
        redirect_tag: "worker-catalog-v7".to_string(),
        ..server::TestCatalogConfig::default()
    };
    let (metadata, server_base) = setup_test_catalog(config).await;
    let filter = CompatibilityFilter::new(HostInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        supported_protocol_range: (1, 3),
    });

    let client = CatalogClient::new(
        format!("{server_base}/releases/download/worker-catalog-v7"),
        filter,
    );

    let result = client
        .fetch_catalog(&metadata.root, &HashMap::new(), 0)
        .await
        .expect("fetch_catalog via concrete URL should succeed");

    assert!(result.pinned_asset_base.contains("worker-catalog-v7"));
    assert_eq!(result.targets.len(), 1);
}

#[tokio::test]
async fn redirect_to_unversioned_tag_is_rejected() {
    let config = server::TestCatalogConfig {
        redirect_tag: "some-other-release-v12".to_string(),
        ..server::TestCatalogConfig::default()
    };
    let (server_base, _state) = server::start_server(config).await;
    let filter = CompatibilityFilter::new(HostInfo {
        os: "linux".into(),
        arch: "x86_64".into(),
        supported_protocol_range: (1, 3),
    });

    let client = CatalogClient::new(format!("{server_base}/latest"), filter);

    let result = client
        .fetch_catalog(
            &fixtures_root_minimal(),
            &HashMap::new(),
            0,
        )
        .await;

    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("Discovery") || msg.contains("worker-catalog-v"),
                "error should mention discovery failure or tag mismatch: {msg}"
            );
        }
        Ok(_) => panic!("expected an error for unversioned tag redirect"),
    }
}

/// Returns a minimal root that passes `verify_root` with the deterministic
/// test key.
fn fixtures_root_minimal() -> tuf::RootMetadata {
    let key = testing::test_signing_key();
    let tuf_key = tuf::TufKey {
        key_type: "ed25519".to_string(),
        scheme: "ed25519".to_string(),
        keyval: tuf::TufKeyVal {
            public: testing::test_public_key_hex(),
        },
    };
    let key_id = testing::test_key_id();

    let signed = tuf::RootSigned {
        kind: "root".to_string(),
        version: 1,
        expires: "2999-12-31T23:59:59Z".to_string(),
        keys: HashMap::from([(key_id.clone(), tuf_key)]),
        roles: HashMap::from([
            (
                "root".to_string(),
                tuf::RoleConfig { keyids: vec![key_id.clone()], threshold: 1 },
            ),
            (
                "targets".to_string(),
                tuf::RoleConfig { keyids: vec![key_id.clone()], threshold: 1 },
            ),
            (
                "snapshot".to_string(),
                tuf::RoleConfig { keyids: vec![key_id.clone()], threshold: 1 },
            ),
            (
                "timestamp".to_string(),
                tuf::RoleConfig { keyids: vec![key_id.clone()], threshold: 1 },
            ),
        ]),
    };
    let payload = serde_json::to_vec(&serde_json::to_value(&signed).unwrap()).unwrap();
    let sig = key.sign(&payload).to_bytes();
    tuf::RootMetadata {
        signed,
        signatures: vec![tuf::SignatureEntry {
            keyid: key_id,
            sig: hex::encode(sig),
        }],
    }
}
