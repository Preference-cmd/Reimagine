//! Install pipeline persists the worker's manifest profile.
//!
//! The package binary is a shell script that emulates the worker's
//! `--version` self-check output contract: line 1 is the
//! `ExpectedWorkerIdentity` JSON, line 2 is the `WorkerInstanceProfile`
//! JSON. After a full catalog → install run, the durable
//! `InstallationRecord` must carry the captured profile, so the host
//! switch-target resolution can launch the worker without re-probing it.

#![cfg(unix)]

use std::collections::HashMap;

use reimagine_backend_worker_host::catalog::client::CatalogClient;
use reimagine_backend_worker_host::catalog::compatibility::{CompatibilityFilter, HostInfo};
use reimagine_backend_worker_host::testing::{self, PackageFixtureParams, TufMetadataParams};
use reimagine_backend_worker_host::{
    BackendInstanceId, ExpectedWorkerIdentity, InstallConfig, InstallEngine, InventoryStore,
    WorkerInstallationId, WorkerInstanceProfile, WorkerStorePaths,
};

mod fixtures;
use fixtures::server;

fn self_check_binary(
    identity: &ExpectedWorkerIdentity,
    profile: &WorkerInstanceProfile,
) -> Vec<u8> {
    format!(
        "#!/bin/sh\ncat <<'REIMAGINE_SELF_CHECK_EOF'\n{}\n{}\nREIMAGINE_SELF_CHECK_EOF\n",
        serde_json::to_string(identity).unwrap(),
        serde_json::to_string(profile).unwrap(),
    )
    .into_bytes()
}

/// Register a generated catalog's metadata and package files on the
/// fixture server under the pinned tag path (flat naming).
fn register_catalog(state: &server::AppState, tag: &str, fixture: &testing::CatalogFixture) {
    let meta_dir = fixture.catalog_dir.join("metadata");
    for entry in std::fs::read_dir(&meta_dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().to_string();
        if !name.ends_with(".json") {
            continue;
        }
        let data = std::fs::read(meta_dir.join(&name)).unwrap();
        server::register_file(state, &format!("{tag}/{name}"), data);
    }
    for pkg_path in &fixture.package_paths {
        let data = std::fs::read(fixture.catalog_dir.join(pkg_path)).unwrap();
        server::register_file(state, &format!("{tag}/{pkg_path}"), data);
    }
}

#[tokio::test]
async fn install_persists_worker_manifest_profile() {
    let tag = "worker-catalog-v12".to_string();

    let identity = ExpectedWorkerIdentity {
        backend_instance_id: BackendInstanceId("burn:wgpu:default".to_owned()),
        installation_id: WorkerInstallationId("burn-wgpu-v1".to_owned()),
        backend_kind: "burn".to_owned(),
        target: std::env::consts::ARCH.to_string(),
        manifest_digest: "test-manifest-0000".to_string(),
    };
    let profile = WorkerInstanceProfile {
        backend_instance_id: identity.backend_instance_id.clone(),
        device_label: "wgpu:default".to_owned(),
        capabilities: vec![
            "latent.create_empty".to_owned(),
            "diffusion.sample".to_owned(),
        ],
        operation_options: serde_json::json!({}),
    };
    let params = PackageFixtureParams {
        binary_content: self_check_binary(&identity, &profile),
        ..PackageFixtureParams::default()
    };

    let catalog_dir = tempfile::tempdir().unwrap();
    let fixture = testing::generate_full_catalog(
        catalog_dir.path(),
        &TufMetadataParams::default(),
        &[params],
    );

    let (server_base, state) = server::start_server(server::TestCatalogConfig {
        redirect_tag: tag.clone(),
        ..server::TestCatalogConfig::default()
    })
    .await;
    register_catalog(&state, &tag, &fixture);

    let filter = CompatibilityFilter::new(HostInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        supported_protocol_range: (1, 3),
    });
    let client = CatalogClient::new(format!("{server_base}/latest"), filter);
    let verified = client
        .fetch_catalog(&fixture.metadata.root, &HashMap::new(), 0)
        .await
        .expect("catalog fetch");
    let target = verified
        .targets
        .into_iter()
        .next()
        .expect("one compatible target");

    let store_root = tempfile::tempdir().unwrap();
    let store_paths = WorkerStorePaths::new(store_root.path().to_path_buf());
    let inventory = InventoryStore::new(store_paths.clone());
    let engine = InstallEngine::new(
        InstallConfig::default(),
        store_paths.clone(),
        InventoryStore::new(store_paths.clone()),
    );

    let record = engine
        .install(&client, &target)
        .await
        .expect("install pipeline succeeds");

    assert_eq!(
        record.manifest_profile.as_ref(),
        Some(&profile),
        "the install record must carry the worker's captured manifest profile"
    );

    let stored = inventory
        .get(&identity.installation_id.0)
        .expect("record persisted in inventory");
    assert_eq!(
        stored.manifest_profile,
        Some(profile),
        "the durable inventory record must round-trip the manifest profile"
    );
}
