use std::sync::Arc;

use reimagine_agent::WorkspaceScope;
use reimagine_app_host::{EmptyWorkerInventoryProvider, WorkspaceHost};
use reimagine_app_host::{
    StaticWorkerInventoryProvider, WorkerBackendCandidate, WorkerInventorySnapshot,
};
use reimagine_backend_worker_host::{ExpectedWorkerIdentity, WorkerLaunchSpec, WorkerLimits};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, ProtocolRange, WorkerInstallationId, WorkerInstanceProfile,
};
use reimagine_config::InferenceBackendConfig;
use reimagine_runtime::VecRunEventSink;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-mb04-{prefix}-{nonce}"))
}

fn missing_worker_candidate() -> WorkerBackendCandidate {
    let instance = BackendInstanceId("burn:wgpu:default".to_owned());
    WorkerBackendCandidate::try_new(
        WorkerLaunchSpec {
            executable: std::path::PathBuf::from("/missing/reimagine-burn-worker"),
            expected: ExpectedWorkerIdentity {
                backend_instance_id: instance.clone(),
                installation_id: WorkerInstallationId("test-installation".to_owned()),
                backend_kind: "burn".to_owned(),
                target: "test-target".to_owned(),
                manifest_digest: "sha256-test".to_owned(),
            },
            supported_protocols: ProtocolRange::new(1, 1),
            limits: WorkerLimits::default(),
            environment: Vec::new(),
            transport: Default::default(),
        },
        WorkerInstanceProfile {
            backend_instance_id: instance,
            device_label: "wgpu:default".to_owned(),
            capabilities: vec!["latent.create_empty".to_owned()],
            operation_options: serde_json::json!({}),
        },
    )
    .expect("matching inventory candidate")
}

#[tokio::test]
async fn inventory_profile_does_not_start_unselected_worker() {
    let base = temp_dir("dormant");
    let provider = StaticWorkerInventoryProvider::new(WorkerInventorySnapshot::new(vec![
        missing_worker_candidate(),
    ]));
    let workspace = WorkspaceHost::try_with_backend_config_and_worker_inventory(
        WorkspaceScope::new("mb04-dormant"),
        &base,
        InferenceBackendConfig::default(),
        Arc::new(VecRunEventSink::new()),
        Arc::new(provider),
    )
    .await
    .expect("unselected inventory must not start its process");

    let profile = workspace.compute_profile();
    let burn = profile
        .backend_profiles
        .iter()
        .find(|profile| profile.backend.as_str() == "burn")
        .expect("Burn profile");
    assert!(
        burn.instances.is_empty(),
        "no installed Burn worker, so the profile must not expose live instances"
    );
    assert!(
        burn.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code().as_str() == "APP_HOST/LOCAL_WORKER_NOT_INSTALLED"),
        "profile should carry a not-installed diagnostic"
    );
    assert!(matches!(
        workspace.selected_worker().await,
        Err(reimagine_app_host::WorkerSwitchError::NoActiveWorker)
    ));
    let _ = tokio::fs::remove_dir_all(base).await;
}

#[tokio::test]
async fn selected_worker_start_failure_becomes_unavailable_without_candle_fallback() {
    let base = temp_dir("activation-failure");
    let selected = "burn:wgpu:default";
    let provider = StaticWorkerInventoryProvider::new(WorkerInventorySnapshot::new(vec![
        missing_worker_candidate(),
    ]));
    let workspace = WorkspaceHost::try_with_backend_config_and_worker_inventory(
        WorkspaceScope::new("mb04-activation-failure"),
        &base,
        InferenceBackendConfig {
            selected_instance: Some(selected.to_owned()),
            priority_order: vec!["candle:cpu".to_owned()],
            ..InferenceBackendConfig::default()
        },
        Arc::new(VecRunEventSink::new()),
        Arc::new(provider),
    )
    .await
    .expect("worker failure is represented as readiness state");

    assert_eq!(workspace.resolved_backend_instance().as_str(), selected);
    let profile = workspace.compute_profile();
    let instance = profile
        .backend_profiles
        .iter()
        .flat_map(|backend| &backend.instances)
        .find(|instance| instance.instance.as_str() == selected)
        .expect("selected worker instance");
    assert!(matches!(
        instance.status,
        reimagine_inference::BackendInstanceStatus::Unavailable
    ));
    assert!(
        instance.diagnostics[0]
            .message()
            .contains("failed to spawn")
    );
    let _ = tokio::fs::remove_dir_all(base).await;
}

#[tokio::test]
async fn no_worker_bootstrap_keeps_editing_available_and_explicit_burn_pinned() {
    let base = temp_dir("no-worker");
    let selected = "burn:wgpu:default";
    let workspace = WorkspaceHost::try_with_backend_config_and_worker_inventory(
        WorkspaceScope::new("mb04-no-worker"),
        &base,
        InferenceBackendConfig {
            selected_instance: Some(selected.to_owned()),
            priority_order: vec!["candle:cpu".to_owned()],
            ..InferenceBackendConfig::default()
        },
        Arc::new(VecRunEventSink::new()),
        Arc::new(EmptyWorkerInventoryProvider),
    )
    .await
    .expect("no worker is a healthy workspace state");

    assert_eq!(workspace.resolved_backend_instance().as_str(), selected);
    assert!(!workspace.list_node_defs().is_empty());
    let profile = workspace.compute_profile();
    let burn = profile
        .backend_profiles
        .iter()
        .find(|profile| profile.backend.as_str() == "burn")
        .expect("Burn management profile remains visible");
    assert!(burn.instances.is_empty());
    assert!(
        burn.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code().as_str() == "APP_HOST/LOCAL_WORKER_NOT_INSTALLED")
    );

    let _ = tokio::fs::remove_dir_all(base).await;
}

/// BE-32 switch-target resolution over a durable installation record:
/// once the install pipeline persists `manifest_profile` on the record,
/// the desktop-host resolution steps (read record → find executable →
/// build candidate) succeed for an actually-installed worker.
#[test]
fn installed_record_with_manifest_profile_resolves_switch_candidate() {
    use reimagine_backend_worker_host::{InstallationRecord, InventoryStore, WorkerStorePaths};

    let temp = tempfile::tempdir().expect("temp dir");
    let install_dir = temp.path().join("installed").join("burn-wgpu-v1");
    std::fs::create_dir_all(&install_dir).expect("install dir");
    std::fs::write(
        install_dir.join("reimagine-inference-burn-worker"),
        b"binary",
    )
    .expect("worker binary");

    let instance = BackendInstanceId("burn:wgpu:default".to_owned());
    let profile = WorkerInstanceProfile {
        backend_instance_id: instance.clone(),
        device_label: "wgpu:default".to_owned(),
        capabilities: vec![
            "latent.create_empty".to_owned(),
            "diffusion.sample".to_owned(),
        ],
        operation_options: serde_json::json!({}),
    };
    let record = InstallationRecord {
        installation_id: WorkerInstallationId("burn-wgpu-v1".to_owned()),
        version: "1.0.0".to_owned(),
        identity: ExpectedWorkerIdentity {
            backend_instance_id: instance.clone(),
            installation_id: WorkerInstallationId("burn-wgpu-v1".to_owned()),
            backend_kind: "burn".to_owned(),
            target: "test-target".to_owned(),
            manifest_digest: "digest".to_owned(),
        },
        installed_at: chrono::Utc::now(),
        install_path: install_dir.to_string_lossy().into_owned(),
        manifest_profile: Some(profile.clone()),
    };

    let store = InventoryStore::new(WorkerStorePaths::new(temp.path().to_path_buf()));
    store.add(&record).expect("inventory add");

    // Mirror `desktop_host::resolve_switch_candidate` over the durable record.
    let stored = store
        .get(&record.installation_id.0)
        .expect("installed record");
    let executable = installed_worker_executable(&stored.install_path)
        .expect("executable resolved from install dir");
    let manifest_profile = stored
        .manifest_profile
        .clone()
        .expect("install pipeline must persist the manifest profile");
    let candidate = WorkerBackendCandidate::try_new(
        WorkerLaunchSpec {
            executable,
            expected: stored.identity.clone(),
            supported_protocols: ProtocolRange::new(1, 1),
            limits: WorkerLimits::default(),
            environment: Vec::new(),
            transport: Default::default(),
        },
        manifest_profile,
    )
    .expect("installed worker must resolve to a switch candidate");

    assert_eq!(candidate.backend_instance().as_str(), "burn:wgpu:default");
    assert_eq!(candidate.manifest_profile().device_label, "wgpu:default");
    assert!(
        candidate
            .manifest_profile()
            .capabilities
            .contains(&"diffusion.sample".to_owned())
    );
}

fn installed_worker_executable(install_dir: &str) -> Option<std::path::PathBuf> {
    let files = std::fs::read_dir(install_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    let preferred = files.iter().find(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("burn-worker"))
    });
    preferred.cloned().or_else(|| files.into_iter().next())
}
