use std::path::PathBuf;

use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_inference::{
    BackendInstance, BackendInstanceObservation, BackendInstanceStatus, BackendProfileProvider,
    BackendRunLifecycle, BackendRunLifecycleRequest, CreateEmptyLatentRequest, DeviceKind,
    InferenceBackend, InferenceCapability, InferenceError,
};
use reimagine_inference_burn::{
    BurnBackend, BurnBackendConfig, BurnBackendInstanceRuntimeHooks, BurnDevice,
    BurnProfileProvider,
};

fn backend() -> BurnBackend {
    BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("burn backend")
}

#[test]
fn config_defaults_to_cpu_device_and_stores_paths() {
    let config = BurnBackendConfig::new("/models", "/output");

    assert_eq!(config.models_dir(), &PathBuf::from("/models"));
    assert_eq!(config.output_dir(), &PathBuf::from("/output"));
    assert_eq!(config.device().label(), "cpu");
    assert_eq!(config.device_label(), "cpu");
}

#[test]
fn device_builds_cpu_and_rejects_unknown_labels() {
    assert!(BurnDevice::new("cpu").try_build_device().is_ok());

    let err = BurnDevice::new("gpu").try_build_device().unwrap_err();
    assert!(err.to_string().contains("gpu"));
}

#[tokio::test]
async fn profile_reports_builtin_burn_cpu_without_capabilities() {
    let profile = BurnProfileProvider::new().backend_profile().await;

    assert_eq!(profile.backend.as_str(), "burn");
    assert_eq!(
        profile.plugin.as_ref().map(|p| p.as_str()),
        Some("builtin.burn")
    );
    assert_eq!(
        profile.extension.as_ref().map(|e| e.as_str()),
        Some("backend.burn")
    );

    let cpu = profile
        .instances
        .iter()
        .find(|instance| instance.instance.as_str() == "burn:cpu")
        .expect("burn:cpu profile");
    assert_eq!(cpu.status, BackendInstanceStatus::Available);
    assert_eq!(cpu.backend.as_str(), "burn");
    assert_eq!(cpu.device.label, "cpu");
    assert_eq!(cpu.device.kind, DeviceKind::Cpu);
    assert!(cpu.capabilities.is_empty());
    assert!(cpu.operation_options.is_empty());
    assert!(cpu.diagnostics.is_empty());
}

#[test]
fn backend_kind_instance_and_capabilities_are_empty() {
    let backend = backend();
    let capabilities = backend.capabilities();

    assert_eq!(backend.backend_kind().as_str(), "burn");
    assert_eq!(backend.backend_instance(), BackendInstance::new("burn:cpu"));
    assert_eq!(capabilities.backend_kind().as_str(), "burn");
    assert!(capabilities.capability_supports().is_empty());
    assert!(!capabilities.supports_capability(InferenceCapability::LoadBundle));
}

#[tokio::test]
async fn direct_methods_return_structured_backend_not_implemented() {
    let backend = backend();
    let request = CreateEmptyLatentRequest::new(
        512,
        512,
        1,
        RunId::new("run-burn"),
        WorkflowId::new("workflow-burn"),
        WorkflowVersion::new(1),
        NodeId::new("latent-burn"),
    );

    let err = backend.create_empty_latent(request).await.unwrap_err();
    match err {
        InferenceError::BackendNotImplemented {
            capability,
            backend_kind,
            message,
        } => {
            assert_eq!(capability, InferenceCapability::CreateEmptyLatent);
            assert_eq!(backend_kind, "burn");
            assert!(message.unwrap().contains("skeleton"));
        }
        other => panic!("expected BackendNotImplemented, got {other:?}"),
    }
}

#[tokio::test]
async fn runtime_hooks_are_noop_and_report_snapshot_identity() {
    let hooks =
        BurnBackendInstanceRuntimeHooks::new(BackendInstance::new("burn:cpu"), Some("cpu".into()));
    let request = BackendRunLifecycleRequest {
        run_id: reimagine_core::model::RunId::new("run-burn-hooks"),
    };

    let begin = hooks.begin_run(request.clone()).await.expect("begin");
    let cleanup = hooks.cleanup_run(request).await.expect("cleanup");
    let snapshot = hooks.snapshot().await;

    assert_eq!(begin.backend_instance, BackendInstance::new("burn:cpu"));
    assert!(begin.diagnostics.is_empty());
    assert_eq!(cleanup.backend_instance, BackendInstance::new("burn:cpu"));
    assert!(cleanup.diagnostics.is_empty());
    assert_eq!(snapshot.backend_instance, BackendInstance::new("burn:cpu"));
    assert_eq!(snapshot.backend.as_str(), "burn");
    assert_eq!(
        snapshot.plugin.as_ref().map(|plugin| plugin.as_str()),
        Some("builtin.burn")
    );
    assert_eq!(
        snapshot
            .extension
            .as_ref()
            .map(|extension| extension.as_str()),
        Some("backend.burn")
    );
    assert_eq!(
        snapshot.device.as_ref().map(|device| device.label.as_str()),
        Some("cpu")
    );
    assert!(snapshot.observations.is_empty());
    assert!(snapshot.diagnostics.is_empty());
}
