use std::path::PathBuf;

use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_inference::{
    BackendInstance, BackendInstanceObservation, BackendInstanceStatus, BackendProfileProvider,
    BackendRunLifecycle, BackendRunLifecycleRequest, CreateEmptyLatentRequest, DeviceKind,
    InferenceBackend, InferenceCapability,
};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig, BurnDevice, BurnProfileProvider};

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
async fn profile_reports_builtin_burn_cpu_with_load_bundle_capability() {
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
    // burn/05 advertised LoadBundle; burn/09 extends the CPU
    // instance profile with CreateEmptyLatent. Future issues will
    // add more capabilities.
    assert_eq!(
        cpu.capabilities,
        vec![
            InferenceCapability::LoadBundle,
            InferenceCapability::CreateEmptyLatent,
        ]
    );
    assert!(cpu.operation_options.is_empty());
    assert!(cpu.diagnostics.is_empty());
}

#[test]
fn backend_kind_instance_and_capabilities_report_load_bundle_and_create_empty_latent() {
    let backend = backend();
    let capabilities = backend.capabilities();

    assert_eq!(backend.backend_kind().as_str(), "burn");
    assert_eq!(backend.backend_instance(), BackendInstance::new("burn:cpu"));
    assert_eq!(capabilities.backend_kind().as_str(), "burn");
    // burn/05 advertised LoadBundle; burn/09 adds
    // CreateEmptyLatent. The capability set is intentionally
    // small in V1; later issues (text/sample/decode/encode/image)
    // will extend it.
    assert_eq!(capabilities.capability_supports().len(), 2);
    assert!(capabilities.supports_capability(InferenceCapability::LoadBundle));
    assert!(capabilities.supports_capability(InferenceCapability::CreateEmptyLatent));
}

#[tokio::test]
async fn create_empty_latent_succeeds_with_burn_affine_handle() {
    // burn/09 implements `latent.create_empty`. The skeleton-era
    // `BackendNotImplemented` path is no longer the source of
    // truth for CreateEmptyLatent; downstream capabilities
    // (text_encode, diffusion_sample, latent_decode/encode,
    // image_*) remain BackendNotImplemented until their
    // dedicated issues land — this is exercised by the burn
    // backend's `not_implemented` helper, which is shared with
    // the same path the other capabilities take.
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

    let response = backend
        .create_empty_latent(request)
        .await
        .expect("burn/09 implements latent.create_empty");
    let latent = response.into_latent();
    assert_eq!(latent.payload().backend().as_str(), "burn");
    assert_eq!(latent.payload().backend_instance().as_str(), "burn:cpu");
    assert_eq!(
        latent.latent_space().id().as_str(),
        "stable_diffusion/sdxl/base"
    );
    assert_eq!(latent.payload().shape().dims(), &[1_usize, 4, 64, 64]);
}

#[tokio::test]
async fn runtime_hooks_report_snapshot_identity_and_cache_counts() {
    let backend = backend();
    let hooks = backend.runtime_hooks(None, None, None);
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
    assert!(snapshot.plugin.is_none());
    assert!(snapshot.extension.is_none());
    assert!(snapshot.device.is_none());
    assert_eq!(
        snapshot.observations.get("cached_models"),
        Some(&"0".to_owned())
    );
    assert_eq!(
        snapshot.observations.get("run_payloads"),
        Some(&"0".to_owned())
    );
    assert!(snapshot.diagnostics.is_empty());
}
