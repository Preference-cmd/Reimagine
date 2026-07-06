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

/// Expected short device label for the default
/// `BurnBackendConfig::new(...)` configuration.
fn expected_cpu_label() -> &'static str {
    #[cfg(feature = "wgpu")]
    {
        "wgpu:default"
    }
    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    {
        "flex:cpu"
    }
}

/// Expected full backend instance label for the default CPU
/// configuration under each feature.
fn expected_cpu_instance() -> &'static str {
    #[cfg(feature = "wgpu")]
    {
        "burn:wgpu:default"
    }
    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    {
        "burn:flex:cpu"
    }
}

fn expected_device_kind() -> DeviceKind {
    #[cfg(feature = "wgpu")]
    {
        DeviceKind::Gpu
    }
    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    {
        DeviceKind::Cpu
    }
}

#[test]
fn config_defaults_to_cpu_device_and_stores_paths() {
    let config = BurnBackendConfig::new("/models", "/output");

    assert_eq!(config.models_dir(), &PathBuf::from("/models"));
    assert_eq!(config.output_dir(), &PathBuf::from("/output"));
    assert_eq!(config.device().label(), expected_cpu_label());
    assert_eq!(config.device_label(), expected_cpu_label());
}

#[test]
fn device_resolves_feature_default_and_rejects_unknown_labels() {
    let expected_label = expected_cpu_label();
    let built = BurnDevice::default_device();
    assert_eq!(built.label(), expected_label);

    let err = BurnDevice::try_build_device("gpu").unwrap_err();
    assert!(err.to_string().contains("gpu"));
}

#[tokio::test]
async fn profile_reports_builtin_burn_default_instance_with_capabilities() {
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
        .find(|instance| instance.instance.as_str() == expected_cpu_instance())
        .expect("default burn profile");
    assert_eq!(cpu.status, BackendInstanceStatus::Available);
    assert_eq!(cpu.backend.as_str(), "burn");
    assert_eq!(cpu.device.label, expected_cpu_label());
    assert_eq!(cpu.device.kind, expected_device_kind());
    assert_eq!(
        cpu.capabilities,
        vec![
            InferenceCapability::LoadBundle,
            InferenceCapability::CreateEmptyLatent,
            InferenceCapability::TextEncode,
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
    assert_eq!(
        backend.backend_instance(),
        BackendInstance::new(expected_cpu_instance())
    );
    assert_eq!(capabilities.backend_kind().as_str(), "burn");
    // burn/08b-d-f merged text.encode, burn/10 adds
    // DiffusionSample, burn/11 adds LatentDecode,
    // burn/12 adds ImageImport/ImageSave/ImagePreview.
    assert_eq!(capabilities.capability_supports().len(), 8);
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
    assert_eq!(
        latent.payload().backend_instance().as_str(),
        expected_cpu_instance()
    );
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
    let expected_instance = BackendInstance::new(expected_cpu_instance());

    assert_eq!(begin.backend_instance, expected_instance);
    assert!(begin.diagnostics.is_empty());
    assert_eq!(cleanup.backend_instance, expected_instance);
    assert!(cleanup.diagnostics.is_empty());
    assert_eq!(snapshot.backend_instance, expected_instance);
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
