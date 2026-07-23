use super::*;
use crate::{SamplerName, SchedulerName};
use std::sync::Arc;

fn diag_in_target(message: &str) -> Diagnostic {
    Diagnostic::new(
        reimagine_core::model::DiagnosticId::new("d-1"),
        reimagine_core::diagnostic::DiagnosticCode::new("TEST/CODE"),
        reimagine_core::diagnostic::DiagnosticSeverity::Info,
        reimagine_core::diagnostic::DiagnosticSourceName::new("test"),
        message.to_string(),
        reimagine_core::diagnostic::DiagnosticTarget::new(
            reimagine_core::diagnostic::DiagnosticTargetDomain::new("test"),
        ),
    )
}

#[test]
fn kind_from_label_recognizes_known_labels() {
    assert_eq!(kind_from_label("cpu"), DeviceKind::Cpu);
    assert_eq!(kind_from_label("CPU"), DeviceKind::Cpu);
    assert_eq!(kind_from_label("metal"), DeviceKind::Gpu);
    assert_eq!(kind_from_label("mps"), DeviceKind::Gpu);
    assert_eq!(kind_from_label("cuda"), DeviceKind::Gpu);
    assert_eq!(kind_from_label("cuda:0"), DeviceKind::Gpu);
    assert_eq!(kind_from_label("CUDA:1"), DeviceKind::Gpu);
    assert_eq!(kind_from_label("remote"), DeviceKind::Remote);
    assert_eq!(kind_from_label("remote:foo"), DeviceKind::Remote);
    assert_eq!(kind_from_label(""), DeviceKind::Unknown);
    assert_eq!(kind_from_label("tpu"), DeviceKind::Unknown);
    assert_eq!(kind_from_label("npu:0"), DeviceKind::Unknown);
}

#[test]
fn device_profile_new_derives_kind_from_label() {
    let cpu = DeviceProfile::new("cpu");
    assert_eq!(cpu.label, "cpu");
    assert_eq!(cpu.kind, DeviceKind::Cpu);
    assert!(cpu.name.is_none());
    assert!(cpu.ordinal.is_none());
    assert!(cpu.memory.is_none());
    assert!(cpu.supported_dtypes.is_empty());

    let metal = DeviceProfile::new("metal");
    assert_eq!(metal.kind, DeviceKind::Gpu);

    let cuda0 = DeviceProfile::new("cuda:0");
    assert_eq!(cuda0.kind, DeviceKind::Gpu);

    let remote = DeviceProfile::new("remote:foo");
    assert_eq!(remote.kind, DeviceKind::Remote);

    let unknown = DeviceProfile::new("tpu");
    assert_eq!(unknown.kind, DeviceKind::Unknown);
}

#[test]
fn device_profile_with_kind_overrides_derived_kind() {
    let p = DeviceProfile::new("cpu").with_kind(DeviceKind::Gpu);
    assert_eq!(p.kind, DeviceKind::Gpu);
    assert_eq!(p.label, "cpu");
}

#[test]
fn device_profile_with_helpers_populate_optional_fields() {
    let memory = MemoryProfile::new().with_total(16 * 1024 * 1024 * 1024);
    let p = DeviceProfile::new("cuda:0")
        .with_name("NVIDIA RTX 4090")
        .with_ordinal(0)
        .with_memory(memory.clone())
        .with_supported_dtype("f32")
        .with_supported_dtype("f16");
    assert_eq!(p.name.as_deref(), Some("NVIDIA RTX 4090"));
    assert_eq!(p.ordinal, Some(0));
    assert_eq!(p.memory, Some(memory));
    assert_eq!(p.supported_dtypes.len(), 2);
    assert_eq!(p.supported_dtypes[0].as_str(), "f32");
    assert_eq!(p.supported_dtypes[1].as_str(), "f16");
}

#[test]
fn memory_profile_builder_methods_populate_fields() {
    let m = MemoryProfile::with_total_and_free(1024, 512);
    assert_eq!(m.total_bytes, Some(1024));
    assert_eq!(m.free_bytes, Some(512));

    let m = MemoryProfile::new().with_total(2048).with_free(1024);
    assert_eq!(m.total_bytes, Some(2048));
    assert_eq!(m.free_bytes, Some(1024));
}

#[test]
fn dtype_profile_new_and_from_conversions() {
    let p = DTypeProfile::new("f32");
    assert_eq!(p.as_str(), "f32");

    let from_str: DTypeProfile = "bf16".into();
    assert_eq!(from_str.as_str(), "bf16");

    let from_string: DTypeProfile = String::from("u8").into();
    assert_eq!(from_string.as_str(), "u8");
}

#[test]
fn workspace_compute_profile_serde_roundtrip() {
    let backend_profile = BackendProfile::new(Backend::new("candle"))
        .with_instance(
            BackendInstanceProfile::new(
                BackendInstance::new("candle:cpu"),
                Backend::new("candle"),
                DeviceProfile::new("cpu"),
                BackendInstanceStatus::Available,
            )
            .with_capability(InferenceCapability::DiffusionSample),
        )
        .with_diagnostic(diag_in_target("probe ok"));
    let original = WorkspaceComputeProfile::new()
        .with_backend_profile(backend_profile)
        .with_diagnostic(diag_in_target("top-level"));

    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: WorkspaceComputeProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, original);
}

#[test]
fn backend_profile_serde_roundtrip() {
    let plugin = Plugin::try_from("builtin.candle").unwrap();
    let extension = Extension::try_from("backend.candle").unwrap();
    let original = BackendProfile::new(Backend::new("candle"))
        .with_plugin(plugin, extension)
        .with_instance(BackendInstanceProfile::new(
            BackendInstance::new("candle:metal"),
            Backend::new("candle"),
            DeviceProfile::new("metal"),
            BackendInstanceStatus::Unavailable,
        ))
        .with_diagnostic(diag_in_target("metal unavailable"));

    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: BackendProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, original);
}

#[test]
fn backend_instance_profile_serde_roundtrip() {
    let memory = MemoryProfile::with_total_and_free(1024, 512);
    let device = DeviceProfile::new("cuda:0")
        .with_name("GPU 0")
        .with_ordinal(0)
        .with_memory(memory)
        .with_supported_dtype("f16");
    let original = BackendInstanceProfile::new(
        BackendInstance::new("candle:cuda"),
        Backend::new("candle"),
        device,
        BackendInstanceStatus::Available,
    )
    .with_capability(InferenceCapability::LoadBundle)
    .with_capability(InferenceCapability::DiffusionSample)
    .with_diagnostic(diag_in_target("ok"));

    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: BackendInstanceProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, original);
}

#[test]
fn backend_instance_profile_carries_diffusion_sample_operation_options() {
    let original = BackendInstanceProfile::new(
        BackendInstance::new("candle:cpu"),
        Backend::new("candle"),
        DeviceProfile::new("cpu"),
        BackendInstanceStatus::Available,
    )
    .with_operation_options(OperationOptionsProfile::diffusion_sample(
        vec![SamplerOptionProfile::new(SamplerName::Euler)],
        vec![SchedulerOptionProfile::new(SchedulerName::Normal)],
        vec![SamplerSchedulerPairProfile::new(
            SamplerName::Euler,
            SchedulerName::Normal,
        )],
    ));

    let options = original
        .operation_options
        .iter()
        .find(|options| options.capability == InferenceCapability::DiffusionSample)
        .expect("diffusion options");
    let OperationOptionsProfileKind::DiffusionSample {
        samplers,
        schedulers,
        supported_pairs,
    } = &options.options;
    assert_eq!(samplers[0].name.as_str(), "euler");
    assert_eq!(schedulers[0].name.as_str(), "normal");
    assert_eq!(supported_pairs[0].sampler.as_str(), "euler");
    assert_eq!(supported_pairs[0].scheduler.as_str(), "normal");

    let json = serde_json::to_string(&original).expect("serialize");
    assert!(json.contains("operation_options"));
    assert!(json.contains("diffusion.sample"));
    assert!(json.contains("euler"));
    assert!(json.contains("normal"));
    let parsed: BackendInstanceProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, original);
}

#[test]
fn device_profile_serde_roundtrip() {
    let original = DeviceProfile::new("cuda:0")
        .with_name("NVIDIA")
        .with_ordinal(0)
        .with_memory(MemoryProfile::with_total_and_free(2048, 1024))
        .with_supported_dtype("f32")
        .with_supported_dtype("bf16");

    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: DeviceProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, original);
}

#[test]
fn memory_profile_serde_roundtrip() {
    let original = MemoryProfile::with_total_and_free(2048, 1024);
    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: MemoryProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, original);

    let empty = MemoryProfile::new();
    let json = serde_json::to_string(&empty).expect("serialize empty");
    let parsed: MemoryProfile = serde_json::from_str(&json).expect("deserialize empty");
    assert_eq!(parsed, empty);
}

#[test]
fn dtype_profile_serde_roundtrip() {
    let original = DTypeProfile::new("f32");
    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: DTypeProfile = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, original);
}

#[test]
fn device_kind_serde_roundtrip_uses_pascal_case() {
    for (variant, expected) in [
        (DeviceKind::Cpu, "\"Cpu\""),
        (DeviceKind::Gpu, "\"Gpu\""),
        (DeviceKind::Remote, "\"Remote\""),
        (DeviceKind::Unknown, "\"Unknown\""),
    ] {
        let json = serde_json::to_string(&variant).expect("serialize");
        assert_eq!(
            json, expected,
            "kind {variant:?} should serialize as {expected}"
        );
        let parsed: DeviceKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, variant);
    }
}

#[test]
fn backend_instance_status_serde_roundtrip_uses_pascal_case() {
    for (variant, expected) in [
        (BackendInstanceStatus::Available, "\"Available\""),
        (BackendInstanceStatus::Unavailable, "\"Unavailable\""),
    ] {
        let json = serde_json::to_string(&variant).expect("serialize");
        assert_eq!(json, expected);
        let parsed: BackendInstanceStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, variant);
    }
}

#[test]
fn backend_profile_provider_trait_is_object_safe() {
    // Compile-time check: the trait can be used as a trait object
    // and dispatched through `&dyn BackendProfileProvider`.
    fn _accepts(_provider: &dyn BackendProfileProvider) {}

    struct Stub;

    #[async_trait::async_trait]
    impl BackendProfileProvider for Stub {
        async fn backend_profile(&self) -> BackendProfile {
            BackendProfile::new(Backend::new("stub"))
        }
    }

    let stub = Stub;
    let _provider: Arc<dyn BackendProfileProvider> = Arc::new(stub);
    let _ = _accepts;
}

#[test]
fn invalid_candle_device_diagnostic_targets_compute_profile() {
    let d = diagnostics::invalid_candle_device("tpu");
    assert_eq!(d.code().as_str(), "INFERENCE_PROFILE/INVALID_DEVICE");
    assert_eq!(d.source().as_str(), "inference");
    assert_eq!(
        d.primary().domain().as_str(),
        "app-host.compute_profile",
        "diagnostic should surface at app-host compute_profile"
    );
    assert!(d.primary().path().unwrap_or("").contains("tpu"));
    assert!(d.message().contains("tpu"));
}

#[test]
fn candle_device_unavailable_diagnostic_carries_reason() {
    let d = diagnostics::candle_device_unavailable("metal", "no metal runtime");
    assert_eq!(d.code().as_str(), "INFERENCE_PROFILE/DEVICE_UNAVAILABLE");
    assert_eq!(d.source().as_str(), "inference");
    assert_eq!(
        d.primary().domain().as_str(),
        "app-host.compute_profile",
        "diagnostic should surface at app-host compute_profile"
    );
    assert!(d.message().contains("metal"));
    assert!(d.message().contains("no metal runtime"));
}
