use reimagine_inference::{
    BackendInstanceProfile, BackendInstanceStatus, BackendProfile, BackendProfileProvider,
    DeviceKind, InferenceCapability,
};

#[tokio::test]
async fn cpu_instance_is_always_available_and_present() {
    let provider = super::CandleProfileProvider::new();
    let profile = provider.backend_profile().await;

    let cpu = find_instance(&profile, "candle:cpu");
    assert_eq!(cpu.status, BackendInstanceStatus::Available);
}

#[tokio::test]
async fn cpu_instance_carries_cpu_label_and_kind() {
    let provider = super::CandleProfileProvider::new();
    let profile = provider.backend_profile().await;

    let cpu = find_instance(&profile, "candle:cpu");
    assert_eq!(cpu.device.label, "cpu");
    assert_eq!(cpu.device.kind, DeviceKind::Cpu);
}

#[tokio::test]
async fn cpu_instance_advertises_all_v1_capabilities() {
    let provider = super::CandleProfileProvider::new();
    let profile = provider.backend_profile().await;

    let cpu = find_instance(&profile, "candle:cpu");
    for cap in InferenceCapability::all_v1() {
        assert!(
            cpu.capabilities.contains(cap),
            "cpu instance should advertise {cap}"
        );
    }
}

#[tokio::test]
async fn metal_instance_is_present_with_metal_label_and_gpu_kind() {
    let provider = super::CandleProfileProvider::new();
    let profile = provider.backend_profile().await;

    let metal = find_instance(&profile, "candle:metal");
    assert_eq!(metal.device.label, "metal");
    assert_eq!(metal.device.kind, DeviceKind::Gpu);
}

#[tokio::test]
async fn metal_instance_advertises_all_v1_capabilities() {
    let provider = super::CandleProfileProvider::new();
    let profile = provider.backend_profile().await;

    let metal = find_instance(&profile, "candle:metal");
    for cap in InferenceCapability::all_v1() {
        assert!(
            metal.capabilities.contains(cap),
            "metal instance should advertise {cap}"
        );
    }
}

#[tokio::test]
async fn metal_unavailable_instance_carries_inference_source_diagnostic() {
    let provider = super::CandleProfileProvider::new();
    let profile = provider.backend_profile().await;

    let metal = find_instance(&profile, "candle:metal");
    if metal.status == BackendInstanceStatus::Unavailable {
        assert!(
            !metal.diagnostics.is_empty(),
            "unavailable metal instance must carry at least one diagnostic"
        );
        assert!(
            metal
                .diagnostics
                .iter()
                .any(|d| d.source().as_str() == "inference"),
            "unavailable metal diagnostic must originate from the inference source"
        );
    }
}

#[tokio::test]
async fn backend_profile_carries_builtin_candle_plugin_provenance() {
    let provider = super::CandleProfileProvider::new();
    let profile = provider.backend_profile().await;

    assert_eq!(
        profile.plugin.as_ref().map(|p| p.as_str()),
        Some("builtin.candle")
    );
    assert_eq!(
        profile.extension.as_ref().map(|e| e.as_str()),
        Some("backend.candle")
    );
}

#[tokio::test]
async fn backend_label_is_candle() {
    let provider = super::CandleProfileProvider::new();
    let profile = provider.backend_profile().await;

    assert_eq!(profile.backend.as_str(), "candle");
}

#[tokio::test]
async fn profile_constructor_does_not_panic_on_any_host() {
    // Just calling the probe on a host without Metal must not panic.
    let provider = super::CandleProfileProvider::new();
    let _profile = provider.backend_profile().await;
}

fn find_instance<'a>(profile: &'a BackendProfile, identity: &str) -> &'a BackendInstanceProfile {
    profile
        .instances
        .iter()
        .find(|inst| inst.instance.as_str() == identity)
        .unwrap_or_else(|| panic!("profile missing `{identity}` instance"))
}
