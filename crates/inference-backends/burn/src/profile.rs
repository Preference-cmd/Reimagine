use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceProfile, BackendInstanceStatus, BackendProfile,
    BackendProfileProvider, DeviceKind, DeviceProfile, InferenceCapability,
};
use reimagine_plugin::{Extension, Plugin};

pub(crate) const BACKEND_LABEL: &str = "burn";
pub(crate) const PLUGIN_LABEL: &str = "builtin.burn";
pub(crate) const EXTENSION_LABEL: &str = "backend.burn";

#[derive(Debug, Clone, Default)]
pub struct BurnProfileProvider;

impl BurnProfileProvider {
    pub const fn new() -> Self {
        Self
    }

    pub fn backend_kind() -> Backend {
        Backend::new(BACKEND_LABEL)
    }

    pub fn plugin_provenance() -> (Plugin, Extension) {
        let plugin = Plugin::try_from(PLUGIN_LABEL).expect("valid built-in Burn plugin id");
        let extension =
            Extension::try_from(EXTENSION_LABEL).expect("valid built-in Burn extension id");
        (plugin, extension)
    }

    /// Advertise backend instances matching the active feature.
    ///
    /// V1 production instances follow the active feature:
    /// WGPU by default and Flex for CPU fallback builds.
    pub fn probe(&self) -> BackendProfile {
        let backend = Self::backend_kind();
        let (plugin, extension) = Self::plugin_provenance();

        let mut profile = BackendProfile::new(backend).with_plugin(plugin, extension);

        // Under `wgpu` (default), advertise synthesized instances
        // per graphics API. Real adapter enumeration at runtime
        // is deferred to a later issue; V1 lists the static
        // variants so callers can plan against the label space.
        #[cfg(feature = "wgpu")]
        {
            profile = add_wgpu_instances(profile);
        }

        // Under `flex` (only), advertise the single `burn:flex:cpu`
        // instance — burn-flex has no device selection.
        #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
        {
            profile = add_flex_instance(profile);
        }

        profile
    }
}

#[cfg(feature = "wgpu")]
fn add_wgpu_instances(profile: BackendProfile) -> BackendProfile {
    use burn_wgpu::WgpuDevice;

    // Synthesized adapter enumerations for the static label
    // space. Each entry corresponds to the canonical
    // `burn:wgpu:<label>` instance. CPU fallback is represented
    // by `burn:flex:cpu`, not by WGPU's internal CPU adapter.
    let adapters: [(&'static str, WgpuDevice, DeviceKind); 3] = [
        ("wgpu:default", WgpuDevice::DefaultDevice, DeviceKind::Gpu),
        ("wgpu:metal", WgpuDevice::IntegratedGpu(0), DeviceKind::Gpu),
        ("wgpu:vulkan", WgpuDevice::DiscreteGpu(0), DeviceKind::Gpu),
    ];

    let mut next = profile;
    for (label, device, kind) in adapters {
        let instance = BackendInstance::new(format!("burn:{label}"));
        let device_profile = DeviceProfile::new(label).with_kind(adapter_kind(&device, kind));
        let instance_profile = BackendInstanceProfile::new(
            instance,
            BurnProfileProvider::backend_kind(),
            device_profile,
            BackendInstanceStatus::Available,
        )
        .with_capability(InferenceCapability::LoadBundle)
        .with_capability(InferenceCapability::CreateEmptyLatent)
        .with_capability(InferenceCapability::TextEncode)
        .with_capability(InferenceCapability::DiffusionSample)
        .with_capability(InferenceCapability::LatentDecode)
        .with_capability(InferenceCapability::ImageSave)
        .with_capability(InferenceCapability::ImagePreview);
        next = next.with_instance(instance_profile);
    }
    next
}

#[cfg(feature = "wgpu")]
fn adapter_kind(device: &burn_wgpu::WgpuDevice, default: DeviceKind) -> DeviceKind {
    use burn_wgpu::WgpuDevice;
    match device {
        WgpuDevice::Cpu => DeviceKind::Cpu,
        WgpuDevice::IntegratedGpu(_) | WgpuDevice::DiscreteGpu(_) | WgpuDevice::VirtualGpu(_) => {
            DeviceKind::Gpu
        }
        #[allow(deprecated)]
        WgpuDevice::BestAvailable => DeviceKind::Gpu,
        WgpuDevice::DefaultDevice | WgpuDevice::Existing(_) => default,
    }
}

#[cfg(all(not(feature = "wgpu"), feature = "flex"))]
fn add_flex_instance(profile: BackendProfile) -> BackendProfile {
    let instance = BackendInstance::new("burn:flex:cpu");
    let device_profile = DeviceProfile::new("flex:cpu").with_kind(DeviceKind::Cpu);
    let instance_profile = BackendInstanceProfile::new(
        instance,
        BurnProfileProvider::backend_kind(),
        device_profile,
        BackendInstanceStatus::Available,
    )
    .with_capability(InferenceCapability::LoadBundle)
    .with_capability(InferenceCapability::CreateEmptyLatent)
    .with_capability(InferenceCapability::TextEncode);
    profile.with_instance(instance_profile)
}

#[async_trait::async_trait]
impl BackendProfileProvider for BurnProfileProvider {
    async fn backend_profile(&self) -> BackendProfile {
        self.probe()
    }
}
