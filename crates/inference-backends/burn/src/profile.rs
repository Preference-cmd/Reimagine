use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceProfile, BackendInstanceStatus, BackendProfile,
    BackendProfileProvider, DeviceKind, DeviceProfile, InferenceCapability,
};
use reimagine_plugin::{Extension, Plugin};

pub(crate) const BACKEND_LABEL: &str = "burn";
pub(crate) const PLUGIN_LABEL: &str = "builtin.burn";
pub(crate) const EXTENSION_LABEL: &str = "backend.burn";
pub(crate) const CPU_INSTANCE: &str = "burn:cpu";

#[derive(Debug, Clone, Default)]
pub struct BurnProfileProvider;

impl BurnProfileProvider {
    pub const fn new() -> Self {
        Self
    }

    pub fn backend_kind() -> Backend {
        Backend::new(BACKEND_LABEL)
    }

    pub fn cpu_instance() -> BackendInstance {
        BackendInstance::new(CPU_INSTANCE)
    }

    pub fn plugin_provenance() -> (Plugin, Extension) {
        let plugin = Plugin::try_from(PLUGIN_LABEL).expect("valid built-in Burn plugin id");
        let extension =
            Extension::try_from(EXTENSION_LABEL).expect("valid built-in Burn extension id");
        (plugin, extension)
    }

    /// Advertise backend instances matching the active feature.
    ///
    /// burn/13 keeps the legacy `burn:cpu` ndarray instance
    /// alongside the new `burn:wgpu:*` / `burn:flex:*`
    /// instances. The legacy instance keeps its full
    /// capability stack (`LoadBundle + CreateEmptyLatent +
    /// TextEncode`) so downstream callers depending on the
    /// CPU instance's capabilities (load_bundle, burn/09
    /// latent creation, burn/08f text.encode) keep working.
    /// Each new wgpu/flex instance advertises only
    /// [`InferenceCapability::LoadBundle`] for V1 — real
    /// forward-pass capability parity lands in burn/08f+,
    /// burn/10, and burn/11.
    pub fn probe(&self) -> BackendProfile {
        let backend = Self::backend_kind();
        let (plugin, extension) = Self::plugin_provenance();

        let ndarray_cpu = BackendInstanceProfile::new(
            Self::cpu_instance(),
            backend.clone(),
            DeviceProfile::new("cpu").with_kind(DeviceKind::Cpu),
            BackendInstanceStatus::Available,
        )
        .with_capability(InferenceCapability::LoadBundle)
        .with_capability(InferenceCapability::CreateEmptyLatent)
        .with_capability(InferenceCapability::TextEncode);

        let mut profile = BackendProfile::new(backend)
            .with_plugin(plugin, extension)
            .with_instance(ndarray_cpu);

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
    // `burn:wgpu:<label>` instance advertised in the burn/13
    // issue (D3).
    let adapters: [(&'static str, WgpuDevice, DeviceKind); 3] = [
        ("wgpu:metal", WgpuDevice::IntegratedGpu(0), DeviceKind::Gpu),
        ("wgpu:vulkan", WgpuDevice::DiscreteGpu(0), DeviceKind::Gpu),
        ("wgpu:cpu", WgpuDevice::Cpu, DeviceKind::Cpu),
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
        .with_capability(InferenceCapability::LoadBundle);
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
    .with_capability(InferenceCapability::LoadBundle);
    profile.with_instance(instance_profile)
}

#[async_trait::async_trait]
impl BackendProfileProvider for BurnProfileProvider {
    async fn backend_profile(&self) -> BackendProfile {
        self.probe()
    }
}
