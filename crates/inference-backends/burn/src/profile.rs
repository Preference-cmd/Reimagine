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

    pub fn probe(&self) -> BackendProfile {
        let backend = Self::backend_kind();
        let (plugin, extension) = Self::plugin_provenance();
        let cpu = BackendInstanceProfile::new(
            Self::cpu_instance(),
            backend.clone(),
            DeviceProfile::new("cpu").with_kind(DeviceKind::Cpu),
            BackendInstanceStatus::Available,
        )
        .with_capability(InferenceCapability::LoadBundle)
        .with_capability(InferenceCapability::CreateEmptyLatent);

        BackendProfile::new(backend)
            .with_plugin(plugin, extension)
            .with_instance(cpu)
    }
}

#[async_trait::async_trait]
impl BackendProfileProvider for BurnProfileProvider {
    async fn backend_profile(&self) -> BackendProfile {
        self.probe()
    }
}
