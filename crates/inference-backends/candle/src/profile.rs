//! Backend-local device probing and inference-owned profile DTO
//! construction for the Candle backend.
//!
//! The provider is a discovery factory: it does not require an
//! already-constructed [`CandleBackend`](crate::CandleBackend). It
//! probes Candle's own [`candle_core::Device`] constructors to
//! enumerate the CPU / Metal candidates the host can construct with
//! this build, then maps those backend-local results to the
//! inference-owned [`BackendProfile`] vocabulary. App-host is the
//! sole collector of [`BackendProfile`]s into a
//! [`WorkspaceComputeProfile`](reimagine_inference::WorkspaceComputeProfile).
//!
//! V1 enumerates `candle:cpu` (always available) and `candle:metal`
//! (probed via [`candle_core::Device::new_metal`]). A failed Metal
//! probe surfaces an `Unavailable` instance with a diagnostic rather
//! than aborting workspace bootstrap. CUDA / remote backends are out
//! of scope for V1 and will land behind additional provider entries.
//!
//! The provider must not depend on runtime, app-host, axum, tauri,
//! or model-manager. It deliberately produces only the host-neutral
//! inference DTOs.

use candle_core::Device;

use reimagine_inference::diagnostics::candle_device_unavailable;
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceProfile, BackendInstanceStatus, BackendProfile,
    BackendProfileProvider, DeviceKind, DeviceProfile, InferenceCapability,
    OperationOptionsProfile, SamplerName, SamplerOptionProfile, SamplerSchedulerPairProfile,
    SchedulerName, SchedulerOptionProfile,
};
use reimagine_plugin::{Extension, Plugin};

const BACKEND_LABEL: &str = "candle";
const PLUGIN_LABEL: &str = "builtin.candle";
const EXTENSION_LABEL: &str = "backend.candle";
const CPU_INSTANCE: &str = "candle:cpu";
const METAL_INSTANCE: &str = "candle:metal";

/// Discovery provider that probes Candle's own device constructors and
/// reports the CPU / Metal candidates the current host can build.
///
/// `CandleProfileProvider` carries no state. Construction is
/// allocation-free; probing happens inside
/// [`BackendProfileProvider::backend_profile`].
#[derive(Debug, Clone, Default)]
pub struct CandleProfileProvider;

impl CandleProfileProvider {
    /// Construct a new profile provider.
    pub const fn new() -> Self {
        Self
    }

    fn backend_kind() -> Backend {
        Backend::new(BACKEND_LABEL)
    }

    fn plugin_provenance() -> (Plugin, Extension) {
        let plugin = Plugin::try_from(PLUGIN_LABEL).expect("valid built-in plugin id");
        let extension = Extension::try_from(EXTENSION_LABEL).expect("valid built-in extension id");
        (plugin, extension)
    }

    fn v1_capabilities() -> Vec<InferenceCapability> {
        InferenceCapability::all_v1().to_vec()
    }

    fn attach_capabilities(
        mut instance: BackendInstanceProfile,
        capabilities: &[InferenceCapability],
    ) -> BackendInstanceProfile {
        for cap in capabilities {
            instance = instance.with_capability(*cap);
        }
        instance.with_operation_options(Self::diffusion_sample_options())
    }

    fn diffusion_sample_options() -> OperationOptionsProfile {
        OperationOptionsProfile::diffusion_sample(
            vec![SamplerOptionProfile::new(SamplerName::Euler)],
            vec![SchedulerOptionProfile::new(SchedulerName::Normal)],
            vec![SamplerSchedulerPairProfile::new(
                SamplerName::Euler,
                SchedulerName::Normal,
            )],
        )
    }
}

impl CandleProfileProvider {
    /// Build the backend profile synchronously.
    ///
    /// The probe itself is synchronous — [`Device::new_metal`] runs
    /// in-process and returns its result without `await`. App-host's
    /// sync bootstrap paths call this directly so they can build the
    /// [`WorkspaceComputeProfile`](reimagine_inference::WorkspaceComputeProfile)
    /// without spinning up a runtime. The async
    /// [`BackendProfileProvider::backend_profile`] trait method
    /// forwards to this implementation.
    pub fn probe(&self) -> BackendProfile {
        let backend = Self::backend_kind();
        let (plugin, extension) = Self::plugin_provenance();
        let capabilities = Self::v1_capabilities();

        let cpu_device = DeviceProfile::new("cpu").with_kind(DeviceKind::Cpu);
        let cpu_instance = BackendInstanceProfile::new(
            BackendInstance::new(CPU_INSTANCE),
            backend.clone(),
            cpu_device,
            BackendInstanceStatus::Available,
        );
        let cpu_instance = Self::attach_capabilities(cpu_instance, &capabilities);

        let metal_instance = probe_metal_instance(&backend, &capabilities);

        BackendProfile::new(backend)
            .with_plugin(plugin, extension)
            .with_instance(cpu_instance)
            .with_instance(metal_instance)
    }
}

#[async_trait::async_trait]
impl BackendProfileProvider for CandleProfileProvider {
    async fn backend_profile(&self) -> BackendProfile {
        self.probe()
    }
}

fn probe_metal_instance(
    backend: &Backend,
    capabilities: &[InferenceCapability],
) -> BackendInstanceProfile {
    let device = DeviceProfile::new("metal").with_kind(DeviceKind::Gpu);
    let instance_id = BackendInstance::new(METAL_INSTANCE);

    match Device::new_metal(0) {
        Ok(_device) => CandleProfileProvider::attach_capabilities(
            BackendInstanceProfile::new(
                instance_id,
                backend.clone(),
                device,
                BackendInstanceStatus::Available,
            ),
            capabilities,
        ),
        Err(err) => {
            let reason = err.to_string();
            CandleProfileProvider::attach_capabilities(
                BackendInstanceProfile::new(
                    instance_id,
                    backend.clone(),
                    device,
                    BackendInstanceStatus::Unavailable,
                ),
                capabilities,
            )
            .with_diagnostic(candle_device_unavailable("metal", &reason))
        }
    }
}

#[cfg(test)]
mod tests;
