use std::sync::Arc;

use reimagine_config::AppConfig;
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceDescriptor, BackendInstanceRuntimeHooks,
    BackendProfile, DeviceProfile, InferenceBackend,
};
use reimagine_inference_candle::{
    CandleBackend, CandleBackendConfig, CandleBackendError, CandleDevice, CandleProfileProvider,
};
use reimagine_plugin::{Extension, Plugin};

#[derive(Debug)]
pub(crate) enum BackendCandidateError {
    Candle(CandleBackendError),
}

impl std::fmt::Display for BackendCandidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Candle(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for BackendCandidateError {}

impl From<CandleBackendError> for BackendCandidateError {
    fn from(value: CandleBackendError) -> Self {
        Self::Candle(value)
    }
}

pub(crate) struct BuiltBackendInstance {
    pub(crate) descriptor: BackendInstanceDescriptor,
    pub(crate) backend: Arc<dyn InferenceBackend>,
    pub(crate) runtime_hooks: Arc<dyn BackendInstanceRuntimeHooks>,
}

pub(crate) trait BackendCandidate: Send + Sync {
    fn backend(&self) -> Backend;

    fn profile(&self) -> BackendProfile;

    fn build(
        &self,
        config: &AppConfig,
        instance: &BackendInstance,
        device: Option<DeviceProfile>,
    ) -> Result<BuiltBackendInstance, BackendCandidateError>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CandleBackendCandidate;

impl CandleBackendCandidate {
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl BackendCandidate for CandleBackendCandidate {
    fn backend(&self) -> Backend {
        Backend::new("candle")
    }

    fn profile(&self) -> BackendProfile {
        CandleProfileProvider::new().probe()
    }

    fn build(
        &self,
        config: &AppConfig,
        instance: &BackendInstance,
        device: Option<DeviceProfile>,
    ) -> Result<BuiltBackendInstance, BackendCandidateError> {
        let device_label = instance
            .as_str()
            .split_once(':')
            .map(|(_, label)| label)
            .unwrap_or("cpu");
        let candle_config = CandleBackendConfig::new(
            config.paths().models_dir().to_path_buf(),
            config.paths().output_dir().to_path_buf(),
        )
        .with_device(CandleDevice::new(device_label));
        let backend = Arc::new(CandleBackend::new(candle_config)?);
        let plugin = Plugin::try_from("builtin.candle").expect("valid built-in plugin id");
        let extension = Extension::try_from("backend.candle").expect("valid built-in extension id");
        let descriptor =
            BackendInstanceDescriptor::new(instance.clone(), backend.backend_kind().clone())
                .with_plugin(plugin.clone(), extension.clone());
        let descriptor = if let Some(device) = device.clone() {
            descriptor.with_device(device)
        } else {
            descriptor
        };
        let runtime_hooks = backend.runtime_hooks(Some(plugin), Some(extension), device);
        let backend: Arc<dyn InferenceBackend> = backend;
        Ok(BuiltBackendInstance {
            descriptor,
            backend,
            runtime_hooks: Arc::new(runtime_hooks),
        })
    }
}

pub(crate) fn builtin_backend_candidates() -> Vec<Arc<dyn BackendCandidate>> {
    vec![Arc::new(CandleBackendCandidate::new())]
}
