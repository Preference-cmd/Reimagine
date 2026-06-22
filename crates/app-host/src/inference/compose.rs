use std::sync::Arc;

use reimagine_config::{AppConfig, InferenceBackendConfig, InferenceBackendKind};
use reimagine_inference::registry::register_builtin_inference_executors;
use reimagine_inference::{
    DefaultInferenceRuntime, InferenceBackend, InferenceBackendRegistry, InferenceRuntime,
    RejectAllBridgePolicy,
};
use reimagine_inference_candle::{
    CandleBackend, CandleBackendConfig, CandleBackendError, CandleDevice, CandleRunResourceBackend,
};
use reimagine_runtime::NodeExecutorRegistry;

use crate::ModelService;
use crate::inference::resolver::ModelResolverAdapter;

#[derive(Debug)]
pub(crate) struct ComposedBackends {
    registry: InferenceBackendRegistry,
    resource_backend: CandleRunResourceBackend,
}

#[derive(Debug)]
pub(crate) struct ComposedInferenceRuntime {
    pub(crate) executor_registry: NodeExecutorRegistry,
    pub(crate) resource_backend: CandleRunResourceBackend,
}

pub(crate) fn compose_inference_runtime(
    config: &AppConfig,
    backend_config: &InferenceBackendConfig,
    model_service: Arc<ModelService>,
) -> Result<ComposedInferenceRuntime, CandleBackendError> {
    let composed_backends = compose_inference_backends(config, backend_config)?;
    let inference_runtime = compose_runtime_router(composed_backends.registry);

    let mut executor_registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(
        &mut executor_registry,
        inference_runtime,
        Arc::new(ModelResolverAdapter::new(
            model_service,
            config.paths().clone(),
        )),
    )
    .expect("register executors");

    Ok(ComposedInferenceRuntime {
        executor_registry,
        resource_backend: composed_backends.resource_backend,
    })
}

fn compose_inference_backends(
    config: &AppConfig,
    backend_config: &InferenceBackendConfig,
) -> Result<ComposedBackends, CandleBackendError> {
    let mut registry = InferenceBackendRegistry::new();
    let resource_backend = match backend_config.backend {
        InferenceBackendKind::Candle => {
            let backend = build_candle_backend(config, backend_config)?;
            let device_label = backend.device_label().to_string();
            let resource_backend = backend.resource_backend();
            let backend: Arc<dyn InferenceBackend> = backend;
            let instance =
                reimagine_inference::BackendInstance::new(format!("candle:{device_label}"));
            let descriptor = reimagine_inference::BackendInstanceDescriptor::new(
                instance,
                backend.backend_kind().clone(),
            )
            .with_device(reimagine_inference::DeviceProfile::new(device_label));
            registry.register(descriptor, backend);
            resource_backend
        }
    };

    Ok(ComposedBackends {
        registry,
        resource_backend,
    })
}

fn build_candle_backend(
    config: &AppConfig,
    backend_config: &InferenceBackendConfig,
) -> Result<Arc<CandleBackend>, CandleBackendError> {
    let device = CandleDevice::new(&backend_config.candle_device);
    let candle_config = CandleBackendConfig::new(
        config.paths().models_dir().to_path_buf(),
        config.paths().output_dir().to_path_buf(),
    )
    .with_device(device);
    Ok(Arc::new(CandleBackend::new(candle_config)?))
}

fn compose_runtime_router(registry: InferenceBackendRegistry) -> Arc<dyn InferenceRuntime> {
    Arc::new(DefaultInferenceRuntime::new(
        Arc::new(registry),
        Arc::new(RejectAllBridgePolicy),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_config::{AppPaths, InferenceBackendKind};
    use reimagine_inference_candle::CandleBackendError;

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-app-host-compose-{prefix}-{nonce}"))
    }

    #[test]
    fn compose_backends_registers_selected_candle_backend() {
        let base = temp_dir("selected-backend");
        let config = AppConfig::new(AppPaths::new(&base));
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            ..InferenceBackendConfig::default()
        };

        let composed = compose_inference_backends(&config, &backend_config).expect("backends");

        assert_eq!(
            composed.registry.len(),
            1,
            "selected backend should register once"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn compose_backends_applies_selected_candle_device() {
        let base = temp_dir("selected-device");
        let config = AppConfig::new(AppPaths::new(&base));
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "tpu".to_string(),
            ..InferenceBackendConfig::default()
        };

        let err = compose_inference_backends(&config, &backend_config)
            .expect_err("invalid selected Candle device should be rejected");

        match err {
            CandleBackendError::DeviceUnavailable { requested, .. } => {
                assert_eq!(requested, "tpu");
            }
            other => panic!("expected DeviceUnavailable, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}
