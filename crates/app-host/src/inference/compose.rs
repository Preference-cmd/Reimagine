use std::sync::Arc;

use reimagine_config::{AppConfig, InferenceBackendConfig, InferenceBackendKind};
use reimagine_inference::registry::register_builtin_inference_executors;
use reimagine_inference::{
    BackendInstance, BackendInstanceDescriptor, BackendOverrides, DefaultInferenceRuntime,
    DeviceProfile, InferenceBackend, InferenceBackendRegistry, RejectAllBridgePolicy,
    StaticBackendSelectionPolicy,
};
use reimagine_inference_candle::{
    CandleBackend, CandleBackendConfig, CandleBackendError, CandleDevice, CandleResourceMechanism,
};
use reimagine_plugin::{Extension, Plugin};
use reimagine_runtime::NodeExecutorRegistry;

use crate::ModelService;
use crate::inference::resolver::ModelResolverAdapter;

#[derive(Debug)]
pub(crate) struct ComposedBackends {
    registry: InferenceBackendRegistry,
    resource_backend: CandleResourceMechanism,
    selected_instance: BackendInstance,
}

#[derive(Debug)]
pub(crate) struct ComposedInferenceRuntime {
    pub(crate) executor_registry: NodeExecutorRegistry,
    pub(crate) resource_backend: CandleResourceMechanism,
}

pub(crate) fn compose_inference_runtime(
    config: &AppConfig,
    backend_config: &InferenceBackendConfig,
    model_service: Arc<ModelService>,
) -> Result<ComposedInferenceRuntime, CandleBackendError> {
    let composed_backends = compose_inference_backends(config, backend_config)?;
    let inference_runtime = compose_runtime_router(
        composed_backends.registry,
        composed_backends.selected_instance,
    );

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
    let (resource_backend, selected_instance) = match backend_config.backend {
        InferenceBackendKind::Candle => {
            let backend = build_candle_backend(config, backend_config)?;
            let device_label = backend.device_label().to_string();
            let plugin = Plugin::try_from("builtin.candle").expect("valid built-in plugin id");
            let extension =
                Extension::try_from("backend.candle").expect("valid built-in extension id");
            let device = DeviceProfile::new(&device_label);
            let resource_backend = backend.resource_mechanism(
                Some(plugin.clone()),
                Some(extension.clone()),
                Some(device.clone()),
            );
            let backend: Arc<dyn InferenceBackend> = backend;
            let instance = BackendInstance::new(format!("candle:{device_label}"));
            let descriptor =
                BackendInstanceDescriptor::new(instance.clone(), backend.backend_kind().clone())
                    .with_device(device)
                    .with_plugin(plugin, extension);
            registry.register(descriptor, backend);
            (resource_backend, instance)
        }
    };

    Ok(ComposedBackends {
        registry,
        resource_backend,
        selected_instance,
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

fn compose_runtime_router(
    registry: InferenceBackendRegistry,
    selected_instance: BackendInstance,
) -> Arc<DefaultInferenceRuntime> {
    let policy = StaticBackendSelectionPolicy::with_overrides(
        BackendOverrides::new(),
        vec![selected_instance.clone()],
        Some(vec![selected_instance]),
        Vec::new(),
    );
    Arc::new(DefaultInferenceRuntime::with_policy(
        Arc::new(registry),
        Arc::new(policy),
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
        assert_eq!(
            composed.selected_instance,
            reimagine_inference::BackendInstance::new("candle:cpu")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn compose_backends_attaches_builtin_candle_plugin_provenance() {
        let base = temp_dir("plugin-provenance");
        let config = AppConfig::new(AppPaths::new(&base));
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            ..InferenceBackendConfig::default()
        };

        let composed = compose_inference_backends(&config, &backend_config).expect("backends");
        let descriptors = composed.registry.descriptors();
        let descriptor = descriptors.first().expect("registered descriptor");

        assert_eq!(descriptor.instance, composed.selected_instance);
        assert_eq!(
            descriptor.plugin.as_ref().map(|p| p.as_str()),
            Some("builtin.candle")
        );
        assert_eq!(
            descriptor.extension.as_ref().map(|e| e.as_str()),
            Some("backend.candle")
        );
        assert_eq!(
            descriptor.device.as_ref().map(|d| d.label.as_str()),
            Some("cpu")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn compose_runtime_router_uses_config_projected_backend_policy() {
        let instance = reimagine_inference::BackendInstance::new("candle:cpu");
        let registry = InferenceBackendRegistry::new();
        let runtime = compose_runtime_router(registry, instance.clone());
        let request = reimagine_inference::BackendSelectionRequest {
            capability: reimagine_inference::InferenceCapability::LoadBundle,
            node_id: None,
            affinities: Vec::new(),
            registered: Vec::new(),
            explicit_override: None,
        };

        assert_eq!(
            runtime.selection_policy().candidates(&request),
            vec![instance.clone()]
        );
        assert!(
            runtime
                .selection_policy()
                .allows_explicit_override(&instance, &request)
        );
        assert!(!runtime.selection_policy().allows_explicit_override(
            &reimagine_inference::BackendInstance::new("candle:metal"),
            &request
        ));
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
