use std::sync::Arc;

use reimagine_config::AppConfig;
use reimagine_inference::registry::register_builtin_inference_executors;
use reimagine_inference::{
    BackendInstance, BackendInstanceDescriptor, BackendOverrides, DefaultInferenceRuntime,
    DeviceProfile, InferenceBackend, InferenceBackendRegistry, RejectAllBridgePolicy,
    StaticBackendSelectionPolicy,
};
use reimagine_inference_candle::{
    CandleBackend, CandleBackendConfig, CandleBackendError, CandleBackendInstanceRuntimeHooks,
    CandleDevice,
};
use reimagine_plugin::{Extension, Plugin};
use reimagine_runtime::NodeExecutorRegistry;

use crate::ModelService;
use crate::inference::resolver::ModelResolverAdapter;

#[derive(Debug)]
pub(crate) struct ComposedBackends {
    registry: InferenceBackendRegistry,
    runtime_hooks: CandleBackendInstanceRuntimeHooks,
    selected_instance: BackendInstance,
}

#[derive(Debug)]
pub(crate) struct ComposedInferenceRuntime {
    pub(crate) executor_registry: NodeExecutorRegistry,
    pub(crate) runtime_hooks: CandleBackendInstanceRuntimeHooks,
}

/// Construct the inference runtime for a workspace.
///
/// `resolved_label` is the post-resolution Candle device label
/// (e.g. `"cpu"` or `"metal"`) produced by the app-host
/// [`resolve`](super::resolve) step. Validation against the live
/// profile and any fallback decision happens upstream — by the time
/// this function runs the label is guaranteed to map to a Candle
/// [`Device`]. The label drives both the constructed
/// [`CandleBackend`] and the registered [`BackendInstance`]
/// identity.
pub(crate) fn compose_inference_runtime(
    config: &AppConfig,
    resolved_label: &str,
    model_service: Arc<ModelService>,
) -> Result<ComposedInferenceRuntime, CandleBackendError> {
    let composed_backends = compose_inference_backends(config, resolved_label)?;
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
        runtime_hooks: composed_backends.runtime_hooks,
    })
}

/// Construct the Candle backend registry using the resolved device label.
///
/// The label has already been normalized and matched against the live
/// profile by app-host — `mps` was rewritten to `metal`, unknown or
/// unavailable labels were rewritten to `cpu`. This function trusts
/// the resolved label and only calls into the concrete Candle backend
/// to construct the device.
fn compose_inference_backends(
    config: &AppConfig,
    resolved_label: &str,
) -> Result<ComposedBackends, CandleBackendError> {
    let mut registry = InferenceBackendRegistry::new();
    let backend = build_candle_backend(config, resolved_label)?;
    let device_label = backend.device_label().to_string();
    let plugin = Plugin::try_from("builtin.candle").expect("valid built-in plugin id");
    let extension = Extension::try_from("backend.candle").expect("valid built-in extension id");
    let device = DeviceProfile::new(&device_label);
    let runtime_hooks = backend.runtime_hooks(
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

    Ok(ComposedBackends {
        registry,
        runtime_hooks,
        selected_instance: instance,
    })
}

fn build_candle_backend(
    config: &AppConfig,
    resolved_label: &str,
) -> Result<Arc<CandleBackend>, CandleBackendError> {
    let device = CandleDevice::new(resolved_label);
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
    use reimagine_config::AppPaths;

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-app-host-compose-{prefix}-{nonce}"))
    }

    #[test]
    fn compose_backends_registers_cpu_instance_for_resolved_cpu_label() {
        let base = temp_dir("resolved-cpu");
        let config = AppConfig::new(AppPaths::new(&base));

        let composed = compose_inference_backends(&config, "cpu").expect("backends");

        assert_eq!(
            composed.registry.len(),
            1,
            "selected backend should register once"
        );
        assert_eq!(
            composed.selected_instance,
            BackendInstance::new("candle:cpu")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn compose_backends_attaches_builtin_candle_plugin_provenance() {
        let base = temp_dir("plugin-provenance");
        let config = AppConfig::new(AppPaths::new(&base));

        let composed = compose_inference_backends(&config, "cpu").expect("backends");
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
    fn compose_uses_metal_label_for_metal_instance() {
        let base = temp_dir("resolved-metal");
        let config = AppConfig::new(AppPaths::new(&base));

        let result = compose_inference_backends(&config, "metal");
        match result {
            Ok(composed) => {
                assert_eq!(
                    composed.selected_instance,
                    BackendInstance::new("candle:metal")
                );
            }
            Err(CandleBackendError::DeviceUnavailable { requested, .. }) => {
                assert_eq!(
                    requested, "metal",
                    "non-Metal hosts may reject direct resolved-metal composition"
                );
            }
            Err(other) => panic!("expected metal composition or DeviceUnavailable, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn compose_runtime_router_uses_config_projected_backend_policy() {
        let instance = BackendInstance::new("candle:cpu");
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
        assert!(
            !runtime
                .selection_policy()
                .allows_explicit_override(&BackendInstance::new("candle:metal"), &request)
        );
    }

    #[test]
    fn compose_runtime_accepts_resolved_label_without_backend_config() {
        // The new compose_inference_runtime signature accepts a
        // resolved device label instead of an `InferenceBackendConfig`.
        // Smoke-check that the new signature works end-to-end on the
        // resolved-cpu path that every existing test exercises.
        let base = temp_dir("resolved-label-smoke");
        let config = AppConfig::new(AppPaths::new(&base));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));

        let _ = compose_inference_runtime(&config, "cpu", model_service).expect("runtime");
        let _ = std::fs::remove_dir_all(&base);
    }
}
