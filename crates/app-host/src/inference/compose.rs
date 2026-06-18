use std::sync::Arc;

use reimagine_config::{AppConfig, InferenceBackendConfig};
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
pub(crate) struct ComposedInferenceRuntime {
    pub(crate) executor_registry: NodeExecutorRegistry,
    pub(crate) resource_backend: CandleRunResourceBackend,
}

pub(crate) fn compose_inference_runtime(
    config: &AppConfig,
    backend_config: &InferenceBackendConfig,
    model_service: Arc<ModelService>,
) -> Result<ComposedInferenceRuntime, CandleBackendError> {
    let candle_backend = build_candle_backend(config, backend_config)?;
    let inference_runtime = compose_runtime_router(Arc::clone(&candle_backend));
    let resource_backend = candle_backend.resource_backend();

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

fn compose_runtime_router(backend: Arc<CandleBackend>) -> Arc<dyn InferenceRuntime> {
    let backend: Arc<dyn InferenceBackend> = backend;
    let mut inference_registry = InferenceBackendRegistry::new();
    inference_registry.register(backend);
    Arc::new(DefaultInferenceRuntime::new(
        Arc::new(inference_registry),
        Arc::new(RejectAllBridgePolicy),
    ))
}
