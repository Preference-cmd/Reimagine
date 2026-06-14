use std::path::Path;
use std::sync::Arc;

use reimagine_agent::{AgentToolRegistry, WorkspaceScope};
use reimagine_config::{AppConfig, AppPaths};
use reimagine_core::model::ModelRef;
use reimagine_inference::registry::register_builtin_inference_executors;
use reimagine_inference::{InferenceError, ModelResolver, ResolvedInferenceModel};
use reimagine_inference_candle::{CandleBackend, CandleBackendConfig, CandleBackendError};
use reimagine_nodes::BuiltinNodeCatalog;
use reimagine_runtime::{BoxedRunEventSink, NodeExecutorRegistry, RuntimeService, VecRunEventSink};

use crate::services::WorkspaceServices;
use crate::tools::register_app_tools;
use crate::{AgentService, BackendSelection, ModelService, WorkflowService};

#[derive(Debug)]
pub struct WorkspaceHost {
    workspace_scope: WorkspaceScope,
    config: Arc<AppConfig>,
    workflow_service: Arc<WorkflowService>,
    model_service: Arc<ModelService>,
    runtime_service: Arc<RuntimeService>,
    agent_service: Arc<AgentService>,
    node_catalog: Arc<BuiltinNodeCatalog>,
    services: Arc<WorkspaceServices>,
}

impl WorkspaceHost {
    pub fn new(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        runtime_service: Arc<RuntimeService>,
        node_catalog: Arc<BuiltinNodeCatalog>,
    ) -> Self {
        let config = Arc::new(config);
        let workflow_service = Arc::new(WorkflowService::new(config.paths().clone()));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));
        let services = Arc::new(WorkspaceServices::new(
            workspace_scope.clone(),
            Arc::clone(&config),
            Arc::clone(&workflow_service),
            Arc::clone(&model_service),
            Arc::clone(&runtime_service),
            Arc::clone(&node_catalog),
        ));
        let mut registry = AgentToolRegistry::new();
        register_app_tools(&mut registry, Arc::clone(&services));
        let registry = Arc::new(registry);
        let agent_service = Arc::new(AgentService::with_registry(
            workspace_scope.clone(),
            Arc::clone(&registry),
        ));
        Self {
            workspace_scope,
            config,
            workflow_service,
            model_service,
            runtime_service,
            agent_service,
            node_catalog,
            services,
        }
    }

    pub fn with_defaults(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self::with_defaults_and_backend(
            workspace_scope,
            base_path,
            BackendSelection::Candle,
            Arc::new(VecRunEventSink::new()),
        )
    }

    pub fn with_defaults_and_event_sink(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        event_sink: BoxedRunEventSink,
    ) -> Self {
        Self::with_defaults_and_backend(
            workspace_scope,
            base_path,
            BackendSelection::Candle,
            event_sink,
        )
    }

    pub fn with_defaults_and_backend(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        _backend_selection: BackendSelection,
        event_sink: BoxedRunEventSink,
    ) -> Self {
        let config = AppConfig::new(AppPaths::new(base_path));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));
        let candle_backend = build_candle_backend(config.paths()).expect("backend");
        let backend: Arc<dyn reimagine_inference::InferenceBackend> = candle_backend.clone();
        let resource_backend = candle_backend.resource_backend();
        let mut registry = NodeExecutorRegistry::default();
        register_builtin_inference_executors(
            &mut registry,
            backend,
            Arc::new(ModelResolverAdapter::new(
                model_service.clone(),
                config.paths().clone(),
            )),
        )
        .expect("register executors");
        let runtime_service = Arc::new(RuntimeService::new(
            registry,
            Arc::new(resource_backend),
            event_sink,
            Arc::new(reimagine_runtime::SystemClock),
        ));
        let node_catalog = Arc::new(BuiltinNodeCatalog::v1());
        Self::new(workspace_scope, config, runtime_service, node_catalog)
    }

    pub fn workspace_scope(&self) -> &WorkspaceScope {
        &self.workspace_scope
    }
    pub fn base_path(&self) -> &Path {
        self.config.paths().base_path()
    }
    pub fn config(&self) -> &Arc<AppConfig> {
        &self.config
    }
    pub fn workflow_service(&self) -> &Arc<WorkflowService> {
        &self.workflow_service
    }
    pub fn model_service(&self) -> &Arc<ModelService> {
        &self.model_service
    }
    pub fn runtime_service(&self) -> &Arc<RuntimeService> {
        &self.runtime_service
    }
    pub fn agent_service(&self) -> &Arc<AgentService> {
        &self.agent_service
    }
    pub fn node_catalog(&self) -> &Arc<BuiltinNodeCatalog> {
        &self.node_catalog
    }
    pub fn services(&self) -> &Arc<WorkspaceServices> {
        &self.services
    }
}

fn build_candle_backend(app_paths: &AppPaths) -> Result<Arc<CandleBackend>, CandleBackendError> {
    let config = CandleBackendConfig::new(app_paths.models_dir().to_path_buf());
    Ok(Arc::new(CandleBackend::new(config)?))
}

struct ModelResolverAdapter {
    model_service: Arc<ModelService>,
    app_paths: AppPaths,
}

impl ModelResolverAdapter {
    fn new(model_service: Arc<ModelService>, app_paths: AppPaths) -> Self {
        Self {
            model_service,
            app_paths,
        }
    }
}

#[async_trait::async_trait]
impl ModelResolver for ModelResolverAdapter {
    async fn resolve(
        &self,
        model_ref: &ModelRef,
    ) -> Result<ResolvedInferenceModel, InferenceError> {
        let resolution = self
            .model_service
            .resolve_descriptor(model_ref)
            .await
            .map_err(|error| InferenceError::ModelResolutionFailed {
                message: error.to_string(),
            })?;

        let Some(descriptor) = resolution.into_value() else {
            return Err(InferenceError::ModelResolutionFailed {
                message: format!("model ref {} could not be resolved", model_ref.id()),
            });
        };

        let manifest = self.model_service.cached_manifest().ok_or_else(|| {
            InferenceError::ModelResolutionFailed {
                message: "model manifest not cached after resolution".to_string(),
            }
        })?;

        let source_path = reimagine_model_manager::resolve_source_path(
            &manifest,
            descriptor.source(),
            self.app_paths.models_dir(),
        )
        .ok_or_else(|| InferenceError::ModelResolutionFailed {
            message: format!(
                "could not resolve source path for model {}",
                descriptor.id()
            ),
        })?;

        Ok(ResolvedInferenceModel::new(
            descriptor.id().clone(),
            descriptor.model_series().clone(),
            descriptor.variant().clone(),
            model_ref.role(),
            source_path,
            map_model_format(descriptor.format()),
        ))
    }
}

fn map_model_format(
    format: reimagine_model_manager::ModelFormat,
) -> reimagine_inference::ModelFormat {
    match format {
        reimagine_model_manager::ModelFormat::Safetensors => {
            reimagine_inference::ModelFormat::SafeTensors
        }
        reimagine_model_manager::ModelFormat::Gguf => reimagine_inference::ModelFormat::Gguf,
        reimagine_model_manager::ModelFormat::Ckpt => reimagine_inference::ModelFormat::PyTorch,
        reimagine_model_manager::ModelFormat::Unknown => reimagine_inference::ModelFormat::Other,
    }
}
