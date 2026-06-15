use std::path::Path;
use std::sync::Arc;

use reimagine_agent::{AgentToolRegistry, WorkspaceScope};
use reimagine_config::{AppConfig, AppPaths, ConfigDocument, InferenceBackendConfig};
use reimagine_core::model::ModelRef;
use reimagine_inference::registry::register_builtin_inference_executors;
use reimagine_inference::{InferenceError, ModelResolver, ResolvedInferenceModel};
use reimagine_inference_candle::{
    CandleBackend, CandleBackendConfig, CandleBackendError, CandleDevice,
};
use reimagine_nodes::BuiltinNodeCatalog;
use reimagine_runtime::{BoxedRunEventSink, NodeExecutorRegistry, RuntimeService, VecRunEventSink};

use crate::services::WorkspaceServices;
use crate::tools::register_app_tools;
use crate::{AgentService, AppHostError, BackendSelection, ModelService, WorkflowService};

#[derive(Debug)]
pub struct WorkspaceHost {
    workspace_scope: WorkspaceScope,
    config: Arc<AppConfig>,
    backend_config: InferenceBackendConfig,
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
        backend_config: InferenceBackendConfig,
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
            backend_config,
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

    pub async fn try_with_defaults(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
    ) -> Result<Self, AppHostError> {
        let base_path = base_path.into();
        let config = AppConfig::new(AppPaths::new(&base_path));
        let backend_config = load_backend_config_result(&config).await?;
        Ok(Self::with_backend_config_inner(
            workspace_scope,
            config,
            backend_config,
            Arc::new(VecRunEventSink::new()),
        ))
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
        let backend_config = load_backend_config(&config);
        Self::with_backend_config_inner(workspace_scope, config, backend_config, event_sink)
    }

    pub fn with_backend_config(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        backend_config: InferenceBackendConfig,
        event_sink: BoxedRunEventSink,
    ) -> Self {
        let config = AppConfig::new(AppPaths::new(base_path));
        Self::with_backend_config_inner(workspace_scope, config, backend_config, event_sink)
    }

    fn with_backend_config_inner(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        event_sink: BoxedRunEventSink,
    ) -> Self {
        let model_service = Arc::new(ModelService::new(config.paths().clone()));
        let candle_backend =
            build_candle_backend(config.paths(), &backend_config).expect("backend");
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
        Self::new(
            workspace_scope,
            config,
            backend_config,
            runtime_service,
            node_catalog,
        )
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
    pub fn backend_config(&self) -> &InferenceBackendConfig {
        &self.backend_config
    }
}

fn load_backend_config(config: &AppConfig) -> InferenceBackendConfig {
    let path = config
        .paths()
        .config_dir()
        .join(InferenceBackendConfig::KEY);
    match std::fs::read_to_string(&path) {
        Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
        Err(_) => InferenceBackendConfig::default(),
    }
}

async fn load_backend_config_result(
    config: &AppConfig,
) -> reimagine_config::ConfigResult<InferenceBackendConfig> {
    let handle = config.config::<InferenceBackendConfig>()?;
    let (backend_config, _) = handle.load().await?;
    Ok(backend_config)
}

fn build_candle_backend(
    app_paths: &AppPaths,
    backend_config: &InferenceBackendConfig,
) -> Result<Arc<CandleBackend>, CandleBackendError> {
    let device = CandleDevice::new(&backend_config.candle_device);
    let candle_config =
        CandleBackendConfig::new(app_paths.models_dir().to_path_buf()).with_device(device);
    Ok(Arc::new(CandleBackend::new(candle_config)?))
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

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_config::InferenceBackendKind;
    use std::fs;

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tid = std::thread::current().id();
        std::env::temp_dir().join(format!("reimagine-app-host-ws-{prefix}-{nonce:?}-{tid:?}"))
    }

    #[test]
    fn workspace_with_defaults_uses_candle() {
        let base = temp_dir("defaults");
        let workspace = WorkspaceHost::with_defaults(WorkspaceScope::new("test-defaults"), &base);
        assert_eq!(workspace.base_path(), base);
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Candle
        );
        assert_eq!(workspace.backend_config().candle_device, "cpu");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn workspace_from_config_file_selects_candle() {
        let base = temp_dir("config-file");
        let config_dir = base.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(
            config_dir.join("inference_backend.json"),
            r#"{"backend": "candle", "candle_device": "cpu"}"#,
        )
        .unwrap();

        let workspace =
            WorkspaceHost::with_defaults(WorkspaceScope::new("test-config-file"), &base);
        assert_eq!(workspace.base_path(), base);
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Candle
        );
        assert_eq!(workspace.backend_config().candle_device, "cpu");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn missing_config_file_defaults_to_candle() {
        let base = temp_dir("no-config");
        let workspace = WorkspaceHost::with_defaults(WorkspaceScope::new("test-no-config"), &base);
        assert_eq!(workspace.base_path(), base);
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Candle
        );
        assert_eq!(workspace.backend_config().candle_device, "cpu");
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn invalid_config_json_returns_error() {
        let base = temp_dir("invalid-json");
        let config_dir = base.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(
            config_dir.join("inference_backend.json"),
            r#"{"backend": "nope"}"#,
        )
        .unwrap();

        let config = AppConfig::new(reimagine_config::AppPaths::new(&base));
        let result = load_backend_config_result(&config).await;
        assert!(result.is_err(), "invalid backend should return error");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("inference_backend.json") || msg.contains("not valid JSON"),
            "error should include config path, got: {msg}"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn malformed_json_returns_error() {
        let base = temp_dir("malformed-json");
        let config_dir = base.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(config_dir.join("inference_backend.json"), "not json at all").unwrap();

        let result =
            load_backend_config_result(&AppConfig::new(reimagine_config::AppPaths::new(&base)))
                .await;
        assert!(result.is_err(), "malformed json should return error");
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn try_with_defaults_missing_config_returns_ok_default() {
        let base = temp_dir("try-missing");
        let workspace =
            WorkspaceHost::try_with_defaults(WorkspaceScope::new("test-try-missing"), &base)
                .await
                .expect("missing config should succeed with defaults");
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Candle
        );
        assert_eq!(workspace.backend_config().candle_device, "cpu");
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn try_with_defaults_valid_config_returns_ok() {
        let base = temp_dir("try-valid");
        let config_dir = base.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(
            config_dir.join("inference_backend.json"),
            r#"{"backend": "candle", "candle_device": "cpu"}"#,
        )
        .unwrap();

        let workspace =
            WorkspaceHost::try_with_defaults(WorkspaceScope::new("test-try-valid"), &base)
                .await
                .expect("valid config should succeed");
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Candle
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn try_with_defaults_invalid_json_returns_error() {
        let base = temp_dir("try-invalid");
        let config_dir = base.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(
            config_dir.join("inference_backend.json"),
            r#"{"backend": "unsupported_backend"}"#,
        )
        .unwrap();

        let err = WorkspaceHost::try_with_defaults(WorkspaceScope::new("test-try-invalid"), &base)
            .await
            .expect_err("invalid config should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("inference_backend.json") || msg.contains("bootstrap"),
            "error should mention config file or bootstrap, got: {msg}"
        );
        let _ = fs::remove_dir_all(&base);
    }
}
