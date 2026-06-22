use std::path::Path;
use std::sync::Arc;

use reimagine_agent::{AgentToolRegistry, WorkspaceScope};
use reimagine_config::{AppConfig, AppPaths, ConfigDocument, InferenceBackendConfig};
use reimagine_core::model::NodeDef;
use reimagine_nodes::BuiltinNodeCatalog;
use reimagine_runtime::{BoxedRunEventSink, RuntimeService, VecRunEventSink};

use crate::inference::compose::compose_inference_runtime;
use crate::node_catalog::{NodeCatalogAlignment, NodeCatalogService};
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
    node_catalog: Arc<NodeCatalogService>,
    builtin_catalog: Arc<BuiltinNodeCatalog>,
    services: Arc<WorkspaceServices>,
}

impl WorkspaceHost {
    pub fn new(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        runtime_service: Arc<RuntimeService>,
        builtin_catalog: Arc<BuiltinNodeCatalog>,
    ) -> Self {
        let config = Arc::new(config);
        let workflow_service = Arc::new(WorkflowService::new(config.paths().clone()));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));
        let backend = BackendSelection::from(backend_config.backend);
        let node_catalog = Arc::new(NodeCatalogService::new(
            Arc::clone(&builtin_catalog),
            backend,
        ));
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
            builtin_catalog,
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
        Self::try_with_defaults_and_event_sink(
            workspace_scope,
            base_path,
            Arc::new(VecRunEventSink::new()),
        )
        .await
    }

    pub async fn try_with_defaults_and_event_sink(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        event_sink: BoxedRunEventSink,
    ) -> Result<Self, AppHostError> {
        let base_path = base_path.into();
        let config = AppConfig::new(AppPaths::new(&base_path));
        let backend_config = load_backend_config_result(&config).await?;
        Ok(Self::with_backend_config_inner(
            workspace_scope,
            config,
            backend_config,
            event_sink,
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
        let composed = compose_inference_runtime(&config, &backend_config, model_service.clone())
            .expect("backend");
        let runtime_service = Arc::new(RuntimeService::new(
            composed.executor_registry,
            Arc::new(composed.runtime_hooks),
            event_sink,
            Arc::new(reimagine_runtime::SystemClock),
        ));
        let builtin_catalog = Arc::new(BuiltinNodeCatalog::v1());
        Self::new(
            workspace_scope,
            config,
            backend_config,
            runtime_service,
            builtin_catalog,
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
    pub fn node_catalog(&self) -> &Arc<NodeCatalogService> {
        &self.node_catalog
    }

    /// Borrow the underlying built-in catalog handle.
    ///
    /// Most host adapters should use [`Self::node_catalog`] and the
    /// `NodeCatalogService` host-neutral list/fetch helpers instead of
    /// reading the catalog directly. This accessor is kept for callers
    /// (such as tests) that need direct access to the V1
    /// [`BuiltinNodeCatalog`].
    pub fn builtin_node_catalog(&self) -> &Arc<BuiltinNodeCatalog> {
        &self.builtin_catalog
    }

    /// List every `NodeDef` exposed by the workspace catalog.
    pub fn list_node_defs(&self) -> Vec<NodeDef> {
        self.node_catalog.list_node_defs()
    }

    /// Fetch a single `NodeDef` by `NodeTypeId` from the workspace catalog.
    pub fn find_node_def(&self, type_id: &reimagine_core::model::NodeTypeId) -> Option<NodeDef> {
        self.node_catalog.find_node_def(type_id)
    }

    /// Compute the alignment report between the workspace catalog and
    /// the runtime executor registry.
    pub fn check_node_catalog_alignment(&self) -> NodeCatalogAlignment {
        self.node_catalog
            .check_alignment(self.runtime_service.registry())
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

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_config::InferenceBackendKind;
    use reimagine_core::model::NodeTypeId;
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

    #[test]
    fn app_host_inference_composition_registers_builtin_executors() {
        let base = temp_dir("compose-runtime");
        let config = AppConfig::new(reimagine_config::AppPaths::new(&base));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));

        let composed = super::compose_inference_runtime(
            &config,
            &InferenceBackendConfig::default(),
            Arc::clone(&model_service),
        )
        .expect("compose inference runtime");

        assert!(
            composed
                .executor_registry
                .get(&NodeTypeId::new("builtin.checkpoint_loader"))
                .is_some(),
            "checkpoint loader executor should be registered"
        );
        assert!(
            composed
                .executor_registry
                .get(&NodeTypeId::new("builtin.ksampler"))
                .is_some(),
            "ksampler executor should be registered"
        );
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
