use std::path::Path;
use std::sync::Arc;

use reimagine_agent::{AgentEventSink, AgentToolRegistry, WorkspaceScope};
use reimagine_config::{AppConfig, AppPaths, ConfigDocument, InferenceBackendConfig};
use reimagine_core::model::NodeDef;
use reimagine_inference::WorkspaceComputeProfile;
use reimagine_nodes::BuiltinNodeCatalog;
use reimagine_runtime::{BoxedRunEventSink, RuntimeService, VecRunEventSink};

#[cfg(test)]
use crate::inference::compose::compose_inference_runtime;
use crate::inference::compose::{bootstrap_inference, bootstrap_inference_with_worker_inventory};
use crate::inference::selection::resolved_candle_device_label;
use crate::model_acquisition_service::ModelAcquisitionService;
use crate::node_catalog::{NodeCatalogAlignment, NodeCatalogService};
use crate::services::WorkspaceServices;
use crate::tools::register_app_tools;
use crate::{AgentService, AppHostError, BackendSelection, ModelService, WorkflowService};
use crate::{EmptyWorkerInventoryProvider, WorkerInventoryProvider};

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
    compute_profile: Arc<WorkspaceComputeProfile>,
    resolved_backend_instance: reimagine_inference::BackendInstance,
}

impl WorkspaceHost {
    pub fn new(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        runtime_service: Arc<RuntimeService>,
        builtin_catalog: Arc<BuiltinNodeCatalog>,
        compute_profile: Arc<WorkspaceComputeProfile>,
        resolved_backend_instance: reimagine_inference::BackendInstance,
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
            Arc::new(ModelAcquisitionService::new(
                config.paths().clone(),
                &config,
            )),
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
            compute_profile,
            resolved_backend_instance,
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

    /// Same as [`try_with_defaults_and_event_sink`] but also injects an
    /// `AgentEventSink` into the agent service.
    pub async fn try_with_defaults_and_event_sinks(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        event_sink: BoxedRunEventSink,
        agent_event_sink: Arc<dyn AgentEventSink>,
    ) -> Result<Self, AppHostError> {
        let base_path = base_path.into();
        let config = AppConfig::new(AppPaths::new(&base_path));
        let backend_config = load_backend_config_result(&config).await?;
        Self::with_backend_config_and_worker_inventory_inner(
            workspace_scope,
            config,
            backend_config,
            event_sink,
            agent_event_sink,
            Arc::new(EmptyWorkerInventoryProvider),
        )
        .await
    }

    pub async fn try_with_backend_config_and_worker_inventory(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        backend_config: InferenceBackendConfig,
        event_sink: BoxedRunEventSink,
        worker_inventory: Arc<dyn WorkerInventoryProvider>,
    ) -> Result<Self, AppHostError> {
        let config = AppConfig::new(AppPaths::new(base_path));
        Self::with_backend_config_and_worker_inventory_inner(
            workspace_scope,
            config,
            backend_config,
            event_sink,
            Arc::new(reimagine_agent::VecAgentEventSink::new()),
            worker_inventory,
        )
        .await
    }

    /// Bootstrap asynchronously with a run event sink but the default
    /// [`VecAgentEventSink`] for agent events.
    pub async fn try_with_defaults_and_event_sink(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        event_sink: BoxedRunEventSink,
    ) -> Result<Self, AppHostError> {
        Self::try_with_defaults_and_event_sinks(
            workspace_scope,
            base_path,
            event_sink,
            Arc::new(reimagine_agent::VecAgentEventSink::new()),
        )
        .await
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
        Self::with_backend_config_inner(
            workspace_scope,
            config,
            backend_config,
            event_sink,
            Arc::new(reimagine_agent::VecAgentEventSink::new()) as Arc<dyn AgentEventSink>,
        )
    }

    pub fn with_backend_config(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        backend_config: InferenceBackendConfig,
        event_sink: BoxedRunEventSink,
    ) -> Self {
        let config = AppConfig::new(AppPaths::new(base_path));
        Self::with_backend_config_inner(
            workspace_scope,
            config,
            backend_config,
            event_sink,
            Arc::new(reimagine_agent::VecAgentEventSink::new()) as Arc<dyn AgentEventSink>,
        )
    }

    fn with_backend_config_inner(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        event_sink: BoxedRunEventSink,
        agent_event_sink: Arc<dyn AgentEventSink>,
    ) -> Self {
        let model_service = Arc::new(ModelService::new(config.paths().clone()));

        let bootstrapped = bootstrap_inference(&config, &backend_config, model_service.clone())
            .expect("default backend composition should build a usable Candle fallback");
        let runtime_service = Arc::new(RuntimeService::new(
            bootstrapped.runtime.executor_registry,
            bootstrapped.runtime.runtime_hooks.clone(),
            event_sink,
            Arc::new(reimagine_runtime::SystemClock),
        ));
        let builtin_catalog = Arc::new(BuiltinNodeCatalog::v1());
        let mut host = Self::new(
            workspace_scope,
            config,
            backend_config,
            runtime_service,
            builtin_catalog,
            Arc::new(bootstrapped.compute_profile),
            bootstrapped.runtime.selected_instance,
        );
        // Replace the default AgentService with one that uses the injected event sink
        let registry = host.agent_service.registry().clone();
        let providers = host.agent_service.providers().clone();
        host.agent_service = Arc::new(AgentService::with_registry_providers_and_sink(
            host.workspace_scope.clone(),
            registry,
            providers,
            agent_event_sink,
        ));
        host
    }

    async fn with_backend_config_and_worker_inventory_inner(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        event_sink: BoxedRunEventSink,
        agent_event_sink: Arc<dyn AgentEventSink>,
        worker_inventory: Arc<dyn WorkerInventoryProvider>,
    ) -> Result<Self, AppHostError> {
        let model_service = Arc::new(ModelService::new(config.paths().clone()));
        let bootstrapped = bootstrap_inference_with_worker_inventory(
            &config,
            &backend_config,
            model_service,
            worker_inventory,
        )
        .await
        .map_err(|error| AppHostError::InferenceBootstrap {
            message: error.to_string(),
        })?;
        let runtime_service = Arc::new(RuntimeService::new(
            bootstrapped.runtime.executor_registry,
            bootstrapped.runtime.runtime_hooks.clone(),
            event_sink,
            Arc::new(reimagine_runtime::SystemClock),
        ));
        let builtin_catalog = Arc::new(BuiltinNodeCatalog::v1());
        let mut host = Self::new(
            workspace_scope,
            config,
            backend_config,
            runtime_service,
            builtin_catalog,
            Arc::new(bootstrapped.compute_profile),
            bootstrapped.runtime.selected_instance,
        );
        let registry = host.agent_service.registry().clone();
        let providers = host.agent_service.providers().clone();
        host.agent_service = Arc::new(AgentService::with_registry_providers_and_sink(
            host.workspace_scope.clone(),
            registry,
            providers,
            agent_event_sink,
        ));
        Ok(host)
    }

    pub fn with_agent_event_sink(self, event_sink: Arc<dyn AgentEventSink>) -> Self {
        let registry = self.agent_service.registry().clone();
        let providers = self.agent_service.providers().clone();
        let agent_service = Arc::new(AgentService::with_registry_providers_and_sink(
            self.workspace_scope.clone(),
            registry,
            providers,
            event_sink,
        ));
        Self {
            agent_service,
            ..self
        }
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

    /// Return the workspace's most recent compute profile.
    ///
    /// The profile aggregates one [`BackendProfile`] per registered
    /// backend (V1: only `candle`) plus a top-level diagnostics
    /// collection that records any fallback decisions made during
    /// bootstrap (e.g. `mps` → `cpu` when Metal is unavailable, or any
    /// unknown configured device label).
    ///
    /// The accessor returns a clone of the cached profile and does not
    /// re-probe the host; the snapshot is computed once during
    /// [`WorkspaceHost`] bootstrap.
    pub fn compute_profile(&self) -> WorkspaceComputeProfile {
        (*self.compute_profile).clone()
    }

    /// Return the host-neutral
    /// [`crate::dto::ComputeProfileDto`] projection of the workspace's
    /// most recent compute profile.
    ///
    /// Use this from HTTP / IPC adapters so the wire shape does not
    /// depend on inference-internal enums (`DeviceKind`,
    /// `BackendInstanceStatus`, `InferenceCapability`, …) or on
    /// backend-native handles. The returned DTO is the V1 wire contract
    /// for `GET /compute-profile` and the equivalent Tauri command.
    pub fn compute_profile_dto(&self) -> crate::dto::ComputeProfileDto {
        self.compute_profile().into()
    }

    /// Return the resolved Candle device label used to construct the
    /// runtime. After bootstrap fallback this is always `"cpu"` or
    /// `"metal"`. Useful for tests and host adapters that need to
    /// know which [`reimagine_inference::BackendInstance`] the
    /// workspace is actually running.
    pub fn resolved_candle_device_label(&self) -> String {
        resolved_candle_device_label(&self.resolved_backend_instance)
    }

    /// Return the resolved [`reimagine_inference::BackendInstance`]
    /// the workspace bootstrap selected.
    pub fn resolved_backend_instance(&self) -> &reimagine_inference::BackendInstance {
        &self.resolved_backend_instance
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
    use reimagine_inference::BackendInstance;
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

        let composed = compose_inference_runtime(
            &config,
            BackendInstance::new("candle:cpu"),
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
        assert_eq!(workspace.backend_config().candle_device, "cpu");
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

    // ── Compute profile tests (Task 3) ─────────────────────────────

    fn assert_cpu_available(profile: &WorkspaceComputeProfile) {
        let cpu = profile
            .backend_profiles
            .iter()
            .flat_map(|bp| bp.instances.iter())
            .find(|inst| inst.instance == BackendInstance::new("candle:cpu"))
            .expect("candle:cpu instance present in profile");
        assert_eq!(
            cpu.status,
            reimagine_inference::BackendInstanceStatus::Available,
            "candle:cpu should always be Available"
        );
    }

    fn assert_metal_present(profile: &WorkspaceComputeProfile) {
        let metal = profile
            .backend_profiles
            .iter()
            .flat_map(|bp| bp.instances.iter())
            .find(|inst| inst.instance == BackendInstance::new("candle:metal"))
            .expect("candle:metal instance present in profile");
        assert_eq!(
            metal.status,
            reimagine_inference::BackendInstanceStatus::Available,
            "candle:metal should be Available on Apple hardware"
        );
    }

    fn metal_is_available_on_this_host() -> bool {
        reimagine_inference_candle::CandleDevice::new("metal")
            .try_build_device()
            .is_ok()
    }

    #[test]
    fn compute_profile_contains_available_cpu_instance() {
        let base = temp_dir("profile-cpu");
        let workspace = WorkspaceHost::with_defaults(WorkspaceScope::new("profile-cpu"), &base);
        let profile = workspace.compute_profile();
        assert_cpu_available(&profile);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn compute_profile_works_without_running_a_workflow() {
        let base = temp_dir("profile-no-run");
        let workspace = WorkspaceHost::with_defaults(WorkspaceScope::new("profile-no-run"), &base);
        // compute_profile() must work immediately after construction,
        // without any workflow run or runtime boot.
        let profile = workspace.compute_profile();
        assert!(!profile.backend_profiles.is_empty());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn invalid_candle_device_falls_back_to_cpu_with_diagnostic() {
        let base = temp_dir("profile-tpu");
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "tpu".to_string(),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::with_backend_config(
            WorkspaceScope::new("profile-tpu"),
            &base,
            backend_config,
            Arc::new(VecRunEventSink::new()),
        );

        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Candle,
            "configured backend stays Candle"
        );
        assert_eq!(
            workspace.resolved_candle_device_label(),
            "cpu",
            "workspace must fall back to CPU when device label is invalid"
        );

        let profile = workspace.compute_profile();
        assert_cpu_available(&profile);

        let diagnostic = profile
            .diagnostics
            .iter()
            .find(|d| d.message().contains("tpu"))
            .unwrap_or_else(|| {
                panic!(
                    "expected a fallback diagnostic mentioning `tpu`, got: {:?}",
                    profile.diagnostics
                )
            });
        assert_eq!(
            diagnostic.code().as_str(),
            "INFERENCE_PROFILE/INVALID_DEVICE",
            "fallback diagnostic should use the invalid-device code"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn unknown_open_selected_instance_remains_pinned_with_diagnostic() {
        let base = temp_dir("profile-unknown-selected");
        let backend_config = InferenceBackendConfig {
            selected_instance: Some("ghost:cpu".to_string()),
            candle_device: "metal".to_string(),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::with_backend_config(
            WorkspaceScope::new("profile-unknown-selected"),
            &base,
            backend_config,
            Arc::new(VecRunEventSink::new()),
        );

        assert_eq!(
            workspace.resolved_backend_instance(),
            &BackendInstance::new("ghost:cpu"),
            "unknown explicit selection must fail closed"
        );
        let profile = workspace.compute_profile();
        assert_cpu_available(&profile);
        let diagnostic = profile
            .diagnostics
            .iter()
            .find(|d| d.code().as_str() == "APP_HOST/BACKEND_SELECTED_INSTANCE_UNKNOWN")
            .unwrap_or_else(|| {
                panic!(
                    "expected unknown selected-instance diagnostic, got: {:?}",
                    profile.diagnostics
                )
            });
        assert!(diagnostic.message().contains("ghost:cpu"));
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn mps_label_picks_metal_when_available_cpu_otherwise() {
        let base = temp_dir("profile-mps");
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "mps".to_string(),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::with_backend_config(
            WorkspaceScope::new("profile-mps"),
            &base,
            backend_config,
            Arc::new(VecRunEventSink::new()),
        );

        let profile = workspace.compute_profile();
        let resolved = workspace.resolved_candle_device_label();
        assert_cpu_available(&profile);

        if metal_is_available_on_this_host() {
            assert_metal_present(&profile);
            assert_eq!(
                resolved, "metal",
                "mps normalizes to metal when Metal is available"
            );
            assert!(
                profile.diagnostics.is_empty(),
                "no fallback diagnostic when Metal is available, got: {:?}",
                profile.diagnostics
            );
        } else {
            assert_eq!(
                resolved, "cpu",
                "mps falls back to cpu when Metal is unavailable"
            );
            let diagnostic = profile
                .diagnostics
                .iter()
                .find(|d| d.code().as_str() == "INFERENCE_PROFILE/DEVICE_UNAVAILABLE")
                .unwrap_or_else(|| {
                    panic!(
                        "expected a DEVICE_UNAVAILABLE fallback diagnostic, got: {:?}",
                        profile.diagnostics
                    )
                });
            assert!(
                diagnostic.message().contains("mps") || diagnostic.message().contains("metal"),
                "diagnostic should mention mps or metal, got: {}",
                diagnostic.message()
            );
        }
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn compute_profile_works_after_async_try_with_defaults() {
        let base = temp_dir("profile-try-defaults");
        let workspace =
            WorkspaceHost::try_with_defaults(WorkspaceScope::new("profile-try-defaults"), &base)
                .await
                .expect("try_with_defaults should succeed with no config");

        // The accessor must work after the async bootstrap path
        // without any workflow run.
        let profile = workspace.compute_profile();
        assert_cpu_available(&profile);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn fallback_cpu_keeps_same_registry_wiring() {
        // The fallback path must register `candle:cpu` with the same
        // descriptor shape the cpu path uses — same plugin / extension
        // / device / runtime hooks.
        let base = temp_dir("profile-fallback-wiring");
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "tpu".to_string(),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::with_backend_config(
            WorkspaceScope::new("profile-fallback-wiring"),
            &base,
            backend_config,
            Arc::new(VecRunEventSink::new()),
        );

        let registry = workspace.runtime_service.registry();
        let cpu_id = NodeTypeId::new("builtin.checkpoint_loader");
        assert!(
            registry.get(&cpu_id).is_some(),
            "fallback to cpu must still register the built-in checkpoint loader executor"
        );
        let _ = fs::remove_dir_all(&base);
    }
}
