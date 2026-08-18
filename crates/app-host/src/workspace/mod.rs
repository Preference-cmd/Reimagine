pub mod builder;

use std::path::Path;
use std::sync::Arc;

pub use builder::WorkspaceHostBuilder;

use reimagine_agent_harness::{AgentEventSink, AgentToolRegistry, WorkspaceScope};
use reimagine_backend_worker_host::WorkerStorePaths;
use reimagine_config::{AppConfig, AppPaths, ConfigDocument, InferenceBackendConfig};
use reimagine_core::model::NodeDef;
use reimagine_inference::WorkspaceComputeProfile;
use reimagine_nodes::BuiltinNodeCatalog;
use reimagine_runtime::{BoxedRunEventSink, RuntimeService, VecRunEventSink};

#[cfg(test)]
use crate::inference::compose::compose_inference_runtime;
use crate::inference::compose::{
    BootstrapInference, ComposedInferenceRuntime, bootstrap_inference,
    bootstrap_inference_with_worker_inventory,
};
use crate::inference::selection::resolved_candle_device_label;
use crate::model_acquisition_service::ModelAcquisitionService;
use crate::node_catalog::{NodeCatalogAlignment, NodeCatalogService};
use crate::provider_config::AgentProviderConfigDocument;
use crate::services::WorkspaceServices;
use crate::tools::register_app_tools;
use crate::{
    AgentService, AppHostError, BackendSelection, BoardService, ModelService, ProjectService,
    WorkflowService,
};
use crate::{InstalledWorkerInventoryProvider, WorkerInventoryProvider};

/// How long a re-bootstrap waits for in-flight runs to drain before giving
/// up and keeping the current backend in place.
const REBOOTSTRAP_DRAIN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Establish the AR-08 project workflow layout (idempotent).
///
/// Called by the async production bootstrap BEFORE any discovery
/// poll loop is spawned, so this await can never sit between the
/// discovery spawn and host construction (AR-37 constraint). It
/// delegates file migration to `reimagine_config::migrate_to_project_layout`
/// (moves legacy `workflows/*.json` into `projects/default/workflows/`,
/// keeping `.backup` originals) and then materialises the default
/// project's `project.json` + `board.json` documents so the project
/// is discoverable by `ProjectService::list_projects` from the first
/// bootstrap. The documents are written directly because the default
/// project directory tree already exists after the config-level
/// migration, so `create_project`'s ProjectAlreadyExists guard would
/// reject a re-run on a partially-migrated workspace.
async fn migrate_project_layout(config: &AppConfig) -> Result<(), AppHostError> {
    use crate::board_service::ensure_board_file;
    use reimagine_core::event::Timestamp;
    use reimagine_core::model::ProjectId;
    use reimagine_core::project::{Project, ProjectMetadata};

    let paths = config.paths();

    // 1. Move legacy top-level workflow documents into the default
    //    project's layout (idempotent; originals kept as .backup).
    reimagine_config::migrate_to_project_layout(paths)
        .await
        .map_err(AppHostError::BootstrapConfig)?;

    // 2. Materialise a discoverable default project document set.
    let default_project_id = ProjectId::new("default");
    let project_file = paths
        .project_dir(default_project_id.as_str())
        .join("project.json");
    if !project_file.is_file() {
        let metadata = ProjectMetadata::new(
            "Default",
            "Default workspace project created by the AR-08 project-layout migration.",
            Timestamp::new("2026-08-18T00:00:00Z"),
            Timestamp::new("2026-08-18T00:00:00Z"),
        );
        let project = Project::new(default_project_id.clone(), metadata);
        crate::project_service::write_project_atomic(&project_file, &project).await?;
    }
    ensure_board_file(paths, &default_project_id).await?;
    Ok(())
}

pub struct WorkspaceHost {
    pub(crate) workspace_scope: WorkspaceScope,
    pub(crate) config: Arc<AppConfig>,
    pub(crate) backend_config: InferenceBackendConfig,
    pub(crate) project_service: Arc<ProjectService>,
    pub(crate) board_service: Arc<BoardService>,
    pub(crate) workflow_service: Arc<WorkflowService>,
    pub(crate) model_service: Arc<ModelService>,
    pub(crate) runtime_service: Arc<RuntimeService>,
    pub(crate) agent_service: Arc<AgentService>,
    pub(crate) node_catalog: Arc<NodeCatalogService>,
    pub(crate) builtin_catalog: Arc<BuiltinNodeCatalog>,
    pub(crate) services: Arc<WorkspaceServices>,
    pub(crate) compute_profile: Arc<WorkspaceComputeProfile>,
    pub(crate) resolved_backend_instance: reimagine_inference::BackendInstance,
    pub(crate) worker_switch: Option<Arc<crate::WorkerSwitchService>>,
    pub(crate) worker_inventory: Arc<dyn WorkerInventoryProvider>,
    pub(crate) topology:
        Option<Arc<tokio::sync::Mutex<crate::inference::topology::ConnectionTopologyManager>>>,
    pub(crate) discovery: Option<Arc<crate::inference::discovery::DiscoveryOrchestrator>>,
    pub(crate) event_sink: BoxedRunEventSink,
    pub(crate) agent_event_sink: Arc<dyn AgentEventSink>,
}

impl std::fmt::Debug for WorkspaceHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceHost")
            .field("workspace_scope", &self.workspace_scope)
            .field("config", &self.config)
            .field("backend_config", &self.backend_config)
            .field("runtime_service", &self.runtime_service)
            .field("node_catalog", &self.node_catalog)
            .field("compute_profile", &self.compute_profile)
            .field("resolved_backend_instance", &self.resolved_backend_instance)
            .field("worker_switch", &self.worker_switch)
            .finish_non_exhaustive()
    }
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
        let board_service = Arc::new(BoardService::new(config.paths().clone()));
        let project_service = Arc::new(ProjectService::new(
            config.paths().clone(),
            Arc::clone(&board_service),
        ));
        let workflow_service = Arc::new(WorkflowService::new(config.paths().clone()));
        let acquisition_service = Arc::new(ModelAcquisitionService::new(
            config.paths().clone(),
            &config,
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            acquisition_service.clone(),
        ));
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
            acquisition_service,
            Arc::clone(&runtime_service),
            Arc::clone(&node_catalog),
        ));
        let mut registry = AgentToolRegistry::new();
        register_app_tools(&mut registry, Arc::clone(&services));
        let registry = Arc::new(registry);
        let agent_service = Arc::new(AgentService::with_registry_and_session_dir(
            workspace_scope.clone(),
            Arc::clone(&registry),
            config.paths().base_path().join("agent-sessions"),
        ));
        Self {
            workspace_scope,
            config,
            backend_config,
            project_service,
            board_service,
            workflow_service,
            model_service,
            runtime_service,
            agent_service,
            node_catalog,
            builtin_catalog,
            services,
            compute_profile,
            resolved_backend_instance,
            worker_switch: None,
            worker_inventory: Arc::new(crate::EmptyWorkerInventoryProvider),
            topology: None,
            discovery: None,
            event_sink: Arc::new(VecRunEventSink::new()),
            agent_event_sink: Arc::new(reimagine_agent_harness::VecAgentEventSink::new())
                as Arc<dyn AgentEventSink>,
        }
    }

    pub fn with_defaults(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self::with_defaults_and_backend(
            workspace_scope,
            base_path,
            BackendSelection::default(),
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
            Arc::new(InstalledWorkerInventoryProvider::for_base_path(&base_path)),
        )
        .await
    }

    /// Bootstrap with the desktop host's `{workspace}.app-data` worker store
    /// convention, loading the backend config from disk.
    ///
    /// The worker inventory is derived from `app_data_root` (the same root
    /// [`crate::WorkerManagementService::offline`] uses), so a production
    /// desktop host that passes its real app-data directory sees installed
    /// workers.
    pub async fn try_with_app_data_root_and_event_sinks(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        app_data_root: impl Into<std::path::PathBuf>,
        event_sink: BoxedRunEventSink,
        agent_event_sink: Arc<dyn AgentEventSink>,
    ) -> Result<Self, AppHostError> {
        let base_path = base_path.into();
        let config = AppConfig::new(AppPaths::new(&base_path));
        let backend_config = load_backend_config_result(&config).await?;
        Self::try_with_app_data_root_and_backend_config(
            workspace_scope,
            &base_path,
            app_data_root,
            backend_config,
            event_sink,
            agent_event_sink,
        )
        .await
    }

    /// Bootstrap with an explicit app-data root and an explicit backend
    /// configuration (used by the re-bootstrap path).
    pub async fn try_with_app_data_root_and_backend_config(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        app_data_root: impl Into<std::path::PathBuf>,
        backend_config: InferenceBackendConfig,
        event_sink: BoxedRunEventSink,
        agent_event_sink: Arc<dyn AgentEventSink>,
    ) -> Result<Self, AppHostError> {
        let config = AppConfig::new(AppPaths::new(base_path));
        Self::with_backend_config_and_worker_inventory_inner(
            workspace_scope,
            config,
            backend_config,
            event_sink,
            agent_event_sink,
            Arc::new(InstalledWorkerInventoryProvider::new(
                WorkerStorePaths::new(app_data_root),
            )),
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
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
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
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
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
            BackendSelection::default(),
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
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()) as Arc<dyn AgentEventSink>,
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
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()) as Arc<dyn AgentEventSink>,
        )
    }

    fn with_backend_config_inner(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        event_sink: BoxedRunEventSink,
        agent_event_sink: Arc<dyn AgentEventSink>,
    ) -> Self {
        let acquisition_service = Arc::new(ModelAcquisitionService::new(
            config.paths().clone(),
            &config,
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            acquisition_service.clone(),
        ));

        let bootstrapped =
            match bootstrap_inference(&config, &backend_config, model_service.clone()) {
                Ok(bootstrapped) => bootstrapped,
                Err(error) => {
                    tracing::error!(
                        %error,
                        "inference bootstrap failed on the synchronous construction path; \
                         continuing with an empty executor registry"
                    );
                    BootstrapInference {
                        runtime: ComposedInferenceRuntime::degraded(),
                        compute_profile: WorkspaceComputeProfile::new(),
                        topology: None,
                        discovery: None,
                    }
                }
            };
        let worker_switch = bootstrapped.runtime.worker_switch.clone();
        let runtime_service = Arc::new(
            RuntimeService::new(
                bootstrapped.runtime.executor_registry,
                bootstrapped.runtime.runtime_hooks.clone(),
                event_sink,
                Arc::new(reimagine_runtime::SystemClock),
            )
            .with_resource_hint_sink(
                worker_switch
                    .as_ref()
                    .and_then(|worker_switch| worker_switch.active_hint_sink()),
            ),
        );
        if let Some(worker_switch) = &worker_switch {
            let cancellation: Arc<dyn crate::RunCancellation> = runtime_service.clone();
            worker_switch.set_run_cancellation(cancellation);
        }
        let builtin_catalog = Arc::new(BuiltinNodeCatalog::v1());
        let agent_session_dir = config.paths().base_path().join("agent-sessions");
        let mut host = Self::new(
            workspace_scope,
            config,
            backend_config,
            runtime_service,
            builtin_catalog,
            Arc::new(bootstrapped.compute_profile),
            bootstrapped.runtime.selected_instance,
        );
        host.worker_switch = worker_switch;
        // Replace the default AgentService with one that uses the injected event sink
        let registry = host.agent_service.registry().clone();
        let providers = host.agent_service.providers().clone();
        host.agent_service = Arc::new(AgentService::with_registry_providers_sink_and_session_dir(
            host.workspace_scope.clone(),
            registry,
            providers,
            agent_event_sink,
            agent_session_dir,
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
        let acquisition_service = Arc::new(ModelAcquisitionService::new(
            config.paths().clone(),
            &config,
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            acquisition_service.clone(),
        ));

        // AR-37: load the app-host-owned provider config document before
        // bootstrapping inference, so no additional await sits between the
        // discovery poll loop spawn and the host construction (the topology
        // integration tests assert the first reconcile pass registers the
        // config endpoints). A missing file yields an empty document (no
        // providers registered).
        let provider_document = {
            let handle = config.config::<AgentProviderConfigDocument>()?;
            let (document, _report) = handle.load().await?;
            document
        };
        let provider_base_path = config.paths().base_path().to_path_buf();

        // AR-08: establish the project workflow layout (default project +
        // legacy migration) BEFORE bootstrapping inference, so no await
        // sits between the discovery poll loop spawn and host
        // construction (AR-37 constraint).
        migrate_project_layout(&config).await?;

        let bootstrapped = bootstrap_inference_with_worker_inventory(
            &config,
            &backend_config,
            model_service,
            Arc::clone(&worker_inventory),
        )
        .await
        .map_err(|error| AppHostError::InferenceBootstrap {
            message: error.to_string(),
        })?;
        let worker_switch = bootstrapped.runtime.worker_switch.clone();
        let runtime_service = Arc::new(
            RuntimeService::new(
                bootstrapped.runtime.executor_registry,
                bootstrapped.runtime.runtime_hooks.clone(),
                Arc::clone(&event_sink),
                Arc::new(reimagine_runtime::SystemClock),
            )
            .with_resource_hint_sink(
                worker_switch
                    .as_ref()
                    .and_then(|worker_switch| worker_switch.active_hint_sink()),
            ),
        );
        if let Some(worker_switch) = &worker_switch {
            let cancellation: Arc<dyn crate::RunCancellation> = runtime_service.clone();
            worker_switch.set_run_cancellation(cancellation);
        }
        let builtin_catalog = Arc::new(BuiltinNodeCatalog::v1());
        let agent_session_dir = config.paths().base_path().join("agent-sessions");
        let mut host = Self::new(
            workspace_scope,
            config,
            backend_config,
            runtime_service,
            builtin_catalog,
            Arc::new(bootstrapped.compute_profile),
            bootstrapped.runtime.selected_instance,
        );
        host.worker_switch = worker_switch;
        host.worker_inventory = Arc::clone(&worker_inventory);
        host.topology = bootstrapped.topology;
        host.discovery = bootstrapped.discovery;
        host.event_sink = Arc::clone(&event_sink);
        host.agent_event_sink = agent_event_sink;
        let registry = host.agent_service.registry().clone();
        let providers = host.agent_service.providers().clone();

        // AR-37: register concrete providers from the loaded document so
        // the production desktop bootstrap path (not just
        // WorkspaceHostBuilder) sees configured providers. App-host is
        // the owner of provider file/secrets loading; Tauri never
        // touches API keys directly.
        let (registered, errors) = crate::register_providers_from_document(
            &providers,
            &provider_document,
            Some(&provider_base_path),
        );
        if !registered.is_empty() {
            tracing::info!(
                providers = ?registered.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
                "registered agent providers (production bootstrap)"
            );
        }
        for error in &errors {
            tracing::warn!(%error, "skipping provider");
        }

        host.agent_service = Arc::new(AgentService::with_registry_providers_sink_and_session_dir(
            host.workspace_scope.clone(),
            registry,
            providers,
            host.agent_event_sink.clone(),
            agent_session_dir,
        ));
        Ok(host)
    }

    /// Establish the AR-08 project workflow layout (idempotent).
    ///
    /// Called by production bootstrap BEFORE any discovery poll loop is
    /// spawned, so the file migration never adds an await between the
    /// discovery spawn and host construction (AR-37 constraint). Moves
    /// legacy top-level `workflows/*.json` into
    /// `projects/default/workflows/` (originals kept as .backup) and
    /// materialises the default project's `project.json` +
    /// `board.json` documents.
    pub async fn migrate_to_project_layout(&self) -> Result<(), AppHostError> {
        migrate_project_layout(&self.config).await
    }

    pub fn with_agent_event_sink(self, event_sink: Arc<dyn AgentEventSink>) -> Self {
        let registry = self.agent_service.registry().clone();
        let providers = self.agent_service.providers().clone();
        let session_dir = self.agent_service.session_dir().to_path_buf();
        let agent_service = Arc::new(AgentService::with_registry_providers_sink_and_session_dir(
            self.workspace_scope.clone(),
            registry,
            providers,
            Arc::clone(&event_sink),
            session_dir,
        ));
        Self {
            agent_service,
            agent_event_sink: event_sink,
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
    pub fn project_service(&self) -> &Arc<ProjectService> {
        &self.project_service
    }
    pub fn board_service(&self) -> &Arc<BoardService> {
        &self.board_service
    }
    pub fn model_service(&self) -> &Arc<ModelService> {
        &self.model_service
    }
    pub fn runtime_service(&self) -> &Arc<RuntimeService> {
        &self.runtime_service
    }
    pub async fn selected_worker(
        &self,
    ) -> Result<crate::WorkerSelectionHandle, crate::WorkerSwitchError> {
        let workers = self
            .worker_switch
            .as_ref()
            .ok_or(crate::WorkerSwitchError::NoActiveWorker)?;
        Ok(workers.selected().await)
    }

    pub async fn drain_and_switch_worker(
        &self,
        target: crate::WorkerBackendCandidate,
        deadline: std::time::Duration,
    ) -> Result<crate::WorkerSelectionHandle, crate::WorkerSwitchError> {
        let workers = self
            .worker_switch
            .as_ref()
            .ok_or(crate::WorkerSwitchError::NoActiveWorker)?;
        workers.drain_and_switch(Arc::new(target), deadline).await
    }

    pub async fn cancel_and_switch_worker(
        &self,
        target: crate::WorkerBackendCandidate,
        deadline: std::time::Duration,
    ) -> Result<crate::WorkerSelectionHandle, crate::WorkerSwitchError> {
        let workers = self
            .worker_switch
            .as_ref()
            .ok_or(crate::WorkerSwitchError::NoActiveWorker)?;
        workers.cancel_and_switch(Arc::new(target), deadline).await
    }

    /// Wait for in-flight runs on the active worker to complete (up to
    /// `deadline`) and then shut the worker down gracefully.
    ///
    /// This is the drain half of the re-bootstrap path: it guarantees no
    /// run is in flight when the old worker process is retired, without
    /// cancelling anything. Hosts without a process worker (built-in
    /// fallback backends) return immediately.
    pub async fn drain_active_worker(
        &self,
        deadline: std::time::Duration,
    ) -> Result<(), crate::WorkerSwitchError> {
        let Some(worker_switch) = &self.worker_switch else {
            return Ok(());
        };
        let handle = worker_switch.selected().await;
        let active = worker_switch.resolve(&handle).await?;
        let leases = active.run_leases();
        leases.begin_draining();
        if !leases.wait_until_empty(deadline).await {
            leases.restore_ready();
            return Err(crate::WorkerSwitchError::DrainTimeout {
                instance: handle.instance().clone(),
            });
        }
        worker_switch.shutdown_active(deadline).await
    }

    /// Re-run the bootstrap flow against `current` with `new_selection`.
    ///
    /// The new backend is composed (and its worker, if any, started) first;
    /// only then are in-flight runs on the current backend drained. If the
    /// drain does not finish within [`REBOOTSTRAP_DRAIN_DEADLINE`], the
    /// freshly built workspace is shut down and the current one is left
    /// untouched.
    pub async fn rebuild_workspace(
        current: &Self,
        new_selection: BackendSelection,
    ) -> Result<Self, AppHostError> {
        let backend_config = backend_config_for_selection(&current.backend_config, new_selection);
        let rebuilt = Self::with_backend_config_and_worker_inventory_inner(
            current.workspace_scope.clone(),
            (*current.config).clone(),
            backend_config,
            current.event_sink.clone(),
            current.agent_event_sink.clone(),
            Arc::clone(&current.worker_inventory),
        )
        .await?;
        if let Err(error) = current
            .drain_active_worker(REBOOTSTRAP_DRAIN_DEADLINE)
            .await
        {
            rebuilt.shutdown().await;
            return Err(AppHostError::RebootFailed {
                message: error.to_string(),
            });
        }
        Ok(rebuilt)
    }

    /// Drain in-flight runs and re-bootstrap the workspace with a new
    /// backend selection (B4-8).
    ///
    /// This preserves the workspace paths and event sinks. On success the
    /// new selection is persisted to `inference_backend.json`, so a
    /// restarted app boots with the re-bootstrapped backend. If the
    /// in-memory swap succeeds but the config write fails, the error
    /// reports the persist failure while the workspace is already swapped.
    pub async fn rebootstrap(
        &mut self,
        new_selection: BackendSelection,
    ) -> Result<(), AppHostError> {
        let rebuilt = Self::rebuild_workspace(self, new_selection).await?;
        let persisted = rebuilt.backend_config.clone();
        *self = rebuilt;
        self.persist_backend_config(&persisted).await?;
        Ok(())
    }

    /// Persist a backend config to `inference_backend.json` through the
    /// workspace's config handle.
    async fn persist_backend_config(
        &self,
        backend_config: &InferenceBackendConfig,
    ) -> Result<(), AppHostError> {
        let handle = self.config.config::<InferenceBackendConfig>()?;
        handle
            .save(backend_config)
            .await
            .map(|_| ())
            .map_err(|error| AppHostError::RebootFailed {
                message: format!("failed to persist backend selection: {error}"),
            })
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

    /// The connection topology manager (T13), when remote workers are
    /// configured. `None` in single-worker mode.
    pub fn topology(
        &self,
    ) -> Option<&Arc<tokio::sync::Mutex<crate::inference::topology::ConnectionTopologyManager>>>
    {
        self.topology.as_ref()
    }

    /// The discovery orchestrator (T12/T13), when a topology manager
    /// exists.
    pub fn discovery(&self) -> Option<&Arc<crate::inference::discovery::DiscoveryOrchestrator>> {
        self.discovery.as_ref()
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
        let mut dto: crate::dto::ComputeProfileDto = self.compute_profile().clone().into();
        if let Some(topology) = &self.topology {
            // try_lock: the discovery poll loop may hold the lock; a
            // busy lock just means this snapshot omits topology workers.
            if let Ok(guard) = topology.try_lock() {
                dto =
                    dto.with_topology_workers(crate::dto::topology_workers_from_pool(guard.pool()));
            }
        }
        dto
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

    /// Shut down the active inference worker, if any.
    ///
    /// This is the application-level shutdown hook. Call it when the host
    /// is exiting so that child worker processes are cleaned up instead of
    /// becoming orphans.
    pub async fn shutdown(&self) {
        let Some(worker_switch) = &self.worker_switch else {
            return;
        };
        let deadline = std::time::Duration::from_secs(5);
        if let Err(error) = worker_switch.shutdown_active(deadline).await {
            eprintln!("[app-host] worker shutdown error: {error}");
        }
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

/// Project a [`BackendSelection`] onto the persisted backend config,
/// keeping device preferences but clearing the pinned instance so the
/// bootstrap policy re-resolves against the live profile.
fn backend_config_for_selection(
    current: &InferenceBackendConfig,
    selection: BackendSelection,
) -> InferenceBackendConfig {
    #[allow(deprecated)]
    let backend = match selection {
        BackendSelection::Burn => reimagine_config::InferenceBackendKind::Burn,
        BackendSelection::Candle => reimagine_config::InferenceBackendKind::Candle,
    };
    InferenceBackendConfig {
        backend,
        selected_instance: None,
        ..current.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_backend_worker_host::{
        ExpectedWorkerIdentity, InstallationRecord, InventoryStore, WorkerInstallationId,
    };
    use reimagine_backend_worker_protocol::BackendInstanceId;
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

    /// Seed a durable burn-worker installation record beneath `app_data_root`.
    fn seed_worker_record(app_data_root: &std::path::Path, install_dir: &std::path::Path) {
        let store = InventoryStore::new(WorkerStorePaths::new(app_data_root));
        let exe = install_dir.join("burn-worker");
        fs::write(&exe, "#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&exe, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let record = InstallationRecord {
            installation_id: WorkerInstallationId("install-burn-1".to_owned()),
            version: "1.0.0".to_owned(),
            identity: ExpectedWorkerIdentity {
                backend_instance_id: BackendInstanceId("burn:wgpu:default".to_owned()),
                installation_id: WorkerInstallationId("install-burn-1".to_owned()),
                backend_kind: "burn".to_owned(),
                target: "test".to_owned(),
                manifest_digest: "seed".to_owned(),
            },
            installed_at: chrono::Utc::now(),
            install_path: install_dir.display().to_string(),
            manifest_profile: None,
        };
        store.add(&record).expect("seed inventory record");
    }

    #[test]
    fn workspace_with_defaults_uses_burn() {
        let base = temp_dir("defaults");
        let workspace = WorkspaceHost::with_defaults(WorkspaceScope::new("test-defaults"), &base);
        assert_eq!(workspace.base_path(), base);
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Burn
        );
        assert_eq!(workspace.backend_config().burn_device, "cpu");
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
    fn missing_config_file_defaults_to_burn() {
        let base = temp_dir("no-config");
        let workspace = WorkspaceHost::with_defaults(WorkspaceScope::new("test-no-config"), &base);
        assert_eq!(workspace.base_path(), base);
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Burn
        );
        assert_eq!(workspace.backend_config().burn_device, "cpu");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn app_host_inference_composition_registers_builtin_executors() {
        let base = temp_dir("compose-runtime");
        let config = AppConfig::new(reimagine_config::AppPaths::new(&base));
        let acquisition_service = Arc::new(ModelAcquisitionService::new(
            config.paths().clone(),
            &config,
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            acquisition_service,
        ));

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
            InferenceBackendKind::Burn
        );
        assert_eq!(workspace.backend_config().burn_device, "cpu");
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

    // ── App-data root wiring (BE-38) ──────────────────────────────

    fn burn_worker_visible(profile: &reimagine_inference::WorkspaceComputeProfile) -> bool {
        profile
            .backend_profiles
            .iter()
            .flat_map(|backend| backend.instances.iter())
            .any(|instance| instance.instance == BackendInstance::new("burn:wgpu:default"))
    }

    #[tokio::test]
    async fn try_with_app_data_root_reads_worker_store_beneath_explicit_root() {
        let base = temp_dir("app-data-root");
        let app_data_root = base.join("app-data");
        let install_dir = base.join("installed-worker");
        fs::create_dir_all(&install_dir).unwrap();
        seed_worker_record(&app_data_root, &install_dir);

        // Selection pinned to the built-in candle instance so the seeded
        // burn worker is never activated (no process spawn in tests).
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            selected_instance: Some("candle:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("app-data-root"),
            &base,
            &app_data_root,
            backend_config,
            Arc::new(VecRunEventSink::new()),
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
        )
        .await
        .expect("bootstrap with explicit app-data root");

        assert_eq!(
            workspace.resolved_backend_instance(),
            &BackendInstance::new("candle:cpu")
        );
        assert!(
            burn_worker_visible(&workspace.compute_profile()),
            "profile must include the worker installed beneath the explicit app-data root"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn base_path_derived_bootstrap_does_not_see_explicit_app_data_store() {
        let base = temp_dir("app-data-miss");
        let app_data_root = base.join("app-data");
        let install_dir = base.join("installed-worker");
        fs::create_dir_all(&install_dir).unwrap();
        seed_worker_record(&app_data_root, &install_dir);

        // The legacy constructor derives the store from `{base}.app-data`,
        // which is NOT the same directory as the explicit root — the seeded
        // record must stay invisible.
        let workspace =
            WorkspaceHost::try_with_defaults(WorkspaceScope::new("app-data-miss"), &base)
                .await
                .expect("bootstrap with base-path-derived store");

        assert!(
            !burn_worker_visible(&workspace.compute_profile()),
            "base-path-derived provider must not see a store beneath the explicit root"
        );
        let _ = fs::remove_dir_all(&base);
    }

    // ── Re-bootstrap path (B4-8) ──────────────────────────────────

    #[tokio::test]
    async fn rebootstrap_swaps_backend_and_preserves_workspace() {
        let base = temp_dir("rebootstrap");
        let mut workspace =
            WorkspaceHost::try_with_defaults(WorkspaceScope::new("rebootstrap"), &base)
                .await
                .expect("initial bootstrap");

        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Burn
        );
        workspace
            .rebootstrap(BackendSelection::Burn)
            .await
            .expect("rebootstrap to the same backend must succeed (restart compute backend)");
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Burn
        );
        assert_eq!(workspace.base_path(), base);

        #[allow(deprecated)]
        workspace
            .rebootstrap(BackendSelection::Candle)
            .await
            .expect("rebootstrap to candle");
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Candle
        );
        assert_eq!(workspace.base_path(), base);
        assert_eq!(workspace.resolved_candle_device_label(), "cpu");

        // The selection must be persisted so a restarted app boots with it.
        let config_path = base.join("config").join(InferenceBackendConfig::KEY);
        let persisted: InferenceBackendConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path).expect("persisted backend config file"),
        )
        .expect("persisted config parses");
        assert_eq!(
            persisted.backend,
            InferenceBackendKind::Candle,
            "rebootstrap must persist the new selection to inference_backend.json"
        );

        let profile = workspace.compute_profile();
        assert_cpu_available(&profile);
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn rebuild_workspace_builds_new_host_without_mutating_current() {
        let base = temp_dir("rebuild");
        let original = WorkspaceHost::try_with_defaults(WorkspaceScope::new("rebuild"), &base)
            .await
            .expect("initial bootstrap");

        let rebuilt = WorkspaceHost::rebuild_workspace(&original, BackendSelection::Burn)
            .await
            .expect("rebuild");

        assert_eq!(rebuilt.base_path(), base);
        assert_eq!(rebuilt.backend_config().backend, InferenceBackendKind::Burn);
        assert_eq!(
            original.backend_config().backend,
            InferenceBackendKind::Burn,
            "rebuild must not mutate the current workspace"
        );
        assert!(
            rebuilt
                .runtime_service
                .registry()
                .get(&NodeTypeId::new("builtin.ksampler"))
                .is_some()
        );
        let _ = fs::remove_dir_all(&base);
    }

    // ---- AR-37: production provider bootstrap ----

    fn seed_valid_provider_config(base: &std::path::Path) {
        use crate::provider_config::{AgentProviderConfigDocument, ProviderConfig};
        use reimagine_agent_provider::OpenAiChatCompletionsConfig;
        let openai = OpenAiChatCompletionsConfig::new(
            "https://api.example.com/v1",
            "sk-test",
            "gpt-4o-mini",
        );
        let provider = ProviderConfig::with_openai_chat_completions("openai", openai);
        let doc = AgentProviderConfigDocument::new(vec![provider]);
        let config_dir = base.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        let json = serde_json::to_string_pretty(&doc).unwrap();
        fs::write(config_dir.join("agent-providers.json"), json).unwrap();
    }

    fn seed_provider_with_missing_inner_config(base: &std::path::Path) {
        use crate::provider_config::{AgentProviderConfigDocument, ProviderConfig};
        use reimagine_agent_provider::OpenAiChatCompletionsConfig;
        let openai = OpenAiChatCompletionsConfig::new(
            "https://api.example.com/v1",
            "sk-test",
            "gpt-4o-mini",
        );
        let broken = ProviderConfig::with_openai_chat_completions("broken", openai.clone());
        let good = ProviderConfig::with_openai_chat_completions("openai", openai);
        let doc = AgentProviderConfigDocument::new(vec![broken, good]);
        let config_dir = base.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        // Null out the first provider inner typed config to simulate
        // a missing-inner entry; build_provider must skip it.
        let mut value = serde_json::to_value(&doc).unwrap();
        value["providers"][0]["openai_chat_completions"] = serde_json::Value::Null;
        let json = serde_json::to_string_pretty(&value).unwrap();
        fs::write(config_dir.join("agent-providers.json"), json).unwrap();
    }

    #[tokio::test]
    async fn production_bootstrap_registers_configured_providers() {
        let base = temp_dir("ar37-valid");
        let app_data_root = base.join("app-data");
        seed_valid_provider_config(&base);
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            selected_instance: Some("candle:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("ar37-valid"),
            &base,
            &app_data_root,
            backend_config,
            Arc::new(VecRunEventSink::new()),
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
        )
        .await
        .expect("production bootstrap with valid provider config");
        let providers = workspace.agent_service.providers();
        assert_eq!(
            providers.provider_names(),
            vec![reimagine_agent_harness::ProviderName::new("openai")],
            "configured provider must be registered by the production path"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn production_bootstrap_missing_config_registers_none_without_panic() {
        let base = temp_dir("ar37-missing");
        let app_data_root = base.join("app-data");
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            selected_instance: Some("candle:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("ar37-missing"),
            &base,
            &app_data_root,
            backend_config,
            Arc::new(VecRunEventSink::new()),
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
        )
        .await
        .expect("bootstrap succeeds with no provider config file");
        assert!(workspace.agent_service.providers().is_empty());
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn production_bootstrap_missing_inner_config_skips_that_provider_only() {
        let base = temp_dir("ar37-broken");
        let app_data_root = base.join("app-data");
        seed_provider_with_missing_inner_config(&base);
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            selected_instance: Some("candle:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("ar37-broken"),
            &base,
            &app_data_root,
            backend_config,
            Arc::new(VecRunEventSink::new()),
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
        )
        .await
        .expect("bootstrap with one broken provider entry");
        let providers = workspace.agent_service.providers();
        assert!(!providers.contains(&reimagine_agent_harness::ProviderName::new("broken")));
        assert!(providers.contains(&reimagine_agent_harness::ProviderName::new("openai")));
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn production_bootstrap_creates_discoverable_default_project() {
        let base = temp_dir("ar08-default-project");
        let app_data_root = base.join("app-data");
        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            selected_instance: Some("candle:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("ar08-default-project"),
            &base,
            &app_data_root,
            backend_config.clone(),
            Arc::new(VecRunEventSink::new()),
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
        )
        .await
        .expect("production bootstrap establishes the project layout");

        // The default project is discoverable with valid documents.
        let projects = workspace
            .project_service()
            .list_projects()
            .await
            .expect("list projects after bootstrap");
        assert!(
            projects.iter().any(|p| p.id().as_str() == "default"),
            "default project must be discoverable after production bootstrap"
        );
        assert!(base.join("projects/default/project.json").is_file());
        assert!(base.join("projects/default/board.json").is_file());

        // Idempotent: a second bootstrap still finds exactly one default.
        let workspace2 = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("ar08-default-project-2"),
            &base,
            &app_data_root,
            backend_config.clone(),
            Arc::new(VecRunEventSink::new()),
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
        )
        .await
        .expect("second production bootstrap stays idempotent");
        let projects2 = workspace2
            .project_service()
            .list_projects()
            .await
            .expect("list projects after second bootstrap");
        assert_eq!(
            projects2
                .iter()
                .filter(|p| p.id().as_str() == "default")
                .count(),
            1,
            "migration must be idempotent"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn migration_copies_legacy_workflows_into_default_project() {
        let base = temp_dir("ar08-legacy-migrate");
        let app_data_root = base.join("app-data");

        // Seed a legacy top-level workflow document.
        let legacy_dir = base.join("workflows");
        fs::create_dir_all(&legacy_dir).unwrap();
        let legacy = serde_json::json!({
            "schema_version": "reimagine.workflow.v1",
            "id": "legacy-wf",
            "version": 1,
            "metadata": { "name": "Legacy" },
            "interface": { "inputs": [], "outputs": [] },
            "nodes": [],
            "edges": [],
            "layout": { "nodes": {} }
        });
        fs::write(
            legacy_dir.join("legacy-wf.json"),
            serde_json::to_vec_pretty(&legacy).unwrap(),
        )
        .unwrap();

        let backend_config = InferenceBackendConfig {
            backend: InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            selected_instance: Some("candle:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("ar08-legacy-migrate"),
            &base,
            &app_data_root,
            backend_config,
            Arc::new(VecRunEventSink::new()),
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
        )
        .await
        .expect("bootstrap with legacy workflows");

        // The migrated copy lives under the default project and
        // the workflow service can load it by id.
        let migrated = base.join("projects/default/workflows/legacy-wf.json");
        assert!(
            migrated.is_file(),
            "legacy workflow migrated into default project"
        );
        // The workflow service discovers the migrated document via the
        // default project directory listing.
        let migrated_id = workspace
            .workflow_service()
            .load_workflow(&reimagine_core::model::WorkflowId::new("legacy-wf"))
            .await
            .expect("migrated workflow loads by id");
        assert_eq!(migrated_id.as_str(), "legacy-wf");
        let _ = fs::remove_dir_all(&base);
    }
}
