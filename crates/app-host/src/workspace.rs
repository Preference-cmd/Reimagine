pub(crate) mod aigc_parts;
pub mod builder;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

#[cfg(feature = "aigc")]
pub(crate) use aigc_parts::AigcParts;
pub use builder::WorkspaceHostBuilder;
use reimagine_core::health::Health;

use crate::WorkspaceScope;
#[cfg(feature = "aigc")]
use reimagine_backend_worker_host::WorkerStorePaths;
use reimagine_config::{AppConfig, AppPaths, ConfigDocument, InferenceBackendConfig};
use reimagine_core::event::EventBus;
use reimagine_core::model::{NodeDef, ProjectId};
#[cfg(feature = "aigc")]
use reimagine_inference::WorkspaceComputeProfile;
use reimagine_nodes::BuiltinNodeCatalog;
use reimagine_runtime::RuntimeService;

#[cfg(feature = "aigc")]
use crate::node_catalog::NodeCatalogAlignment;
use crate::node_catalog::NodeCatalogService;
use crate::services::WorkspaceServices;
use crate::{
    ActivityService, AppHostError, BoardService, CollectionService, DynamicWorkflowService,
    OperationService, ProjectService, WorkflowService,
};
#[cfg(feature = "aigc")]
use crate::{
    BackendSelection, InstalledWorkerInventoryProvider, ModelService, WorkerInventoryProvider,
};
#[cfg(feature = "aigc")]
use aigc::ModelAcquisitionService;
#[cfg(all(test, feature = "aigc"))]
use aigc::inference::compose::compose_inference_runtime;
#[cfg(feature = "aigc")]
use aigc::inference::compose::{BootstrapInference, ComposedInferenceRuntime, bootstrap_inference};
use mod_sdk::ModRegistry;

#[cfg(feature = "aigc")]
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
pub(crate) async fn migrate_project_layout(config: &AppConfig) -> Result<(), AppHostError> {
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
        crate::document_store::write_document(&project_file, &project, "project").await?;
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
    pub(crate) collection_service: Arc<CollectionService>,
    pub(crate) dynamic_workflow_service: Arc<DynamicWorkflowService>,
    pub(crate) workflow_service: Arc<WorkflowService>,
    pub(crate) workflow_services: RwLock<BTreeMap<ProjectId, Arc<WorkflowService>>>,
    pub(crate) event_bus: EventBus,
    pub(crate) runtime_service: Arc<RuntimeService>,
    pub(crate) node_catalog: Arc<NodeCatalogService>,
    pub(crate) builtin_catalog: Arc<BuiltinNodeCatalog>,
    pub(crate) services: Arc<WorkspaceServices>,
    pub(crate) operation_service: Arc<OperationService>,
    /// The registry the operation service dispatches to. The host owns the
    /// shutdown call, so a process mod's child does not outlive the application.
    /// A rebuild shares the same registry with the host it replaces: the mods
    /// are application-scoped, not backend-scoped.
    pub(crate) mod_registry: Arc<ModRegistry>,
    #[cfg(feature = "aigc")]
    pub(crate) aigc: AigcParts,
}

impl std::fmt::Debug for WorkspaceHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("WorkspaceHost");
        d.field("workspace_scope", &self.workspace_scope)
            .field("config", &self.config)
            .field("backend_config", &self.backend_config)
            .field("runtime_service", &self.runtime_service)
            .field("node_catalog", &self.node_catalog)
            .field("collection_service", &self.collection_service)
            .field("dynamic_workflow_service", &self.dynamic_workflow_service)
            .field("operation_service", &self.operation_service);
        #[cfg(feature = "aigc")]
        d.field("aigc", &self.aigc);
        d.finish_non_exhaustive()
    }
}

pub(crate) struct WorkspaceComponents {
    pub(crate) runtime_service: Arc<RuntimeService>,
    #[cfg(feature = "aigc")]
    pub(crate) model_service: Arc<ModelService>,
    #[cfg(feature = "aigc")]
    pub(crate) acquisition_service: Arc<ModelAcquisitionService>,
    #[cfg(feature = "aigc")]
    pub(crate) aigc: AigcParts,
}

impl WorkspaceHost {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assemble(
        workspace_scope: WorkspaceScope,
        config: Arc<AppConfig>,
        backend_config: InferenceBackendConfig,
        event_bus: EventBus,
        builtin_catalog: Arc<BuiltinNodeCatalog>,
        components: WorkspaceComponents,
        operation_service: Arc<OperationService>,
        mod_registry: Arc<ModRegistry>,
    ) -> Self {
        let collection_service = Arc::new(CollectionService::with_event_bus(
            config.paths().clone(),
            event_bus.clone(),
        ));
        let activity_service = Arc::new(ActivityService::new(
            config.paths().clone(),
            event_bus.clone(),
        ));
        operation_service.set_activity_service(Arc::clone(&activity_service));
        let board_service = Arc::new(BoardService::with_event_bus_and_activity(
            config.paths().clone(),
            event_bus.clone(),
            Some(Arc::clone(&activity_service)),
        ));
        let project_service = Arc::new(ProjectService::new(
            config.paths().clone(),
            Arc::clone(&board_service),
            Arc::clone(&collection_service),
        ));
        crate::collection_ops::register_collection_ops(
            &operation_service,
            Arc::clone(&collection_service),
            Arc::clone(&board_service),
        );
        crate::activity::register_activity_ops(&operation_service, activity_service);
        let dynamic_workflow_service = Arc::new(DynamicWorkflowService::with_event_bus(
            config.paths().clone(),
            event_bus.clone(),
            Arc::downgrade(&operation_service),
            Arc::clone(&collection_service),
        ));
        crate::dynamic_workflow_ops::register_dynamic_workflow_ops(
            &operation_service,
            Arc::clone(&dynamic_workflow_service),
        );
        let default_project_id = ProjectId::new("default");
        let workflow_service = Arc::new(WorkflowService::for_project_with_events(
            config.paths().clone(),
            default_project_id.clone(),
            event_bus.clone(),
        ));
        let mut workflow_services = BTreeMap::new();
        workflow_services.insert(default_project_id, Arc::clone(&workflow_service));
        #[cfg(feature = "aigc")]
        let node_catalog = Arc::new(NodeCatalogService::new(
            Arc::clone(&builtin_catalog),
            BackendSelection::from(backend_config.backend),
        ));
        #[cfg(not(feature = "aigc"))]
        let node_catalog = Arc::new(NodeCatalogService::new(Arc::clone(&builtin_catalog)));
        let services = Arc::new(WorkspaceServices::new(
            workspace_scope.clone(),
            Arc::clone(&config),
            Arc::clone(&workflow_service),
            #[cfg(feature = "aigc")]
            Arc::clone(&components.model_service),
            #[cfg(feature = "aigc")]
            components.acquisition_service,
            Arc::clone(&components.runtime_service),
            Arc::clone(&node_catalog),
        ));

        Self {
            workspace_scope,
            config,
            backend_config,
            project_service,
            board_service,
            collection_service,
            dynamic_workflow_service,
            workflow_service,
            workflow_services: RwLock::new(workflow_services),
            event_bus,
            runtime_service: components.runtime_service,
            node_catalog,
            builtin_catalog,
            services,
            operation_service,
            mod_registry,
            #[cfg(feature = "aigc")]
            aigc: components.aigc,
        }
    }

    #[cfg(feature = "aigc")]
    pub fn new(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        runtime_service: Arc<RuntimeService>,
        builtin_catalog: Arc<BuiltinNodeCatalog>,
        compute_profile: Arc<WorkspaceComputeProfile>,
        resolved_backend_instance: reimagine_inference::BackendInstance,
    ) -> Self {
        Self::new_with_event_bus(
            workspace_scope,
            config,
            backend_config,
            runtime_service,
            builtin_catalog,
            compute_profile,
            resolved_backend_instance,
            EventBus::new(),
        )
    }

    #[cfg(feature = "aigc")]
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_event_bus(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        runtime_service: Arc<RuntimeService>,
        builtin_catalog: Arc<BuiltinNodeCatalog>,
        compute_profile: Arc<WorkspaceComputeProfile>,
        resolved_backend_instance: reimagine_inference::BackendInstance,
        event_bus: EventBus,
    ) -> Self {
        let config = Arc::new(config);
        let acquisition_service = Arc::new(ModelAcquisitionService::with_event_bus(
            config.paths().clone(),
            &config,
            event_bus.clone(),
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            acquisition_service.clone(),
        ));
        let aigc = AigcParts {
            status: Health::Available,
            model_service: Arc::clone(&model_service),
            compute_profile,
            resolved_backend_instance,
            worker_switch: None,
            worker_inventory: Arc::new(crate::EmptyWorkerInventoryProvider),
            topology: None,
            discovery: None,
        };
        let components = WorkspaceComponents {
            runtime_service,
            model_service,
            acquisition_service,
            aigc,
        };
        let mod_registry = Arc::new(ModRegistry::empty());
        let operation_service = Arc::new(OperationService::new(
            config.paths().clone(),
            Arc::clone(&mod_registry),
        ));
        Self::assemble(
            workspace_scope,
            config,
            backend_config,
            event_bus,
            builtin_catalog,
            components,
            operation_service,
            mod_registry,
        )
    }

    pub fn with_defaults(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self::with_event_bus(workspace_scope, base_path, EventBus::new())
    }

    pub fn with_event_bus(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        event_bus: EventBus,
    ) -> Self {
        let config = AppConfig::new(AppPaths::new(base_path));
        let backend_config = load_backend_config(&config);
        Self::with_backend_config_inner(workspace_scope, config, backend_config, event_bus)
    }

    #[cfg(feature = "aigc")]
    pub fn with_defaults_and_backend(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        _backend_selection: BackendSelection,
        event_bus: EventBus,
    ) -> Self {
        Self::with_event_bus(workspace_scope, base_path, event_bus)
    }

    pub async fn try_with_defaults(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
    ) -> Result<Self, AppHostError> {
        Self::try_with_event_bus(workspace_scope, base_path, EventBus::new()).await
    }

    /// Bootstrap asynchronously, loading the backend config from disk and the
    /// worker inventory from the workspace.
    pub async fn try_with_event_bus(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        event_bus: EventBus,
    ) -> Result<Self, AppHostError> {
        let base_path = base_path.into();
        let config = AppConfig::new(AppPaths::new(&base_path));
        let backend_config = load_backend_config_result(&config).await?;
        Self::with_backend_config_and_worker_inventory_inner(
            workspace_scope,
            config,
            backend_config,
            event_bus,
            #[cfg(feature = "aigc")]
            Arc::new(InstalledWorkerInventoryProvider::for_base_path(&base_path)),
            Arc::new(ModRegistry::empty()),
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
    #[cfg(feature = "aigc")]
    pub async fn try_with_app_data_root_and_event_bus(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        app_data_root: impl Into<std::path::PathBuf>,
        event_bus: EventBus,
    ) -> Result<Self, AppHostError> {
        let base_path = base_path.into();
        let config = AppConfig::new(AppPaths::new(&base_path));
        let backend_config = load_backend_config_result(&config).await?;
        Self::try_with_app_data_root_and_backend_config(
            workspace_scope,
            &base_path,
            app_data_root,
            backend_config,
            event_bus,
        )
        .await
    }

    /// Bootstrap with an explicit app-data root and an explicit backend
    /// configuration (used by the re-bootstrap path).
    #[cfg(feature = "aigc")]
    pub async fn try_with_app_data_root_and_backend_config(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        app_data_root: impl Into<std::path::PathBuf>,
        backend_config: InferenceBackendConfig,
        event_bus: EventBus,
    ) -> Result<Self, AppHostError> {
        let config = AppConfig::new(AppPaths::new(base_path));
        Self::with_backend_config_and_worker_inventory_inner(
            workspace_scope,
            config,
            backend_config,
            event_bus,
            Arc::new(InstalledWorkerInventoryProvider::new(
                WorkerStorePaths::new(app_data_root),
            )),
            Arc::new(ModRegistry::empty()),
        )
        .await
    }

    /// Bootstrap with the desktop host's app-data worker convention and an
    /// already started [`ModRegistry`].
    ///
    /// The host dispatches operations to `mod_registry` and stops it from
    /// [`WorkspaceHost::shutdown`], so a process mod's child does not outlive
    /// the application. The other app-data constructor keeps an empty
    /// registry for callers that load no mods.
    ///
    /// Available with and without the `aigc` feature, so the desktop host can
    /// pass the same registry to either build; without `aigc` the app-data root
    /// has no worker inventory to describe and is ignored.
    #[cfg_attr(not(feature = "aigc"), allow(unused_variables))]
    pub async fn try_with_app_data_root_and_event_bus_and_mod_registry(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        app_data_root: impl Into<std::path::PathBuf>,
        event_bus: EventBus,
        mod_registry: Arc<ModRegistry>,
    ) -> Result<Self, AppHostError> {
        let base_path = base_path.into();
        let config = AppConfig::new(AppPaths::new(&base_path));
        let backend_config = load_backend_config_result(&config).await?;
        Self::with_backend_config_and_worker_inventory_inner(
            workspace_scope,
            config,
            backend_config,
            event_bus,
            #[cfg(feature = "aigc")]
            Arc::new(InstalledWorkerInventoryProvider::new(
                WorkerStorePaths::new(app_data_root),
            )),
            mod_registry,
        )
        .await
    }

    #[cfg(feature = "aigc")]
    pub async fn try_with_backend_config_and_worker_inventory(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        backend_config: InferenceBackendConfig,
        event_bus: EventBus,
        worker_inventory: Arc<dyn WorkerInventoryProvider>,
    ) -> Result<Self, AppHostError> {
        let config = AppConfig::new(AppPaths::new(base_path));
        Self::with_backend_config_and_worker_inventory_inner(
            workspace_scope,
            config,
            backend_config,
            event_bus,
            worker_inventory,
            Arc::new(ModRegistry::empty()),
        )
        .await
    }

    pub fn with_backend_config(
        workspace_scope: WorkspaceScope,
        base_path: impl Into<std::path::PathBuf>,
        backend_config: InferenceBackendConfig,
        event_bus: EventBus,
    ) -> Self {
        let config = AppConfig::new(AppPaths::new(base_path));
        Self::with_backend_config_inner(workspace_scope, config, backend_config, event_bus)
    }

    fn with_backend_config_inner(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        event_bus: EventBus,
    ) -> Self {
        let components = Self::build_sync_components(&config, &backend_config, &event_bus);
        let config = Arc::new(config);
        let mod_registry = Arc::new(ModRegistry::empty());
        let operation_service = Arc::new(OperationService::new(
            config.paths().clone(),
            Arc::clone(&mod_registry),
        ));
        Self::assemble(
            workspace_scope,
            config,
            backend_config,
            event_bus,
            Arc::new(BuiltinNodeCatalog::v1()),
            components,
            operation_service,
            mod_registry,
        )
    }

    #[cfg(feature = "aigc")]
    fn build_sync_components(
        config: &AppConfig,
        backend_config: &InferenceBackendConfig,
        event_bus: &EventBus,
    ) -> WorkspaceComponents {
        let acquisition_service = Arc::new(ModelAcquisitionService::with_event_bus(
            config.paths().clone(),
            config,
            event_bus.clone(),
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            acquisition_service.clone(),
        ));

        let (status, bootstrapped) =
            match bootstrap_inference(config, backend_config, model_service.clone()) {
                Ok(bootstrapped) => (Health::Available, bootstrapped),
                Err(error) => {
                    let reason = error.to_string();
                    tracing::error!(
                        %error,
                        "inference bootstrap failed on the synchronous construction path; \
                             continuing with an empty executor registry"
                    );
                    (
                        Health::Unavailable(aigc_parts::aigc_bootstrap_failed_diagnostic(reason)),
                        BootstrapInference {
                            runtime: ComposedInferenceRuntime::degraded(),
                            compute_profile: WorkspaceComputeProfile::new(),
                            topology: None,
                            discovery: None,
                        },
                    )
                }
            };
        let worker_switch = bootstrapped.runtime.worker_switch.clone();
        let runtime_service = Arc::new(
            RuntimeService::new(
                bootstrapped.runtime.executor_registry,
                bootstrapped.runtime.runtime_hooks.clone(),
                event_bus.clone(),
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
        let aigc = AigcParts {
            status,
            model_service: Arc::clone(&model_service),
            compute_profile: Arc::new(bootstrapped.compute_profile),
            resolved_backend_instance: bootstrapped.runtime.selected_instance,
            worker_switch,
            worker_inventory: Arc::new(crate::EmptyWorkerInventoryProvider),
            topology: bootstrapped.topology,
            discovery: bootstrapped.discovery,
        };

        WorkspaceComponents {
            runtime_service,
            model_service,
            acquisition_service,
            aigc,
        }
    }

    #[cfg(not(feature = "aigc"))]
    fn build_sync_components(
        _config: &AppConfig,
        _backend_config: &InferenceBackendConfig,
        event_bus: &EventBus,
    ) -> WorkspaceComponents {
        let runtime_service = Arc::new(RuntimeService::new(
            reimagine_inference::NodeExecutorRegistry::default(),
            Arc::new(reimagine_inference::CompositeBackendInstanceRuntimeHooks::new(Vec::new())),
            event_bus.clone(),
            Arc::new(reimagine_runtime::SystemClock),
        ));
        WorkspaceComponents { runtime_service }
    }

    async fn with_backend_config_and_worker_inventory_inner(
        workspace_scope: WorkspaceScope,
        config: AppConfig,
        backend_config: InferenceBackendConfig,
        event_bus: EventBus,
        #[cfg(feature = "aigc")] worker_inventory: Arc<dyn WorkerInventoryProvider>,
        mod_registry: Arc<ModRegistry>,
    ) -> Result<Self, AppHostError> {
        let base_path = config.paths().base_path().to_path_buf();
        let builder = WorkspaceHostBuilder::new(workspace_scope, base_path)
            .backend_config(backend_config)
            .event_bus(event_bus)
            .mod_registry(mod_registry);
        #[cfg(feature = "aigc")]
        let builder = builder.worker_inventory(worker_inventory);
        builder.build().await
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

    /// Return the project-owned workflow service, creating its in-memory
    /// session registry lazily. Callers must validate project existence
    /// through ProjectService before requesting a new project service.
    pub fn workflow_service_for_project(&self, project_id: &ProjectId) -> Arc<WorkflowService> {
        if let Some(service) = self
            .workflow_services
            .read()
            .expect("workflow service registry poisoned")
            .get(project_id)
            .cloned()
        {
            return service;
        }
        let service = Arc::new(WorkflowService::for_project_with_events(
            self.config.paths().clone(),
            project_id.clone(),
            self.event_bus.clone(),
        ));
        self.workflow_services
            .write()
            .expect("workflow service registry poisoned")
            .insert(project_id.clone(), Arc::clone(&service));
        service
    }

    /// Delete a project and clear its in-memory workflow session registry.
    pub async fn delete_project(&self, project_id: &ProjectId) -> Result<(), AppHostError> {
        self.project_service.delete_project(project_id).await?;
        if project_id.as_str() == "default" {
            self.workflow_service.clear();
        } else if let Some(service) = self
            .workflow_services
            .write()
            .expect("workflow service registry poisoned")
            .remove(project_id)
        {
            service.clear();
        }
        Ok(())
    }

    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }

    pub fn project_service(&self) -> &Arc<ProjectService> {
        &self.project_service
    }
    pub fn board_service(&self) -> &Arc<BoardService> {
        &self.board_service
    }
    #[cfg(feature = "aigc")]
    pub fn model_service(&self) -> &Arc<ModelService> {
        &self.aigc.model_service
    }
    pub fn runtime_service(&self) -> &Arc<RuntimeService> {
        &self.runtime_service
    }

    pub fn operation_service(&self) -> &Arc<OperationService> {
        &self.operation_service
    }

    /// Operating health of the workspace AIGC subsystem.
    pub fn aigc_health(&self) -> &Health {
        #[cfg(feature = "aigc")]
        {
            &self.aigc.status
        }
        #[cfg(not(feature = "aigc"))]
        {
            &aigc_parts::AIGC_NOT_INSTALLED_HEALTH
        }
    }

    /// Reason why AIGC is unavailable, if any.
    pub(crate) fn aigc_unavailable_reason(&self) -> Option<&str> {
        self.aigc_health().diagnostic().map(|diag| diag.message())
    }

    #[cfg(feature = "aigc")]
    pub async fn selected_worker(&self) -> Result<crate::WorkerSelectionHandle, AppHostError> {
        if let Some(reason) = self.aigc_unavailable_reason() {
            return Err(AppHostError::AigcUnavailable {
                reason: reason.to_string(),
            });
        }
        let workers = self
            .aigc
            .worker_switch
            .as_ref()
            .ok_or(crate::WorkerSwitchError::NoActiveWorker)?;
        Ok(workers.selected().await)
    }

    #[cfg(feature = "aigc")]
    pub async fn drain_and_switch_worker(
        &self,
        target: crate::WorkerBackendCandidate,
        deadline: std::time::Duration,
    ) -> Result<crate::WorkerSelectionHandle, AppHostError> {
        if let Some(reason) = self.aigc_unavailable_reason() {
            return Err(AppHostError::AigcUnavailable {
                reason: reason.to_string(),
            });
        }
        let workers = self
            .aigc
            .worker_switch
            .as_ref()
            .ok_or(crate::WorkerSwitchError::NoActiveWorker)?;
        workers
            .drain_and_switch(Arc::new(target), deadline)
            .await
            .map_err(AppHostError::from)
    }

    #[cfg(feature = "aigc")]
    pub async fn cancel_and_switch_worker(
        &self,
        target: crate::WorkerBackendCandidate,
        deadline: std::time::Duration,
    ) -> Result<crate::WorkerSelectionHandle, AppHostError> {
        if let Some(reason) = self.aigc_unavailable_reason() {
            return Err(AppHostError::AigcUnavailable {
                reason: reason.to_string(),
            });
        }
        let workers = self
            .aigc
            .worker_switch
            .as_ref()
            .ok_or(crate::WorkerSwitchError::NoActiveWorker)?;
        workers
            .cancel_and_switch(Arc::new(target), deadline)
            .await
            .map_err(AppHostError::from)
    }

    /// Wait for in-flight runs on the active worker to complete (up to
    /// `deadline`) and then shut the worker down gracefully.
    ///
    /// This is the drain half of the re-bootstrap path: it guarantees no
    /// run is in flight when the old worker process is retired, without
    /// cancelling anything. Hosts without a process worker (built-in
    /// fallback backends) return immediately.
    #[cfg(feature = "aigc")]
    pub async fn drain_active_worker(
        &self,
        deadline: std::time::Duration,
    ) -> Result<(), crate::WorkerSwitchError> {
        let Some(worker_switch) = &self.aigc.worker_switch else {
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
    #[cfg(feature = "aigc")]
    pub async fn rebuild_workspace(
        current: &Self,
        new_selection: BackendSelection,
    ) -> Result<Self, AppHostError> {
        let backend_config = backend_config_for_selection(&current.backend_config, new_selection);
        let rebuilt = Self::with_backend_config_and_worker_inventory_inner(
            current.workspace_scope.clone(),
            (*current.config).clone(),
            backend_config,
            current.event_bus.clone(),
            Arc::clone(&current.aigc.worker_inventory),
            Arc::clone(&current.mod_registry),
        )
        .await?;
        if let Some(reason) = rebuilt.aigc_unavailable_reason() {
            rebuilt.shutdown_owned().await;
            return Err(AppHostError::InferenceBootstrap {
                message: reason.to_string(),
            });
        }
        if let Err(error) = current
            .drain_active_worker(REBOOTSTRAP_DRAIN_DEADLINE)
            .await
        {
            rebuilt.shutdown_owned().await;
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
    #[cfg(feature = "aigc")]
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

    #[cfg(feature = "aigc")]
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
    #[cfg(feature = "aigc")]
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
    /// backend plus a top-level diagnostics
    /// collection that records any fallback decisions made during
    /// bootstrap (e.g. `mps` → `cpu` when Metal is unavailable, or any
    /// unknown configured device label).
    ///
    /// The accessor returns a clone of the cached profile and does not
    /// re-probe the host; the snapshot is computed once during
    /// [`WorkspaceHost`] bootstrap.
    #[cfg(feature = "aigc")]
    pub fn compute_profile(&self) -> WorkspaceComputeProfile {
        (*self.aigc.compute_profile).clone()
    }

    /// The connection topology manager (T13), when remote workers are
    /// configured. `None` in single-worker mode.
    #[cfg(feature = "aigc")]
    pub fn topology(
        &self,
    ) -> Option<&Arc<tokio::sync::Mutex<aigc::inference::topology::ConnectionTopologyManager>>>
    {
        self.aigc.topology.as_ref()
    }

    /// The discovery orchestrator (T12/T13), when a topology manager
    /// exists.
    #[cfg(feature = "aigc")]
    pub fn discovery(&self) -> Option<&Arc<aigc::inference::discovery::DiscoveryOrchestrator>> {
        self.aigc.discovery.as_ref()
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
    #[cfg(feature = "aigc")]
    pub fn compute_profile_dto(&self) -> crate::dto::ComputeProfileDto {
        let mut dto: crate::dto::ComputeProfileDto = self.compute_profile().into();
        if let Some(topology) = &self.aigc.topology {
            // try_lock: the discovery poll loop may hold the lock; a
            // busy lock just means this snapshot omits topology workers.
            if let Ok(guard) = topology.try_lock() {
                dto =
                    dto.with_topology_workers(crate::dto::topology_workers_from_pool(guard.pool()));
            }
        }
        dto
    }

    /// Return the resolved [`reimagine_inference::BackendInstance`]
    /// the workspace bootstrap selected.
    #[cfg(feature = "aigc")]
    pub fn resolved_backend_instance(&self) -> &reimagine_inference::BackendInstance {
        &self.aigc.resolved_backend_instance
    }

    /// Shut down the active inference worker and the registered mods, if any.
    ///
    /// This is the application-level shutdown hook. Call it when the host
    /// is exiting so that child worker and mod processes are cleaned up
    /// instead of becoming orphans. Repeated calls stop the mods once:
    /// [`ModRegistry::stop`] owns the idempotency guard.
    ///
    /// A replacement host shares the registry, so the registry is not owned by
    /// either host: a failed or superseded replacement must be discarded with
    /// `shutdown_owned`, and callers must not shut down a superseded host.
    /// Either host's `shutdown` would stop the mods the other one still
    /// dispatches to.
    pub async fn shutdown(&self) {
        self.shutdown_owned().await;
        self.mod_registry.stop().await;
    }

    /// Stop what this host owns outright: in-flight dynamic workflow runs and
    /// its inference worker.
    ///
    /// Mods are deliberately excluded. A re-bootstrap builds a replacement
    /// host around the current registry, and a failed replacement must be
    /// discarded without stopping mods the surviving host still dispatches to.
    async fn shutdown_owned(&self) {
        // Abort in-flight dynamic workflow runs; their documents stay as they
        // are, so the next process reports them as interrupted.
        self.dynamic_workflow_service.shutdown();
        #[cfg(feature = "aigc")]
        {
            let Some(worker_switch) = &self.aigc.worker_switch else {
                return;
            };
            let deadline = std::time::Duration::from_secs(5);
            if let Err(error) = worker_switch.shutdown_active(deadline).await {
                eprintln!("[app-host] worker shutdown error: {error}");
            }
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
#[cfg(feature = "aigc")]
fn backend_config_for_selection(
    current: &InferenceBackendConfig,
    selection: BackendSelection,
) -> InferenceBackendConfig {
    let backend = match selection {
        BackendSelection::Burn => reimagine_config::InferenceBackendKind::Burn,
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
    #[cfg(feature = "aigc")]
    use reimagine_backend_worker_host::{
        ExpectedWorkerIdentity, InstallationRecord, InventoryStore, WorkerInstallationId,
    };
    #[cfg(feature = "aigc")]
    use reimagine_backend_worker_protocol::BackendInstanceId;
    use reimagine_config::InferenceBackendKind;
    #[cfg(feature = "aigc")]
    use reimagine_core::model::NodeTypeId;
    #[cfg(feature = "aigc")]
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

    #[cfg(feature = "aigc")]
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
            protocol_version_min: 2,
            protocol_version_max: 2,
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
    fn workspace_from_config_file_migrates_legacy_candle_to_burn() {
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
            InferenceBackendKind::Burn
        );
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

    #[cfg(feature = "aigc")]
    #[test]
    fn app_host_inference_composition_registers_builtin_executors() {
        let base = temp_dir("compose-runtime");
        let config = AppConfig::new(reimagine_config::AppPaths::new(&base));
        let event_bus = reimagine_core::event::EventBus::new();
        let acquisition_service = Arc::new(ModelAcquisitionService::with_event_bus(
            config.paths().clone(),
            &config,
            event_bus.clone(),
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            acquisition_service,
        ));

        let composed = compose_inference_runtime(
            &config,
            BackendInstance::new("stub:cpu"),
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
            r#"{"backend": "burn", "burn_device": "cpu"}"#,
        )
        .unwrap();

        let workspace =
            WorkspaceHost::try_with_defaults(WorkspaceScope::new("test-try-valid"), &base)
                .await
                .expect("valid config should succeed");
        assert_eq!(
            workspace.backend_config().backend,
            InferenceBackendKind::Burn
        );
        assert_eq!(workspace.backend_config().burn_device, "cpu");
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

    #[cfg(feature = "aigc")]
    #[test]
    fn compute_profile_reports_uninstalled_burn_worker_when_no_worker_present() {
        let base = temp_dir("profile-uninstalled-burn");
        let workspace =
            WorkspaceHost::with_defaults(WorkspaceScope::new("profile-uninstalled-burn"), &base);
        let profile = workspace.compute_profile();
        let total_instances = profile
            .backend_profiles
            .iter()
            .map(|bp| bp.instances.len())
            .sum::<usize>();
        assert_eq!(total_instances, 0);
        let burn_profile = profile
            .backend_profiles
            .iter()
            .find(|bp| bp.backend.as_str() == "burn")
            .expect("must have burn profile");
        let diagnostic = burn_profile
            .diagnostics
            .iter()
            .find(|d| d.code().as_str() == "APP_HOST/LOCAL_WORKER_NOT_INSTALLED")
            .expect("must report LOCAL_WORKER_NOT_INSTALLED diagnostic");
        assert!(diagnostic.message().contains("Burn"));
        let _ = fs::remove_dir_all(&base);
    }

    #[cfg(feature = "aigc")]
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

    #[cfg(feature = "aigc")]
    #[test]
    fn unknown_open_selected_instance_remains_pinned_with_diagnostic() {
        let base = temp_dir("profile-unknown-selected");
        let backend_config = InferenceBackendConfig {
            selected_instance: Some("ghost:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::with_backend_config(
            WorkspaceScope::new("profile-unknown-selected"),
            &base,
            backend_config,
            EventBus::new(),
        );

        assert_eq!(
            workspace.resolved_backend_instance(),
            &BackendInstance::new("ghost:cpu"),
            "unknown explicit selection must fail closed"
        );
        let profile = workspace.compute_profile();
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

    #[cfg(feature = "aigc")]
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
        assert!(!profile.backend_profiles.is_empty());
        let _ = fs::remove_dir_all(&base);
    }

    // ── App-data root wiring (BE-38) ──────────────────────────────

    #[cfg(feature = "aigc")]
    fn burn_worker_visible(profile: &reimagine_inference::WorkspaceComputeProfile) -> bool {
        profile
            .backend_profiles
            .iter()
            .flat_map(|backend| backend.instances.iter())
            .any(|instance| instance.instance == BackendInstance::new("burn:wgpu:default"))
    }

    #[cfg(feature = "aigc")]
    #[tokio::test]
    async fn try_with_app_data_root_reads_worker_store_beneath_explicit_root() {
        let base = temp_dir("app-data-root");
        let app_data_root = base.join("app-data");
        let install_dir = base.join("installed-worker");
        fs::create_dir_all(&install_dir).unwrap();
        seed_worker_record(&app_data_root, &install_dir);

        // Selection pinned to an explicit non-activated instance so the seeded
        // burn worker is never activated (no process spawn in tests).
        let backend_config = InferenceBackendConfig {
            selected_instance: Some("stub:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("app-data-root"),
            &base,
            &app_data_root,
            backend_config,
            EventBus::new(),
        )
        .await
        .expect("bootstrap with explicit app-data root");

        assert_eq!(
            workspace.resolved_backend_instance(),
            &BackendInstance::new("stub:cpu")
        );
        assert!(
            burn_worker_visible(&workspace.compute_profile()),
            "profile must include the worker installed beneath the explicit app-data root"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[cfg(feature = "aigc")]
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

    #[cfg(feature = "aigc")]
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

        // The selection must be persisted so a restarted app boots with it.
        let config_path = base.join("config").join(InferenceBackendConfig::KEY);
        let persisted: InferenceBackendConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path).expect("persisted backend config file"),
        )
        .expect("persisted config parses");
        assert_eq!(
            persisted.backend,
            InferenceBackendKind::Burn,
            "rebootstrap must persist the selection to inference_backend.json"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[cfg(feature = "aigc")]
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

    #[cfg(feature = "aigc")]
    #[tokio::test]
    async fn rebuild_workspace_preserves_current_when_rebuild_fails() {
        let base = temp_dir("rebuild-fail-preserve");
        let mut original =
            WorkspaceHost::try_with_defaults(WorkspaceScope::new("rebuild-fail"), &base)
                .await
                .expect("initial bootstrap must succeed");
        assert!(original.aigc_health().is_available());

        // Configure an invalid worker transport in backend_config so rebuilding fails inference bootstrap.
        original.backend_config = InferenceBackendConfig {
            workers: vec![reimagine_config::WorkerEndpointConfig {
                id: "invalid-worker".to_owned(),
                transport: "carrier-pigeon".to_owned(),
                address: "https://127.0.0.1:9999".to_owned(),
                capabilities: Vec::new(),
                device_label: String::new(),
                fingerprint: None,
            }],
            ..InferenceBackendConfig::default()
        };

        let err = original
            .rebootstrap(BackendSelection::Burn)
            .await
            .expect_err("rebootstrap must fail when new inference bootstrap is unavailable");
        assert!(matches!(err, AppHostError::InferenceBootstrap { .. }));

        // Current workspace host remains untouched and available.
        assert!(original.aigc_health().is_available());
        let default_project = reimagine_core::model::ProjectId::new("default");
        assert_eq!(
            original
                .workflow_service_for_project(&default_project)
                .project_id(),
            &default_project
        );

        let _ = tokio::fs::remove_dir_all(&base).await;
    }

    #[cfg(feature = "aigc")]
    #[tokio::test]
    async fn production_bootstrap_creates_discoverable_default_project() {
        let base = temp_dir("ar08-default-project");
        let app_data_root = base.join("app-data");
        let backend_config = InferenceBackendConfig {
            selected_instance: Some("stub:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("ar08-default-project"),
            &base,
            &app_data_root,
            backend_config.clone(),
            EventBus::new(),
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
            EventBus::new(),
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

    #[cfg(feature = "aigc")]
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
            selected_instance: Some("stub:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };
        let workspace = WorkspaceHost::try_with_app_data_root_and_backend_config(
            WorkspaceScope::new("ar08-legacy-migrate"),
            &base,
            &app_data_root,
            backend_config,
            EventBus::new(),
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
