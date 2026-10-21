use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::WorkspaceScope;
use reimagine_backend_worker_host::WorkerStorePaths;
use reimagine_config::{AppConfig, AppPaths, InferenceBackendConfig};
use reimagine_runtime::BoxedRunEventSink;
use reimagine_runtime::VecRunEventSink;

use crate::InstalledWorkerInventoryProvider;
use crate::inference::compose::bootstrap_inference_with_worker_inventory;
use crate::model_acquisition_service::ModelAcquisitionService;
use crate::project_memory::ProjectMemoryService;
use crate::services::WorkspaceServices;
use crate::{
    AppHostError, BackendSelection, BoardService, DocumentEventBus, ModelService, ProjectService,
    WorkerInventoryProvider, WorkflowService,
};

use super::{WorkspaceHost, load_backend_config_result};

/// Builder for constructing a [`WorkspaceHost`].
///
/// All configuration is expressed as builder methods; the async [`build`](Self::build)
/// phase handles config loading, backend bootstrap, and service wiring.
///
/// # Examples
///
/// ```rust,no_run
/// # use reimagine_app_host::WorkspaceHostBuilder;
/// # use reimagine_app_host::WorkspaceScope;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let host = WorkspaceHostBuilder::new(WorkspaceScope::new("my-workspace"), "/tmp/workspace")
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct WorkspaceHostBuilder {
    // Required
    workspace_scope: WorkspaceScope,
    base_path: PathBuf,

    // Optional - with defaults
    app_data_root: Option<PathBuf>,
    backend_config: Option<InferenceBackendConfig>,
    event_sink: Option<BoxedRunEventSink>,
    worker_inventory: Option<Arc<dyn WorkerInventoryProvider>>,
}

impl WorkspaceHostBuilder {
    /// Create a new builder with the required parameters.
    ///
    /// `workspace_scope` identifies the workspace for logging and isolation.
    /// `base_path` is the root directory for workspace data (config, artifacts, etc.).
    pub fn new(workspace_scope: WorkspaceScope, base_path: impl Into<PathBuf>) -> Self {
        Self {
            workspace_scope,
            base_path: base_path.into(),
            app_data_root: None,
            backend_config: None,
            event_sink: None,
            worker_inventory: None,
        }
    }

    /// Set the application data root for worker inventory.
    ///
    /// When set, the builder uses `InstalledWorkerInventoryProvider` with this
    /// root instead of deriving it from `base_path`.
    ///
    /// Use this in Tauri desktop hosts where the app-data directory is separate
    /// from the workspace directory.
    pub fn app_data_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.app_data_root = Some(path.into());
        self
    }

    /// Set an explicit backend configuration.
    ///
    /// When set, the builder skips loading `inference_backend.json` and uses
    /// this config directly.
    pub fn backend_config(mut self, config: InferenceBackendConfig) -> Self {
        self.backend_config = Some(config);
        self
    }

    /// Set the run event sink for workflow execution events.
    ///
    /// When not set, uses [`VecRunEventSink`] (events are discarded).
    pub fn event_sink(mut self, sink: BoxedRunEventSink) -> Self {
        self.event_sink = Some(sink);
        self
    }

    /// Set a custom worker inventory provider.
    ///
    /// When not set, uses [`InstalledWorkerInventoryProvider`] with either
    /// the explicit `app_data_root` or the derived path.
    pub fn worker_inventory(mut self, provider: Arc<dyn WorkerInventoryProvider>) -> Self {
        self.worker_inventory = Some(provider);
        self
    }

    /// Build the [`WorkspaceHost`].
    ///
    /// This async phase:
    /// 1. Loads backend configuration (from file or explicit)
    /// 2. Creates model acquisition and model services
    /// 3. Bootstraps the inference runtime with worker inventory
    /// 4. Wires all services together
    /// 5. Returns the fully initialized workspace
    pub async fn build(self) -> Result<WorkspaceHost, AppHostError> {
        let config = AppConfig::new(AppPaths::new(&self.base_path));

        // Load backend config: explicit or from file
        let backend_config = match self.backend_config {
            Some(config) => config,
            None => load_backend_config_result(&config).await?,
        };

        // Resolve the run event sink
        let event_sink = self
            .event_sink
            .unwrap_or_else(|| Arc::new(VecRunEventSink::new()));

        // Resolve worker inventory
        let worker_inventory = match self.worker_inventory {
            Some(provider) => provider,
            None => match &self.app_data_root {
                Some(app_data_root) => Arc::new(InstalledWorkerInventoryProvider::new(
                    WorkerStorePaths::new(app_data_root),
                )),
                None => Arc::new(InstalledWorkerInventoryProvider::for_base_path(
                    &self.base_path,
                )),
            },
        };

        // Create services
        let acquisition_service = Arc::new(ModelAcquisitionService::new(
            config.paths().clone(),
            &config,
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            Arc::clone(&acquisition_service),
        ));

        // Bootstrap inference runtime
        let bootstrapped = bootstrap_inference_with_worker_inventory(
            &config,
            &backend_config,
            Arc::clone(&model_service),
            Arc::clone(&worker_inventory),
        )
        .await
        .map_err(|error| AppHostError::InferenceBootstrap {
            message: error.to_string(),
        })?;

        // Wire runtime service
        let worker_switch = bootstrapped.runtime.worker_switch.clone();
        let runtime_service = Arc::new(
            reimagine_runtime::RuntimeService::new(
                bootstrapped.runtime.executor_registry,
                bootstrapped.runtime.runtime_hooks.clone(),
                Arc::clone(&event_sink),
                Arc::new(reimagine_runtime::SystemClock),
            )
            .with_resource_hint_sink(worker_switch.as_ref().and_then(|ws| ws.active_hint_sink())),
        );

        // Wire run cancellation to worker switch
        if let Some(ref worker_switch) = worker_switch {
            let cancellation: Arc<dyn crate::RunCancellation> = runtime_service.clone();
            worker_switch.set_run_cancellation(cancellation);
        }

        // Create remaining services
        let builtin_catalog = Arc::new(reimagine_nodes::BuiltinNodeCatalog::v1());
        let backend = BackendSelection::from(backend_config.backend);
        let node_catalog = Arc::new(crate::node_catalog::NodeCatalogService::new(
            Arc::clone(&builtin_catalog),
            backend,
        ));
        let document_events = Arc::new(DocumentEventBus::new());
        let default_project_id = reimagine_core::model::ProjectId::new("default");
        let workflow_service = Arc::new(WorkflowService::for_project_with_events(
            config.paths().clone(),
            default_project_id.clone(),
            Arc::clone(&document_events),
        ));
        let mut workflow_services = BTreeMap::new();
        workflow_services.insert(default_project_id, Arc::clone(&workflow_service));
        let board_service = Arc::new(BoardService::with_document_events(
            config.paths().clone(),
            Arc::clone(&document_events),
        ));
        let project_service = Arc::new(ProjectService::new(
            config.paths().clone(),
            Arc::clone(&board_service),
        ));
        let project_memory_service = Arc::new(ProjectMemoryService::new(config.paths().clone()));
        let services = Arc::new(WorkspaceServices::new(
            self.workspace_scope.clone(),
            Arc::new(config.clone()),
            Arc::clone(&workflow_service),
            Arc::clone(&model_service),
            acquisition_service,
            Arc::clone(&runtime_service),
            Arc::clone(&node_catalog),
            Arc::clone(&project_memory_service),
        ));

        // Assemble workspace
        Ok(WorkspaceHost {
            workspace_scope: self.workspace_scope,
            config: Arc::new(config),
            backend_config,
            project_service,
            board_service,
            workflow_service,
            workflow_services: RwLock::new(workflow_services),
            document_events,
            model_service,
            runtime_service,
            node_catalog,
            builtin_catalog,
            services,
            compute_profile: Arc::new(bootstrapped.compute_profile),
            resolved_backend_instance: bootstrapped.runtime.selected_instance,
            worker_switch,
            worker_inventory,
            topology: bootstrapped.topology,
            discovery: bootstrapped.discovery,
            event_sink,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tid = std::thread::current().id();
        std::env::temp_dir().join(format!("reimagine-builder-{prefix}-{nonce:?}-{tid:?}"))
    }

    #[tokio::test]
    async fn builder_with_defaults_succeeds() {
        let base = temp_dir("defaults");
        let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test"), &base)
            .build()
            .await
            .expect("builder with defaults should succeed");

        assert_eq!(host.base_path(), base);
        assert_eq!(
            host.backend_config().backend,
            reimagine_config::InferenceBackendKind::Burn
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn builder_with_explicit_config() {
        let base = temp_dir("explicit-config");
        let config = InferenceBackendConfig {
            backend: reimagine_config::InferenceBackendKind::Candle,
            candle_device: "cpu".to_string(),
            ..InferenceBackendConfig::default()
        };

        let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test"), &base)
            .backend_config(config)
            .build()
            .await
            .expect("builder with explicit config");

        assert_eq!(
            host.backend_config().backend,
            reimagine_config::InferenceBackendKind::Candle
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn builder_with_app_data_root() {
        let base = temp_dir("app-data");
        let app_data = base.join("app-data");
        fs::create_dir_all(&app_data).unwrap();

        let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test"), &base)
            .app_data_root(&app_data)
            .build()
            .await
            .expect("builder with app data root");

        assert_eq!(host.base_path(), base);
        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn builder_with_event_sink() {
        let base = temp_dir("sinks");
        let event_sink: BoxedRunEventSink = Arc::new(VecRunEventSink::new());

        let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test"), &base)
            .event_sink(event_sink)
            .build()
            .await
            .expect("builder with event sink");

        assert_eq!(host.base_path(), base);
        let _ = fs::remove_dir_all(&base);
    }
}
