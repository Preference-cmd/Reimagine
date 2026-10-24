use std::path::PathBuf;
use std::sync::Arc;

use crate::WorkspaceScope;
#[cfg(feature = "aigc")]
use reimagine_backend_worker_host::WorkerStorePaths;
use reimagine_config::{AppConfig, AppPaths, InferenceBackendConfig};
use reimagine_runtime::BoxedRunEventSink;
use reimagine_runtime::VecRunEventSink;

#[cfg(feature = "aigc")]
use crate::InstalledWorkerInventoryProvider;
use crate::{AppHostError, DocumentEventBus};
#[cfg(feature = "aigc")]
use crate::{ModelService, WorkerInventoryProvider};
#[cfg(feature = "aigc")]
use aigc::ModelAcquisitionService;
#[cfg(feature = "aigc")]
use aigc::inference::compose::{
    BootstrapInference, ComposedInferenceRuntime, bootstrap_inference_with_worker_inventory,
};
#[cfg(feature = "aigc")]
use reimagine_inference::WorkspaceComputeProfile;

#[cfg(feature = "aigc")]
use super::{AigcParts, AigcStatus};
use super::{WorkspaceHost, load_backend_config_result, migrate_project_layout};

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
    #[cfg(feature = "aigc")]
    worker_inventory: Option<Arc<dyn WorkerInventoryProvider>>,
    document_events: Option<Arc<DocumentEventBus>>,
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
            #[cfg(feature = "aigc")]
            worker_inventory: None,
            document_events: None,
        }
    }

    /// Set an existing document event bus.
    pub fn document_events(mut self, bus: Arc<DocumentEventBus>) -> Self {
        self.document_events = Some(bus);
        self
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
    #[cfg(feature = "aigc")]
    pub fn worker_inventory(mut self, provider: Arc<dyn WorkerInventoryProvider>) -> Self {
        self.worker_inventory = Some(provider);
        self
    }

    /// Build the [`WorkspaceHost`].
    ///
    /// This async phase:
    /// 1. Loads backend configuration (from file or explicit)
    /// 2. Creates model acquisition and model services (with AIGC)
    /// 3. Bootstraps the inference runtime with worker inventory (with AIGC)
    /// 4. Wires all services together
    /// 5. Returns the fully initialized workspace
    pub async fn build(self) -> Result<WorkspaceHost, AppHostError> {
        let config = AppConfig::new(AppPaths::new(&self.base_path));

        // Establish the project workflow layout (default project and
        // legacy migration) before bootstrapping inference, so no await
        // sits between the discovery poll loop spawn and host construction.
        migrate_project_layout(&config).await?;

        // Load backend config: explicit or from file
        let backend_config = match &self.backend_config {
            Some(config) => config.clone(),
            None => load_backend_config_result(&config).await?,
        };

        // Resolve the run event sink
        let event_sink = self
            .event_sink
            .clone()
            .unwrap_or_else(|| Arc::new(VecRunEventSink::new()));

        let components = self
            .build_components(&config, &backend_config, &event_sink)
            .await;

        let document_events = self
            .document_events
            .unwrap_or_else(|| Arc::new(DocumentEventBus::new()));

        Ok(WorkspaceHost::assemble(
            self.workspace_scope,
            Arc::new(config),
            backend_config,
            document_events,
            event_sink,
            Arc::new(reimagine_nodes::BuiltinNodeCatalog::v1()),
            components,
        ))
    }

    #[cfg(feature = "aigc")]
    async fn build_components(
        &self,
        config: &AppConfig,
        backend_config: &InferenceBackendConfig,
        event_sink: &BoxedRunEventSink,
    ) -> super::WorkspaceComponents {
        // Resolve worker inventory
        let worker_inventory = match self.worker_inventory {
            Some(ref provider) => Arc::clone(provider),
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
        let acquisition_service =
            Arc::new(ModelAcquisitionService::new(config.paths().clone(), config));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            Arc::clone(&acquisition_service),
        ));

        // Bootstrap inference runtime (non-fatal on workspace build)
        let (bootstrapped, aigc_status) = match bootstrap_inference_with_worker_inventory(
            config,
            backend_config,
            Arc::clone(&model_service),
            Arc::clone(&worker_inventory),
        )
        .await
        {
            Ok(bootstrapped) => (bootstrapped, AigcStatus::Available),
            Err(error) => {
                let reason = error.to_string();
                tracing::error!(
                    %error,
                    "inference bootstrap failed on workspace build; continuing with degraded runtime"
                );
                (
                    BootstrapInference {
                        runtime: ComposedInferenceRuntime::degraded(),
                        compute_profile: WorkspaceComputeProfile::new(),
                        topology: None,
                        discovery: None,
                    },
                    AigcStatus::Unavailable { reason },
                )
            }
        };

        // Wire runtime service
        let worker_switch = bootstrapped.runtime.worker_switch.clone();
        let runtime_service = Arc::new(
            reimagine_runtime::RuntimeService::new(
                bootstrapped.runtime.executor_registry,
                bootstrapped.runtime.runtime_hooks.clone(),
                Arc::clone(event_sink),
                Arc::new(reimagine_runtime::SystemClock),
            )
            .with_resource_hint_sink(worker_switch.as_ref().and_then(|ws| ws.active_hint_sink())),
        );

        // Wire run cancellation to worker switch
        if let Some(ref worker_switch) = worker_switch {
            let cancellation: Arc<dyn crate::RunCancellation> = runtime_service.clone();
            worker_switch.set_run_cancellation(cancellation);
        }

        let aigc = AigcParts {
            status: aigc_status,
            model_service: Arc::clone(&model_service),
            compute_profile: Arc::new(bootstrapped.compute_profile),
            resolved_backend_instance: bootstrapped.runtime.selected_instance,
            worker_switch,
            worker_inventory,
            topology: bootstrapped.topology,
            discovery: bootstrapped.discovery,
        };

        super::WorkspaceComponents {
            runtime_service,
            model_service,
            acquisition_service,
            aigc,
        }
    }

    #[cfg(not(feature = "aigc"))]
    async fn build_components(
        &self,
        _config: &AppConfig,
        _backend_config: &InferenceBackendConfig,
        event_sink: &BoxedRunEventSink,
    ) -> super::WorkspaceComponents {
        let runtime_service = Arc::new(reimagine_runtime::RuntimeService::new(
            reimagine_inference::NodeExecutorRegistry::default(),
            Arc::new(reimagine_inference::CompositeBackendInstanceRuntimeHooks::new(Vec::new())),
            Arc::clone(event_sink),
            Arc::new(reimagine_runtime::SystemClock),
        ));
        super::WorkspaceComponents { runtime_service }
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
            backend: reimagine_config::InferenceBackendKind::Burn,
            burn_device: "cpu".to_string(),
            ..InferenceBackendConfig::default()
        };

        let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test"), &base)
            .backend_config(config)
            .build()
            .await
            .expect("builder with explicit config");

        assert_eq!(
            host.backend_config().backend,
            reimagine_config::InferenceBackendKind::Burn
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
