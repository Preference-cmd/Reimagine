use std::path::PathBuf;
use std::sync::Arc;

use crate::WorkspaceScope;
#[cfg(feature = "aigc")]
use reimagine_backend_worker_host::WorkerStorePaths;
use reimagine_config::{AppConfig, AppPaths, InferenceBackendConfig};
use reimagine_core::event::EventBus;

use crate::AppHostError;
#[cfg(feature = "aigc")]
use crate::InstalledWorkerInventoryProvider;
#[cfg(feature = "aigc")]
use crate::{ModelService, WorkerInventoryProvider};
#[cfg(feature = "aigc")]
use aigc::ModelAcquisitionService;
#[cfg(feature = "aigc")]
use aigc::inference::compose::{
    BootstrapInference, ComposedInferenceRuntime, bootstrap_inference_with_worker_inventory,
};
use mod_sdk::ModRegistry;
#[cfg(feature = "aigc")]
use reimagine_inference::WorkspaceComputeProfile;
use reimagine_runtime::RuntimeService;

#[cfg(feature = "aigc")]
use super::AigcParts;
#[cfg(feature = "aigc")]
use super::aigc_parts::aigc_bootstrap_failed_diagnostic;
use super::{WorkspaceHost, load_backend_config_result, migrate_project_layout};
#[cfg(feature = "aigc")]
use reimagine_core::health::Health;

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
    event_bus: Option<EventBus>,
    mod_registry: Option<Arc<ModRegistry>>,
    #[cfg(feature = "aigc")]
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
            event_bus: None,
            mod_registry: None,
            #[cfg(feature = "aigc")]
            worker_inventory: None,
        }
    }

    /// Set an existing event bus.
    pub fn event_bus(mut self, bus: EventBus) -> Self {
        self.event_bus = Some(bus);
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

    /// Set a custom mod registry.
    pub fn mod_registry(mut self, registry: Arc<ModRegistry>) -> Self {
        self.mod_registry = Some(registry);
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

        let event_bus = self.event_bus.clone().unwrap_or_default();

        let components = self
            .build_components(&config, &backend_config, &event_bus)
            .await;

        let mod_registry = self
            .mod_registry
            .unwrap_or_else(|| Arc::new(ModRegistry::empty()));
        let operation_service = Arc::new(crate::OperationService::new(
            config.paths().clone(),
            Arc::clone(&mod_registry),
        ));

        Ok(WorkspaceHost::assemble(
            self.workspace_scope,
            Arc::new(config),
            backend_config,
            event_bus,
            Arc::new(reimagine_nodes::BuiltinNodeCatalog::v1()),
            components,
            operation_service,
        ))
    }

    #[cfg(feature = "aigc")]
    async fn build_components(
        &self,
        config: &AppConfig,
        backend_config: &InferenceBackendConfig,
        event_bus: &EventBus,
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
        let acquisition_service = Arc::new(ModelAcquisitionService::with_event_bus(
            config.paths().clone(),
            config,
            event_bus.clone(),
        ));
        let model_service = Arc::new(ModelService::new(
            config.paths().clone(),
            Arc::clone(&acquisition_service),
        ));

        // Bootstrap inference runtime (non-fatal on workspace build)
        let (bootstrapped, aigc_health) = match bootstrap_inference_with_worker_inventory(
            config,
            backend_config,
            Arc::clone(&model_service),
            Arc::clone(&worker_inventory),
        )
        .await
        {
            Ok(bootstrapped) => (bootstrapped, Health::Available),
            Err(error) => {
                let diagnostic = aigc_bootstrap_failed_diagnostic(error.to_string());
                tracing::warn!(
                    reason = %error,
                    "inference bootstrap failed on workspace build; continuing with degraded runtime"
                );
                (
                    BootstrapInference {
                        runtime: ComposedInferenceRuntime::degraded(),
                        compute_profile: WorkspaceComputeProfile::new(),
                        topology: None,
                        discovery: None,
                    },
                    Health::Unavailable(diagnostic),
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
            status: aigc_health,
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
        event_bus: &EventBus,
    ) -> super::WorkspaceComponents {
        let runtime_service = Arc::new(RuntimeService::new(
            reimagine_inference::NodeExecutorRegistry::default(),
            Arc::new(reimagine_inference::CompositeBackendInstanceRuntimeHooks::new(Vec::new())),
            event_bus.clone(),
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
    async fn builder_with_event_bus() {
        let base = temp_dir("sinks");
        let event_bus = EventBus::new();

        let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test"), &base)
            .event_bus(event_bus)
            .build()
            .await
            .expect("builder with event bus");

        assert_eq!(host.base_path(), base);
        let _ = fs::remove_dir_all(&base);
    }
}
