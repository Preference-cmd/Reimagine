use std::path::PathBuf;
use std::sync::Arc;

use reimagine_agent_harness::{AgentEventSink, WorkspaceScope};
use reimagine_backend_worker_host::WorkerStorePaths;
use reimagine_config::{AppConfig, AppPaths, InferenceBackendConfig};
use reimagine_runtime::BoxedRunEventSink;
use reimagine_runtime::VecRunEventSink;

use crate::InstalledWorkerInventoryProvider;
use crate::inference::compose::bootstrap_inference_with_worker_inventory;
use crate::model_acquisition_service::ModelAcquisitionService;
use crate::provider_config::AgentProviderConfigDocument;
use crate::services::WorkspaceServices;
use crate::tools::register_app_tools;
use crate::{
    AgentService, AppHostError, BackendSelection, ModelService, WorkerInventoryProvider,
    WorkflowService,
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
/// # use reimagine_agent_harness::WorkspaceScope;
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
    agent_event_sink: Option<Arc<dyn AgentEventSink>>,
    worker_inventory: Option<Arc<dyn WorkerInventoryProvider>>,
    provider_config: Option<AgentProviderConfigDocument>,
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
            agent_event_sink: None,
            worker_inventory: None,
            provider_config: None,
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

    /// Set the agent event sink for agent loop events.
    ///
    /// When not set, uses a default sink that discards events.
    pub fn agent_event_sink(mut self, sink: Arc<dyn AgentEventSink>) -> Self {
        self.agent_event_sink = Some(sink);
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

    /// Set the provider config document used to construct and register
    /// concrete `AgentProvider` adapters.
    ///
    /// When not set, the builder loads `{config_dir}/agent-providers.json`
    /// through the config store (a missing file yields an empty document,
    /// so no providers are registered).
    pub fn provider_config(mut self, document: AgentProviderConfigDocument) -> Self {
        self.provider_config = Some(document);
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

        // Resolve event sinks
        let event_sink = self
            .event_sink
            .unwrap_or_else(|| Arc::new(VecRunEventSink::new()));
        let agent_event_sink = self
            .agent_event_sink
            .unwrap_or_else(|| Arc::new(reimagine_agent_harness::VecAgentEventSink::new()));

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
        let workflow_service = Arc::new(WorkflowService::new(config.paths().clone()));
        let services = Arc::new(WorkspaceServices::new(
            self.workspace_scope.clone(),
            Arc::new(config.clone()),
            Arc::clone(&workflow_service),
            Arc::clone(&model_service),
            acquisition_service,
            Arc::clone(&runtime_service),
            Arc::clone(&node_catalog),
        ));

        // Create agent service: first create a temporary one to get default registry/providers,
        // then recreate with the injected event sink
        let mut registry = reimagine_agent_harness::AgentToolRegistry::new();
        register_app_tools(&mut registry, Arc::clone(&services));
        let registry = Arc::new(registry);
        let temp_agent_service = Arc::new(AgentService::with_registry(
            self.workspace_scope.clone(),
            Arc::clone(&registry),
        ));
        let providers = temp_agent_service.providers().clone();

        // Register concrete providers from the config document. When no
        // explicit document was injected, load it from the config store;
        // a missing file yields an empty document (no providers).
        let provider_document = match self.provider_config {
            Some(document) => document,
            None => {
                let handle = config.config::<AgentProviderConfigDocument>()?;
                let (document, _report) = handle.load().await?;
                document
            }
        };
        let (registered, errors) = crate::register_providers_from_document(
            &providers,
            &provider_document,
            Some(&self.base_path),
        );
        if !registered.is_empty() {
            tracing::info!(
                providers = ?registered.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
                "registered agent providers"
            );
        }
        for error in &errors {
            tracing::warn!("{error}");
        }

        let agent_service = Arc::new(AgentService::with_registry_providers_and_sink(
            self.workspace_scope.clone(),
            registry,
            providers,
            agent_event_sink.clone(),
        ));

        // Assemble workspace
        Ok(WorkspaceHost {
            workspace_scope: self.workspace_scope,
            config: Arc::new(config),
            backend_config,
            workflow_service,
            model_service,
            runtime_service,
            agent_service,
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
            agent_event_sink,
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
    async fn builder_with_event_sinks() {
        let base = temp_dir("sinks");
        let event_sink: BoxedRunEventSink = Arc::new(VecRunEventSink::new());
        let agent_sink: Arc<dyn AgentEventSink> =
            Arc::new(reimagine_agent_harness::VecAgentEventSink::new());

        let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test"), &base)
            .event_sink(event_sink)
            .agent_event_sink(agent_sink)
            .build()
            .await
            .expect("builder with event sinks");

        assert_eq!(host.base_path(), base);
        let _ = fs::remove_dir_all(&base);
    }
}
