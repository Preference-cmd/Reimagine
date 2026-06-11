use std::path::Path;
use std::sync::Arc;

use reimagine_agent::{AgentToolRegistry, WorkspaceScope};
use reimagine_config::{AppConfig, AppPaths};
use reimagine_nodes::BuiltinNodeCatalog;
use reimagine_runtime::{NodeExecutorRegistry, RuntimeService, VecRunEventSink};

use crate::services::WorkspaceServices;
use crate::tools::register_app_tools;
use crate::{AgentService, ModelService, WorkflowService};

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
        let config = AppConfig::new(AppPaths::new(base_path));
        let runtime_service = Arc::new(RuntimeService::with_defaults(
            NodeExecutorRegistry::default(),
            Arc::new(VecRunEventSink::new()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_host_binds_scope_and_services() {
        let base_path = unique_temp_dir("workspace-host");
        let host = WorkspaceHost::with_defaults(WorkspaceScope::new("ws-1"), &base_path);

        assert_eq!(host.workspace_scope().as_str(), "ws-1");
        assert_eq!(host.base_path(), base_path.as_path());
        assert_eq!(
            host.node_catalog().len(),
            reimagine_nodes::all_builtin_defs().len()
        );
        assert_eq!(host.runtime_service().store().active_count(), 0);
        assert!(!host.agent_service().registry().is_empty());
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-{prefix}-{nonce}"))
    }
}
