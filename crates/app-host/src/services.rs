use std::sync::Arc;

use reimagine_agent::WorkspaceScope;
use reimagine_config::AppConfig;
use reimagine_runtime::RuntimeService;

use crate::model_acquisition_service::ModelAcquisitionService;
use crate::node_catalog::NodeCatalogService;
use crate::{ModelService, WorkflowService};

/// Service container captured by app-host agent tools.
///
/// Keeps `Arc` handles to all workspace-scoped services so tools can
/// hold a single `Arc<WorkspaceServices>` rather than individual
/// service arcs.
#[derive(Debug, Clone)]
pub struct WorkspaceServices {
    workspace_scope: WorkspaceScope,
    config: Arc<AppConfig>,
    workflow_service: Arc<WorkflowService>,
    model_service: Arc<ModelService>,
    model_acquisition_service: Arc<ModelAcquisitionService>,
    runtime_service: Arc<RuntimeService>,
    node_catalog: Arc<NodeCatalogService>,
}

impl WorkspaceServices {
    pub fn new(
        workspace_scope: WorkspaceScope,
        config: Arc<AppConfig>,
        workflow_service: Arc<WorkflowService>,
        model_service: Arc<ModelService>,
        model_acquisition_service: Arc<ModelAcquisitionService>,
        runtime_service: Arc<RuntimeService>,
        node_catalog: Arc<NodeCatalogService>,
    ) -> Self {
        Self {
            workspace_scope,
            config,
            workflow_service,
            model_service,
            model_acquisition_service,
            runtime_service,
            node_catalog,
        }
    }

    pub fn workspace_scope(&self) -> &WorkspaceScope {
        &self.workspace_scope
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

    pub fn model_acquisition_service(&self) -> &Arc<ModelAcquisitionService> {
        &self.model_acquisition_service
    }

    pub fn runtime_service(&self) -> &Arc<RuntimeService> {
        &self.runtime_service
    }

    pub fn node_catalog(&self) -> &Arc<NodeCatalogService> {
        &self.node_catalog
    }
}
