use std::sync::Arc;

use crate::WorkspaceHost;

#[derive(Debug, Clone)]
pub struct AppHost {
    workspace: Arc<WorkspaceHost>,
}

impl AppHost {
    pub fn new(workspace: WorkspaceHost) -> Self {
        Self {
            workspace: Arc::new(workspace),
        }
    }

    pub fn from_workspace(workspace: Arc<WorkspaceHost>) -> Self {
        Self { workspace }
    }

    pub fn workspace(&self) -> &Arc<WorkspaceHost> {
        &self.workspace
    }
}
