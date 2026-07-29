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

    /// Shut down the active inference worker, if any.
    ///
    /// This is the application-level shutdown hook. Call it when the host
    /// is exiting so that child worker processes are cleaned up instead of
    /// becoming orphans.
    pub async fn shutdown(&self) {
        self.workspace.shutdown().await;
    }
}
