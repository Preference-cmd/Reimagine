//! Host-visible run control handle.

use reimagine_core::model::{RunId, WorkflowId, WorkflowVersion};

use crate::cancellation::CancellationToken;

/// High-level state of a run, observable to the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RunState {
    /// Run has been registered in the store and the background task spawned,
    /// but the runner has not yet started processing stages.
    Queued,
    /// Runner is actively processing stages.
    Running,
    /// Run finished successfully.
    Completed,
    /// Run failed; a `RunFailed` event carries the diagnostic.
    Failed,
    /// Run was cancelled by a host request.
    Cancelled,
}

impl RunState {
    /// Returns `true` once the run has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Host-visible run control handle.
///
/// A `RunHandle` is a control surface, **not** the canonical run state. Hosts
/// query [`crate::RuntimeService::snapshot`] and
/// [`crate::RuntimeService::summary`] for the actual run state.
#[derive(Debug, Clone)]
pub struct RunHandle {
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    cancellation: CancellationToken,
}

impl RunHandle {
    pub fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            workflow_version,
            cancellation,
        }
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn workflow_version(&self) -> WorkflowVersion {
        self.workflow_version
    }

    /// Get a clonable cancellation token. Calling `cancel` on this token
    /// stops downstream scheduling in the runner task.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    /// Convenience that returns `true` if the run has been asked to stop.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}
