//! Host-visible run control handle.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_core::task::CancelToken;

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
    cancellation: CancelToken,
    node_cancellations: Arc<Mutex<HashMap<NodeId, CancelToken>>>,
}

impl RunHandle {
    pub fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        cancellation: CancelToken,
        node_cancellations: Arc<Mutex<HashMap<NodeId, CancelToken>>>,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            workflow_version,
            cancellation,
            node_cancellations,
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

    /// Get a cloneable cancellation token. Calling `cancel` on this token
    /// stops downstream scheduling in the runner task.
    pub fn cancellation(&self) -> CancelToken {
        self.cancellation.clone()
    }

    /// Convenience that returns `true` if the run has been asked to stop.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Request cancellation of the entire run.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Request cancellation of a single node.
    pub fn cancel_node(&self, node_id: &NodeId) {
        self.node_cancellation(node_id).cancel();
    }

    /// Get-or-create the node-scoped cancellation token for `node_id`.
    pub fn node_cancellation(&self, node_id: &NodeId) -> CancelToken {
        let mut tokens = self
            .node_cancellations
            .lock()
            .expect("node token map poisoned");
        tokens
            .entry(node_id.clone())
            .or_insert_with(|| self.cancellation.child())
            .clone()
    }

    /// Returns `true` if the node is cancelled — either because the run
    /// was cancelled or because this specific node was cancelled.
    pub fn is_node_cancelled(&self, node_id: &NodeId) -> bool {
        if self.cancellation.is_cancelled() {
            return true;
        }
        self.node_cancellations
            .lock()
            .expect("node token map poisoned")
            .get(node_id)
            .is_some_and(CancelToken::is_cancelled)
    }
}
