//! Per-run mutable session. Internal to the runner task; never exposed to
//! the host.

use std::collections::HashMap;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

use crate::cancellation::CancellationToken;
use crate::value_store::RunValueStore;

/// Internal runner-task session. Tracks the intermediate value store, the
/// cancellation token, and the per-node outcome map.
///
/// Hosts never see this. The runner task owns the only session.
pub struct RunSession {
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    values: RunValueStore,
    cancellation: CancellationToken,
    node_outcomes: HashMap<NodeId, NodeOutcome>,
}

impl RunSession {
    pub fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        correlation_id: Option<CorrelationId>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            workflow_version,
            correlation_id,
            values: RunValueStore::new(),
            cancellation,
            node_outcomes: HashMap::new(),
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

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn values(&self) -> &RunValueStore {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut RunValueStore {
        &mut self.values
    }

    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    pub fn record_outcome(&mut self, node_id: NodeId, outcome: NodeOutcome) {
        self.node_outcomes.insert(node_id, outcome);
    }

    pub fn node_outcome(&self, node_id: &NodeId) -> Option<&NodeOutcome> {
        self.node_outcomes.get(node_id)
    }

    pub fn node_outcomes(&self) -> &HashMap<NodeId, NodeOutcome> {
        &self.node_outcomes
    }

    /// Mutable access to the per-node outcome map. Only the runner task uses
    /// this; hosts never see it.
    pub fn node_outcomes_mut(&mut self) -> &mut HashMap<NodeId, NodeOutcome> {
        &mut self.node_outcomes
    }
}

/// Outcome of running a single node, recorded by the runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeOutcome {
    Queued,
    Running,
    Completed,
    Failed { message: String },
    Skipped { reason: String },
    Cancelled,
}

impl NodeOutcome {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed { .. } | Self::Skipped { .. } | Self::Cancelled
        )
    }
}
