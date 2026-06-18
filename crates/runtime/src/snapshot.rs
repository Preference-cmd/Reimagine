//! Host-neutral observation shapes: `RunSnapshot` and `RunSummary`.

use std::collections::HashMap;

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::Timestamp;
use reimagine_core::model::{ArtifactId, ArtifactRef, NodeId, RunId, WorkflowId, WorkflowVersion};

use crate::handle::RunState;
use crate::scheduler::NodeState;

/// Host-neutral reference to an artifact produced during a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunArtifactRef {
    pub id: ArtifactId,
    pub node_id: NodeId,
    pub reference: ArtifactRef,
}

impl RunArtifactRef {
    pub fn new(id: ArtifactId, node_id: NodeId, reference: ArtifactRef) -> Self {
        Self {
            id,
            node_id,
            reference,
        }
    }
}

/// Live, host-neutral snapshot of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSnapshot {
    pub run_id: RunId,
    pub workflow_id: WorkflowId,
    pub workflow_version: WorkflowVersion,
    pub state: RunState,
    pub node_states: HashMap<NodeId, NodeState>,
    pub diagnostics: Vec<Diagnostic>,
    pub artifacts: Vec<RunArtifactRef>,
    pub started_at: Timestamp,
    pub updated_at: Timestamp,
}

impl RunSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        state: RunState,
        node_states: HashMap<NodeId, NodeState>,
        diagnostics: Vec<Diagnostic>,
        artifacts: Vec<RunArtifactRef>,
        started_at: Timestamp,
        updated_at: Timestamp,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            workflow_version,
            state,
            node_states,
            diagnostics,
            artifacts,
            started_at,
            updated_at,
        }
    }
}

/// Terminal host-neutral summary of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSummary {
    pub run_id: RunId,
    pub workflow_id: WorkflowId,
    pub workflow_version: WorkflowVersion,
    pub state: RunState,
    pub diagnostics: Vec<Diagnostic>,
    pub artifacts: Vec<RunArtifactRef>,
    pub started_at: Timestamp,
    pub finished_at: Timestamp,
}

impl RunSummary {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        state: RunState,
        diagnostics: Vec<Diagnostic>,
        artifacts: Vec<RunArtifactRef>,
        started_at: Timestamp,
        finished_at: Timestamp,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            workflow_version,
            state,
            diagnostics,
            artifacts,
            started_at,
            finished_at,
        }
    }
}
