//! Host-neutral observation shapes: `RunSnapshot` and `RunSummary`.

use std::collections::HashMap;

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::Timestamp;
use reimagine_core::model::{ArtifactId, ArtifactRef, NodeId, RunId, WorkflowId, WorkflowVersion};

use crate::handle::RunState;
use crate::scheduler::NodeState;

/// Incremental update to a [`RunSnapshot`], produced by [`crate::store::RunStore`]
/// on every store mutation.
///
/// Hosts subscribe through `RunStore::delta_stream` / `RunStore::subscribe`:
/// - [`RunSnapshotUpdate::Full`] is emitted at subscribe time as a baseline
///   for late joiners.
/// - [`RunSnapshotUpdate::Delta`] carries only the node states and artifacts
///   that changed since the previous snapshot, so hosts can merge into a
///   local cache without receiving the full snapshot on every transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunSnapshotUpdate {
    /// Nodes that changed and artifacts that appeared since the previous
    /// snapshot. Run-level state changes (e.g. terminal transitions) are not
    /// part of the delta; they are observable through the full snapshots on
    /// the watch channel.
    Delta {
        run_id: RunId,
        /// Only nodes whose state changed (or that appeared) since the last
        /// snapshot. Removed nodes are not representable and are omitted.
        changed_nodes: HashMap<NodeId, NodeState>,
        /// Artifacts that appeared since the last snapshot, in creation order.
        new_artifacts: Vec<RunArtifactRef>,
        timestamp: Timestamp,
    },
    /// Full snapshot, used as the subscribe-time baseline for delta streams.
    Full(RunSnapshot),
}

impl RunSnapshotUpdate {
    /// Run this update belongs to.
    pub fn run_id(&self) -> &RunId {
        match self {
            Self::Delta { run_id, .. } => run_id,
            Self::Full(snapshot) => &snapshot.run_id,
        }
    }

    /// Timestamp of the update.
    pub fn timestamp(&self) -> &Timestamp {
        match self {
            Self::Delta { timestamp, .. } => timestamp,
            Self::Full(snapshot) => &snapshot.updated_at,
        }
    }
}

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
