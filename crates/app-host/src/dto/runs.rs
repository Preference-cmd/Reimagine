//! Run observation DTOs (snapshots, summaries, events).

use std::collections::HashMap;

use reimagine_core::model::{DocumentVersion, NodeId, RunId, WorkflowId};
use reimagine_core::task::TaskState;
use reimagine_runtime::NodeState;
use reimagine_runtime::{RunSnapshot, RunSummary};
use serde::{Deserialize, Serialize};

use super::artifacts::ArtifactDto;

/// `GET /runs/:id` response. V1 returns a JSON projection of the
/// host-neutral [`RunSnapshot`] (or [`RunSummary`] once the run has
/// reached a terminal state).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunDto {
    Snapshot(RunSnapshotDto),
    Summary(RunSummaryDto),
}

/// Host-neutral run snapshot in JSON-friendly form. We do not expose
/// runtime value stores or backend tensor handles here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSnapshotDto {
    pub run_id: RunId,
    pub workflow_id: WorkflowId,
    pub workflow_version: DocumentVersion,
    pub state: TaskState,
    pub node_states: HashMap<NodeId, NodeStateDto>,
    pub diagnostics: Vec<DiagnosticDto>,
    pub artifacts: Vec<ArtifactDto>,
    pub started_at: String,
    pub updated_at: String,
}

impl From<RunSnapshot> for RunSnapshotDto {
    fn from(value: RunSnapshot) -> Self {
        Self {
            run_id: value.run_id,
            workflow_id: value.workflow_id,
            workflow_version: value.workflow_version,
            state: value.state,
            node_states: value
                .node_states
                .into_iter()
                .map(|(node_id, state)| (node_id, state.into()))
                .collect(),
            diagnostics: value.diagnostics.into_iter().map(Into::into).collect(),
            artifacts: value.artifacts.into_iter().map(Into::into).collect(),
            started_at: value.started_at.as_str().to_string(),
            updated_at: value.updated_at.as_str().to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeStateDto {
    Queued,
    Running,
    Completed,
    Failed,
    Skipped,
    Cancelled,
}

impl From<NodeState> for NodeStateDto {
    fn from(value: NodeState) -> Self {
        match value {
            NodeState::Queued => Self::Queued,
            NodeState::Running => Self::Running,
            NodeState::Completed => Self::Completed,
            NodeState::Failed => Self::Failed,
            NodeState::Skipped => Self::Skipped,
            NodeState::Cancelled => Self::Cancelled,
        }
    }
}

/// Terminal run summary in JSON-friendly form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummaryDto {
    pub run_id: RunId,
    pub workflow_id: WorkflowId,
    pub workflow_version: DocumentVersion,
    pub state: TaskState,
    pub diagnostics: Vec<DiagnosticDto>,
    pub artifacts: Vec<ArtifactDto>,
    pub started_at: String,
    pub finished_at: String,
}

impl From<RunSummary> for RunSummaryDto {
    fn from(value: RunSummary) -> Self {
        Self {
            run_id: value.run_id,
            workflow_id: value.workflow_id,
            workflow_version: value.workflow_version,
            state: value.state,
            diagnostics: value.diagnostics.into_iter().map(Into::into).collect(),
            artifacts: value.artifacts.into_iter().map(Into::into).collect(),
            started_at: value.started_at.as_str().to_string(),
            finished_at: value.finished_at.as_str().to_string(),
        }
    }
}

pub use reimagine_core::diagnostic::DiagnosticDto;
