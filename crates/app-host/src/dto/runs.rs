//! Run observation DTOs (snapshots, summaries, events).

use std::collections::HashMap;

use reimagine_core::event::RunEvent;
use reimagine_core::model::{ArtifactId, NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_runtime::{NodeState, RunState};
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
    pub workflow_version: WorkflowVersion,
    pub state: RunState,
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
    pub workflow_version: WorkflowVersion,
    pub state: RunState,
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

/// Diagnostic JSON projection. We pass the host-neutral diagnostic
/// through `serde` as a tagged shape; the underlying diagnostic type
/// already implements `Serialize`/`Deserialize`, but we wrap it here
/// so future fields can be added at the HTTP layer without changing
/// the core schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticDto {
    pub id: String,
    pub code: String,
    pub severity: String,
    pub source: String,
    pub message: String,
    pub target: String,
}

impl From<reimagine_core::diagnostic::Diagnostic> for DiagnosticDto {
    fn from(value: reimagine_core::diagnostic::Diagnostic) -> Self {
        let target = value.primary();
        Self {
            id: value.id().as_str().to_string(),
            code: value.code().as_str().to_string(),
            severity: value.severity().to_string(),
            source: value.source().to_string(),
            message: value.message().to_string(),
            target: format!("{}:{}", target.domain(), target.path().unwrap_or("")),
        }
    }
}

/// `GET /runs/:id/events` response. V1 returns the full event list
/// for the run. SSE/WebSocket streaming is a later refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEventsResponse {
    pub run_id: RunId,
    pub events: Vec<RunEventDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEventDto {
    pub id: String,
    pub run_id: RunId,
    pub workflow_id: WorkflowId,
    pub workflow_version: WorkflowVersion,
    pub kind: String,
    pub node_id: Option<NodeId>,
    pub artifact: Option<ArtifactId>,
    pub diagnostics: Vec<DiagnosticDto>,
    pub created_at: String,
    pub correlation_id: Option<String>,
}

impl From<RunEvent> for RunEventDto {
    fn from(value: RunEvent) -> Self {
        let correlation_id = value.correlation_id().map(|id| id.as_str().to_string());
        let node_id = value.node_id().cloned();
        let artifact = value.artifact().cloned();
        let diagnostics = value
            .diagnostics()
            .iter()
            .map(|d| d.clone().into())
            .collect();
        Self {
            id: value.id().as_str().to_string(),
            run_id: value.run_id().clone(),
            workflow_id: value.workflow_id().clone(),
            workflow_version: value.workflow_version(),
            kind: format!("{:?}", value.kind()),
            node_id,
            artifact,
            diagnostics,
            created_at: value.created_at().as_str().to_string(),
            correlation_id,
        }
    }
}
