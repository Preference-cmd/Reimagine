//! HTTP DTOs (request/response payloads) for the Axum host.
//!
//! These types are the V1 wire contract. The shapes are intentionally
//! minimal — they wrap host-neutral types from `reimagine-core` and
//! `reimagine-runtime` so that the JSON surface stays stable even as
//! the underlying types evolve.
//!
//! Two principles govern this module:
//!
//! 1. DTOs do not contain runtime value stores or backend tensor
//!    handles. The host-neutral [`reimagine_runtime::RunSnapshot`] and
//!    [`reimagine_runtime::RunSummary`] already enforce that boundary;
//!    we just pass them through as JSON.
//! 2. The shapes are flat enough to drive curl-based smoke tests and
//!    rich enough to drive end-to-end automation. We deliberately avoid
//!    re-serializing the underlying structs: any future host-side
//!    transformation belongs here, with a test.

use std::collections::HashMap;

use reimagine_core::ExecutionValue;
use reimagine_core::event::RunEvent;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_core::readiness::{RunTarget, RunTargetSelection};
use reimagine_runtime::NodeState;
use reimagine_runtime::RunState;
use reimagine_runtime::{RunArtifactRef, RunSnapshot, RunSummary};
use serde::{Deserialize, Serialize};

/// V1 `GET /health` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub workspace: String,
}

impl HealthResponse {
    pub fn ok(workspace_id: &str) -> Self {
        Self {
            status: "ok",
            workspace: workspace_id.to_string(),
        }
    }
}

/// `POST /workflows/open` request body.
///
/// V1 accepts either a raw workflow JSON document (`workflow`) or the
/// id of a workflow that the host should look up on disk. The two
/// fields are mutually exclusive; clients that need round-trip
/// persistence should use `id`; clients that want to drive an in-memory
/// test should use `workflow`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenWorkflowRequest {
    /// Optional workflow id. When present, the host loads the JSON from
    /// the workspace's `workflows_dir` and registers it.
    #[serde(default)]
    pub id: Option<WorkflowId>,
    /// Optional inline workflow JSON. When present, the host parses
    /// and registers the workflow under the id declared inside.
    #[serde(default)]
    pub workflow: Option<serde_json::Value>,
}

impl Default for OpenWorkflowRequest {
    fn default() -> Self {
        Self {
            id: None,
            workflow: None,
        }
    }
}

/// `POST /workflows/open` response. The resolved `workflow_id` is the
/// id the host actually registered, which is the same id subsequent
/// routes should reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenWorkflowResponse {
    pub workflow_id: WorkflowId,
    pub source: WorkflowSource,
}

/// How the workflow was opened.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowSource {
    /// Loaded from the workspace's `workflows_dir`.
    Disk,
    /// Provided inline in the open request.
    Inline,
    /// Already registered with the workspace (idempotent open).
    Existing,
}

/// `POST /workflows/:id/run` request body.
///
/// V1 accepts an optional target selection. When `target_selection` is
/// omitted the host uses the `AllDefaultTargets` variant. The host
/// does not currently accept arbitrary `RunInputs` over HTTP — they
/// are host-supplied (e.g. via a follow-up Agent call) and V1 runs
/// use the defaults the runtime already understands.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunWorkflowRequestDto {
    #[serde(default)]
    pub target_selection: Option<TargetSelectionDto>,
    #[serde(default)]
    pub correlation_id: Option<String>,
}

/// Wire form of [`RunTargetSelection`].
///
/// We only serialize the two `RunTarget` variants that V1 actually
/// supports: explicit node targets and explicit node-output targets.
/// Workflow-output targets and the `AllDefaultTargets` shorthand are
/// represented as their concrete `RunTarget` lists.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TargetSelectionDto {
    /// Empty list, equivalent to core's `AllDefaultTargets`.
    AllDefault,
    /// Explicit list of `RunTarget` values.
    Explicit { targets: Vec<RunTargetDto> },
}

impl Default for TargetSelectionDto {
    fn default() -> Self {
        Self::AllDefault
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunTargetDto {
    Node { node_id: NodeId },
    NodeOutput { node_id: NodeId, slot_id: String },
    WorkflowOutput { output_id: String },
}

impl From<RunTarget> for RunTargetDto {
    fn from(value: RunTarget) -> Self {
        match value {
            RunTarget::Node { node_id } => Self::Node { node_id },
            RunTarget::NodeOutput { node_id, slot_id } => Self::NodeOutput {
                node_id,
                slot_id: slot_id.to_string(),
            },
            RunTarget::WorkflowOutput { output_id } => Self::WorkflowOutput {
                output_id: output_id.to_string(),
            },
        }
    }
}

impl From<RunTargetDto> for RunTarget {
    fn from(value: RunTargetDto) -> Self {
        match value {
            RunTargetDto::Node { node_id } => Self::Node { node_id },
            RunTargetDto::NodeOutput { node_id, slot_id } => Self::NodeOutput {
                node_id,
                slot_id: slot_id.into(),
            },
            RunTargetDto::WorkflowOutput { output_id } => Self::WorkflowOutput {
                output_id: output_id.into(),
            },
        }
    }
}

impl From<TargetSelectionDto> for RunTargetSelection {
    fn from(value: TargetSelectionDto) -> Self {
        match value {
            TargetSelectionDto::AllDefault => Self::AllDefaultTargets,
            TargetSelectionDto::Explicit { targets } => {
                Self::ExplicitTargets(targets.into_iter().map(RunTarget::from).collect())
            }
        }
    }
}

/// `POST /workflows/:id/run` response. V1 always returns the host
/// outcome: either a started run handle, a blocked readiness report,
/// or an error. The `Started` variant carries the initial snapshot so
/// clients can poll immediately without a follow-up `GET /runs/:id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum RunWorkflowResponse {
    Started {
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        initial_snapshot: RunSnapshotDto,
    },
    Blocked {
        workflow_id: WorkflowId,
        diagnostics: Vec<DiagnosticDto>,
    },
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactDto {
    pub id: reimagine_core::model::ArtifactId,
    pub node_id: NodeId,
    pub reference: reimagine_core::model::ArtifactRef,
}

impl From<RunArtifactRef> for ArtifactDto {
    fn from(value: RunArtifactRef) -> Self {
        Self {
            id: value.id,
            node_id: value.node_id,
            reference: value.reference,
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
    pub artifact: Option<reimagine_core::model::ArtifactId>,
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

/// Marker that the type is host-safe: it never carries
/// [`ExecutionValue`]-shaped payloads.
#[allow(dead_code)]
const fn _assert_no_runtime_values() {
    // The DTOs above are JSON-only; if a future change accidentally
    // re-introduces a runtime value handle, the API surface breaks.
    // The constant below documents the invariant and gives the next
    // reviewer a hint where to look.
    let _ = std::mem::size_of::<ExecutionValue>();
}
