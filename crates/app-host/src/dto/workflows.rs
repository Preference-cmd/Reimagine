//! Workflow request/response DTOs.

use reimagine_core::model::{NodeId, WorkflowId, WorkflowVersion};
use reimagine_core::model::RunId;
use reimagine_core::readiness::{RunTarget, RunTargetSelection};
use serde::{Deserialize, Serialize};

use super::runs::{DiagnosticDto, RunSnapshotDto};

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
///
/// The `Started` variant also carries any non-blocking diagnostics
/// from the readiness report (for example catalog/executor orphan
/// warnings). An empty `diagnostics` vector means the run is fully
/// clean.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum RunWorkflowResponse {
    Started {
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        initial_snapshot: RunSnapshotDto,
        diagnostics: Vec<DiagnosticDto>,
    },
    Blocked {
        workflow_id: WorkflowId,
        diagnostics: Vec<DiagnosticDto>,
    },
}
