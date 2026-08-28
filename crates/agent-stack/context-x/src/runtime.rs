//! TurnRunner / TurnOutcome / TurnTrace — Phase 3 placeholder.
use crate::gateway::{ModelInvokeErrorKind, ModelOutput};
use crate::ids::RoundId;
use crate::tool::{ArtifactRef, ToolCallId, ToolResultStatus, Truncation};
use crate::turn::TurnContext;

#[derive(Debug, Clone)]
pub struct TurnLimits {
    pub max_model_rounds: u32,
    pub max_tool_calls: u32,
}
impl Default for TurnLimits {
    fn default() -> Self {
        Self {
            max_model_rounds: 10,
            max_tool_calls: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "detail")]
pub enum TurnInterruption {
    ExplicitCancellation,
    TurnDeadlineExceeded,
    RetryExhausted {
        last_kind: ModelInvokeErrorKind,
        last_error: String,
    },
    InvalidModelOutput {
        reason: String,
    },
    MaxModelRounds {
        limit: u32,
    },
    MaxToolCalls {
        limit: u32,
    },
    UnsafeUnknownOutcome {
        call_id: ToolCallId,
    },
    CompactionFailed {
        reason: String,
    },
    RunnerInvariantViolation {
        reason: String,
    },
    ModelMaxTokens,
    ModelRefusal,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttemptTrace {
    pub attempt: crate::ids::AttemptNumber,
    pub kind: Option<ModelInvokeErrorKind>,
    pub is_retryable: bool,
    pub duration_ms: u64,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputSummary {
    pub stop_reason: crate::gateway::ModelStopReason,
    pub usage: Option<crate::gateway::ModelUsage>,
    pub tool_call_count: usize,
    pub assistant_text_bytes: usize,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallTrace {
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub position: usize,
    pub status: ToolResultStatus,
    pub truncation: Truncation,
    pub artifact: Option<ArtifactRef>,
    pub duration_ms: u64,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolBatchTrace {
    pub calls: Vec<ToolCallTrace>,
    pub completion_order: Vec<ToolCallId>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelRoundTrace {
    pub round_id: RoundId,
    pub invocation_id: crate::ids::InvocationId,
    pub frame_version: crate::ids::ContextVersion,
    pub attempts: Vec<AttemptTrace>,
    pub output_summary: Option<OutputSummary>,
    pub applied_block_ids: Vec<crate::ids::BlockId>,
    pub tool_batch: Option<ToolBatchTrace>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurnTrace {
    pub rounds: Vec<ModelRoundTrace>,
    pub tool_calls_total: usize,
    pub total_duration_ms: u64,
}
impl TurnTrace {
    pub fn new() -> Self {
        Self {
            rounds: vec![],
            tool_calls_total: 0,
            total_duration_ms: 0,
        }
    }
}

#[derive(Debug)]
pub enum TurnResult {
    Completed { final_output: ModelOutput },
    Interrupted { cause: TurnInterruption },
}

#[derive(Debug)]
pub struct TurnOutcome {
    pub context: TurnContext,
    pub result: TurnResult,
    pub trace: TurnTrace,
}

pub struct TurnRunner;
impl TurnRunner {
    pub async fn run(
        &self,
        context: TurnContext,
        _config: crate::gateway::TurnRunConfig,
        _control: crate::gateway::RunControl,
    ) -> TurnOutcome {
        TurnOutcome {
            context,
            result: TurnResult::Interrupted {
                cause: TurnInterruption::ExplicitCancellation,
            },
            trace: TurnTrace::new(),
        }
    }
}
