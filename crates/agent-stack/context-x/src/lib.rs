//! reimagine-agent-kernel — ContextBlock conversation kernel (Slice 1, Phase 1-4 scaffold).
//! No dependency on reimagine-core / app-host / Tauri / agent-harness.

#![deny(unsafe_code)]

pub mod block;
pub mod gateway;
pub mod ids;
pub mod runtime;
pub mod tool;
pub mod turn;

pub use block::{BlockPayload, ContextBlock, InputPayload};
pub use gateway::{
    AttemptControl, CallControl, ModelGateway, ModelInvokeError, ModelInvokeErrorKind, RetryPolicy,
    RunControl,
};
pub use ids::{
    AttemptNumber, BlockId, BlockSequence, ContextVersion, FrameId, InvocationId, RoundId, TurnId,
};
pub use runtime::{TurnInterruption, TurnLimits, TurnOutcome, TurnResult, TurnRunner, TurnTrace};
pub use tool::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, IsolationLevel, Tool, ToolCallContext,
    ToolCallId, ToolDefinition, ToolExecutionOutcome, ToolOutput, ToolOutputLimits, ToolOutputMeta,
    ToolResultPayload, ToolResultStatus, Truncation, UnknownOutcomePolicy,
};
pub use turn::{
    Compaction, CompactionInput, CompactionOutput, ContextFrame, FrameScope, ModelContext,
    NoopCompaction, NoopTokenCounter, TokenCounter, TurnContext, WindowBudget,
};
