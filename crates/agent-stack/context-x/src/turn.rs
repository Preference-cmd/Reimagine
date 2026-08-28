//! TurnContext / ContextFrame / WindowBudget / Compaction — Phase 1 placeholder.
use crate::block::{ContextBlock, InputPayload};
use crate::gateway::ModelOutput;
use crate::ids::{BlockId, ContextVersion, RoundId, TurnId};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnLifecycle {
    Open,
    Sealed,
}

#[derive(Debug, Clone, Copy)]
pub struct WindowBudget {
    pub model_window_limit: usize,
    pub compaction_trigger: usize,
}
impl WindowBudget {
    pub fn should_compact(&self, estimated_tokens: usize) -> bool {
        estimated_tokens >= self.compaction_trigger
    }
}
impl Default for WindowBudget {
    fn default() -> Self {
        Self {
            model_window_limit: usize::MAX,
            compaction_trigger: usize::MAX,
        }
    }
}

pub struct CompactionInput {
    pub blocks: Vec<ContextBlock>,
    pub budget: WindowBudget,
    pub estimated_tokens: usize,
}
pub struct CompactionOutput {
    pub blocks: Vec<ContextBlock>,
    pub summary: Option<ContextBlock>,
    pub truncated: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    #[error("compaction failed: {0}")]
    Failed(String),
}

#[async_trait::async_trait]
pub trait Compaction: Send + Sync {
    async fn compact(&self, input: CompactionInput) -> Result<CompactionOutput, CompactionError>;
}

pub struct NoopCompaction;
#[async_trait::async_trait]
impl Compaction for NoopCompaction {
    async fn compact(&self, input: CompactionInput) -> Result<CompactionOutput, CompactionError> {
        Ok(CompactionOutput {
            blocks: input.blocks,
            summary: None,
            truncated: false,
        })
    }
}

#[async_trait::async_trait]
pub trait TokenCounter: Send + Sync {
    fn estimate(&self, blocks: &[ContextBlock]) -> usize;
    fn estimate_value(&self, value: &serde_json::Value) -> usize;
}

pub struct NoopTokenCounter;
#[async_trait::async_trait]
impl TokenCounter for NoopTokenCounter {
    fn estimate(&self, blocks: &[ContextBlock]) -> usize {
        blocks
            .iter()
            .map(|b| {
                serde_json::to_string(&b.payload)
                    .map(|s| s.len() / 4)
                    .unwrap_or(0)
            })
            .sum()
    }
    fn estimate_value(&self, value: &serde_json::Value) -> usize {
        serde_json::to_string(value)
            .map(|s| s.len() / 4)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FrameScope {
    Turn {
        turn_id: TurnId,
        source_version: ContextVersion,
    },
}

#[derive(Debug, Clone)]
pub struct ModelContext {
    pub blocks: Vec<ContextBlock>,
}

#[derive(Debug, Clone)]
pub struct ContextFrame {
    pub scope: FrameScope,
    pub round_id: RoundId,
    pub model_context: ModelContext,
}

#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("sealed")]
    SealedTurn,
    #[error("invalid sequence")]
    InvalidSequence(String),
    #[error("duplicate tool call")]
    DuplicateToolCallId(crate::tool::ToolCallId),
    #[error("unpaired")]
    UnpairedToolResult(crate::tool::ToolCallId),
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("compaction failed: {0}")]
    CompactionFailed(String),
}

#[derive(Debug, Clone)]
pub struct AppliedModelOutput {
    pub block_ids: Vec<BlockId>,
    pub invocation_id: crate::ids::InvocationId,
}

#[derive(Debug)]
pub struct OrderedBlocks(Vec<ContextBlock>);
impl OrderedBlocks {
    pub fn empty() -> Self {
        Self(Vec::new())
    }
}

pub struct TurnContext {
    pub turn_id: TurnId,
    pub blocks: OrderedBlocks,
    pub version: ContextVersion,
    pub lifecycle: TurnLifecycle,
    pub window_budget: WindowBudget,
    pub compaction: Option<Arc<dyn Compaction>>,
    pub token_counter: Option<Arc<dyn TokenCounter>>,
    next_seq: u64,
}
impl std::fmt::Debug for TurnContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnContext")
            .field("turn_id", &self.turn_id)
            .field("version", &self.version)
            .field("lifecycle", &self.lifecycle)
            .field("blocks", &self.blocks)
            .finish()
    }
}

impl TurnContext {
    pub fn new(turn_id: TurnId) -> Self {
        Self {
            turn_id,
            blocks: OrderedBlocks::empty(),
            version: ContextVersion(0),
            lifecycle: TurnLifecycle::Open,
            window_budget: WindowBudget::default(),
            compaction: None,
            token_counter: None,
            next_seq: 0,
        }
    }
    pub fn with_window_budget(mut self, b: WindowBudget) -> Self {
        self.window_budget = b;
        self
    }
    pub fn with_compaction(mut self, c: Arc<dyn Compaction>) -> Self {
        self.compaction = Some(c);
        self
    }
    pub fn with_token_counter(mut self, c: Arc<dyn TokenCounter>) -> Self {
        self.token_counter = Some(c);
        self
    }
    pub fn append_input(&mut self, _payload: InputPayload) -> Result<BlockId, ContextError> {
        Err(ContextError::SealedTurn)
    }
    pub fn apply_inputs(
        &mut self,
        _payloads: Vec<InputPayload>,
    ) -> Result<Vec<BlockId>, ContextError> {
        Ok(vec![])
    }
    pub fn apply_model_output(
        &mut self,
        _invocation: crate::ids::InvocationId,
        _output: ModelOutput,
    ) -> Result<AppliedModelOutput, ContextError> {
        Err(ContextError::SealedTurn)
    }
    pub fn append_tool_results(
        &mut self,
        _results: &[crate::tool::ToolResultPayload],
    ) -> Result<Vec<BlockId>, ContextError> {
        Ok(vec![])
    }
    pub fn estimate_tokens(&self) -> usize {
        self.token_counter
            .as_ref()
            .map(|c| c.estimate(&self.blocks.0))
            .unwrap_or(0)
    }
    pub fn frame(&self, round_id: RoundId) -> Result<ContextFrame, FrameError> {
        Ok(ContextFrame {
            scope: FrameScope::Turn {
                turn_id: self.turn_id.clone(),
                source_version: self.version,
            },
            round_id,
            model_context: ModelContext {
                blocks: self.blocks.0.clone(),
            },
        })
    }
    pub async fn frame_async(&self, round_id: RoundId) -> Result<ContextFrame, FrameError> {
        self.frame(round_id)
    }
    pub fn frame_sync(&self, round_id: RoundId) -> Result<ContextFrame, FrameError> {
        self.frame(round_id)
    }
    pub(crate) fn seal(&mut self) {
        self.lifecycle = TurnLifecycle::Sealed;
    }
    pub fn from_validated_blocks(
        _turn_id: TurnId,
        _blocks: Vec<ContextBlock>,
        _version: ContextVersion,
    ) -> Result<Self, ContextError> {
        Err(ContextError::InvalidSequence(String::new()))
    }
}
