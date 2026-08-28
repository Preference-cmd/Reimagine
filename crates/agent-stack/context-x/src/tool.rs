use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::gateway::CallControl;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    Task,
    Subprocess,
}
impl Default for IsolationLevel {
    fn default() -> Self {
        Self::Task
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownOutcomePolicy {
    Stop,
    Continue,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCallId(pub String);
impl ToolCallId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn generate(tool_name: &str, arguments: &serde_json::Value, position: usize) -> Self {
        let json = serde_json::to_string(arguments).unwrap_or_default();
        let hash = blake3::hash(json.as_bytes());
        let hex = hash.to_hex();
        Self(format!("{}:{}:{}", tool_name, &hex[..8], position))
    }
    pub fn position(&self) -> usize {
        self.0
            .rsplit(':')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallContext {
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResultStatus {
    Succeeded,
    Failed,
    Rejected,
    Cancelled,
    TimedOut,
    UnknownOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutputMeta {
    pub duration_ms: Option<u64>,
    pub original_tokens: Option<usize>,
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    None,
    Middle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: serde_json::Value,
    pub truncation: Truncation,
    pub meta: Option<ToolOutputMeta>,
    pub artifact: Option<ArtifactRef>,
}
impl ToolOutput {
    pub fn is_truncated(&self) -> bool {
        !matches!(self.truncation, Truncation::None)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultPayload {
    pub call_id: ToolCallId,
    pub status: ToolResultStatus,
    pub output: ToolOutput,
}

#[derive(Debug, Clone)]
pub struct ToolExecutionOutcome {
    pub result: ToolResultPayload,
    pub policy: UnknownOutcomePolicy,
}
impl ToolExecutionOutcome {
    pub fn new(result: ToolResultPayload) -> Self {
        Self {
            result,
            policy: UnknownOutcomePolicy::Stop,
        }
    }
    pub fn with_policy(mut self, policy: UnknownOutcomePolicy) -> Self {
        self.policy = policy;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: String,
    pub size_bytes: usize,
    pub kind: ArtifactKind,
    pub persisted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactKind {
    FullOutput,
    PipeCache,
    Binary,
}

pub struct ArtifactHint {
    pub tool_name: String,
    pub call_id: ToolCallId,
    pub kind: ArtifactKind,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("persist failed: {0}")]
    Persist(String),
    #[error("read failed: {0}")]
    Read(String),
}

#[async_trait]
pub trait ArtifactStore: Send + Sync {
    async fn persist(&self, data: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError>;
    async fn read(
        &self,
        id: &str,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError>;
}

#[derive(Debug, Clone)]
pub struct ToolOutputLimits {
    pub max_tokens: usize,
}
impl Default for ToolOutputLimits {
    fn default() -> Self {
        Self { max_tokens: 7_500 }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn output_limits(&self) -> Option<ToolOutputLimits> {
        None
    }
    fn isolation_level(&self) -> IsolationLevel {
        IsolationLevel::Task
    }
    fn unknown_outcome_policy(&self) -> UnknownOutcomePolicy {
        UnknownOutcomePolicy::Stop
    }
    async fn execute(&self, ctx: &ToolCallContext, control: &CallControl) -> ToolExecutionOutcome;
    async fn execute_with_store(
        &self,
        ctx: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolExecutionOutcome {
        let _ = store;
        self.execute(ctx, control).await
    }
}
