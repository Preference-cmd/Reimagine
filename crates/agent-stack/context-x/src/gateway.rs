use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use crate::block::TextPayload;
use crate::ids::{AttemptNumber, InvocationId};
use crate::tool::ToolDefinition;
use crate::turn::ContextFrame;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef(pub String);
impl ModelRef {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolSurface {
    pub definitions: Vec<ToolDefinition>,
}
impl ToolSurface {
    pub fn empty() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }
    pub fn from_definitions(definitions: Vec<ToolDefinition>) -> Self {
        Self { definitions }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDraft {
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Refusal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantPayload {
    pub text: TextPayload,
    pub tool_calls: Vec<ToolCallDraft>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    pub assistant: AssistantPayload,
    pub usage: Option<ModelUsage>,
    pub stop_reason: ModelStopReason,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelInvokeErrorKind {
    Transient,
    TimedOut,
    Cancelled,
    Permanent,
    InvalidRequest,
    UnknownOutcome,
}

#[derive(Debug)]
pub struct ModelInvokeError {
    pub kind: ModelInvokeErrorKind,
    pub message: String,
}
impl ModelInvokeError {
    pub fn new(kind: ModelInvokeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub fn kind(&self) -> &ModelInvokeErrorKind {
        &self.kind
    }
    pub fn is_retryable(&self, policy: &RetryPolicy) -> bool {
        match self.kind {
            ModelInvokeErrorKind::Transient => true,
            ModelInvokeErrorKind::TimedOut => policy.retry_timeouts,
            _ => false,
        }
    }
}
impl std::fmt::Display for ModelInvokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for ModelInvokeError {}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub retry_timeouts: bool,
}
impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 0,
            retry_timeouts: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunControl {
    cancellation: CancellationToken,
    turn_deadline: Option<Instant>,
}
impl RunControl {
    pub fn new(cancellation: CancellationToken, turn_deadline: Option<Instant>) -> Self {
        Self {
            cancellation,
            turn_deadline,
        }
    }
    pub fn with_deadline(cancellation: CancellationToken, deadline: Instant) -> Self {
        Self {
            cancellation,
            turn_deadline: Some(deadline),
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    pub fn turn_deadline(&self) -> Option<Instant> {
        self.turn_deadline
    }
    pub fn is_deadline_exceeded(&self) -> bool {
        self.turn_deadline
            .map(|d| Instant::now() >= d)
            .unwrap_or(false)
    }
    pub fn should_stop(&self) -> bool {
        self.is_cancelled() || self.is_deadline_exceeded()
    }
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
    pub fn for_attempt(&self, attempt_timeout: Option<Duration>) -> AttemptControl {
        let deadline = Self::effective_deadline(self.turn_deadline, attempt_timeout);
        AttemptControl {
            cancellation: self.cancellation.clone(),
            deadline,
        }
    }
    fn effective_deadline(parent: Option<Instant>, timeout: Option<Duration>) -> Option<Instant> {
        let from_timeout = timeout.map(|t| Instant::now() + t);
        match (parent, from_timeout) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttemptControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}
impl AttemptControl {
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
    pub fn is_deadline_exceeded(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
    pub fn for_call(&self, call_timeout: Option<Duration>) -> CallControl {
        let deadline = RunControl::effective_deadline(self.deadline, call_timeout);
        CallControl {
            cancellation: self.cancellation.clone(),
            deadline,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CallControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}
#[derive(Debug, Clone, thiserror::Error)]
pub enum ControlError {
    #[error("cancelled")]
    Cancelled,
    #[error("deadline exceeded")]
    TimedOut,
}
impl CallControl {
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
    pub fn is_deadline_exceeded(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }
    pub fn check(&self) -> Result<(), ControlError> {
        if self.is_cancelled() {
            return Err(ControlError::Cancelled);
        }
        if self.is_deadline_exceeded() {
            return Err(ControlError::TimedOut);
        }
        Ok(())
    }
    pub async fn check_cancelled(&self) -> Result<(), ControlError> {
        tokio::select! {
            _ = self.cancellation.cancelled() => Err(ControlError::Cancelled),
            _ = async {
                if let Some(d) = self.deadline { tokio::time::sleep_until(d.into()).await; } else { std::future::pending::<()>().await; }
            } => Err(ControlError::TimedOut),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub invocation_id: InvocationId,
    pub attempt: AttemptNumber,
    pub frame: ContextFrame,
    pub model: ModelRef,
    pub tool_surface: ToolSurface,
    pub generation: GenerationOptions,
}

#[async_trait]
pub trait ModelGateway: Send + Sync {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError>;
}

pub struct FakeGateway {
    pub outputs: std::sync::Mutex<Vec<Result<ModelOutput, ModelInvokeErrorKind>>>,
}
impl FakeGateway {
    pub fn new(outputs: Vec<Result<ModelOutput, ModelInvokeErrorKind>>) -> Self {
        Self {
            outputs: std::sync::Mutex::new(outputs),
        }
    }
}
#[async_trait]
impl ModelGateway for FakeGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut guard = self.outputs.lock().unwrap();
        if guard.is_empty() {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                "no more fake outputs",
            ));
        }
        match guard.remove(0) {
            Ok(o) => Ok(o),
            Err(k) => Err(ModelInvokeError::new(k, "fake error")),
        }
    }
}

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

pub struct TurnRunConfig {
    pub model: ModelRef,
    pub tool_surface: ToolSurface,
    pub generation: GenerationOptions,
    pub retry: RetryPolicy,
    pub limits: TurnLimits,
    pub tool_output_limits: crate::tool::ToolOutputLimits,
    pub artifact_store: Option<Arc<dyn crate::tool::ArtifactStore>>,
    pub window_budget: crate::turn::WindowBudget,
    pub compaction: Option<Arc<dyn crate::turn::Compaction>>,
    pub token_counter: Option<Arc<dyn crate::turn::TokenCounter>>,
    pub call_timeout: Option<Duration>,
    pub attempt_timeout: Option<Duration>,
}
impl Default for TurnRunConfig {
    fn default() -> Self {
        Self {
            model: ModelRef::new("fake"),
            tool_surface: ToolSurface::empty(),
            generation: GenerationOptions::default(),
            retry: RetryPolicy::default(),
            limits: TurnLimits::default(),
            tool_output_limits: crate::tool::ToolOutputLimits::default(),
            artifact_store: None,
            window_budget: crate::turn::WindowBudget::default(),
            compaction: None,
            token_counter: None,
            call_timeout: None,
            attempt_timeout: None,
        }
    }
}
