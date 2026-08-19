//! Agent DTOs for Tauri/Axum IPC.
//!
//! These shapes project host-neutral `reimagine_agent_harness` types into stable
//! JSON forms that frontends can consume directly. The projections strip
//! backend-internal types and private fields so frontends never see
//! agent-crate internals.

use reimagine_agent_harness::{
    AgentEvent, AgentSession, AgentTurnResult, Message, ToolCallResult, Usage,
};
use serde::{Deserialize, Serialize};

use super::runs::DiagnosticDto;

/// Agent turn response projected for IPC transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentTurnResponse {
    pub turn_id: String,
    pub session_id: String,
    pub mode: String,
    pub provider: String,
    pub model: String,
    pub status: String,
    pub stop_reason: String,
    pub final_response: Option<AgentMessageDto>,
    pub tool_calls: Vec<AgentToolCallDto>,
    pub diagnostics: Vec<DiagnosticDto>,
    pub usage: Option<AgentUsageDto>,
    /// Wall-clock duration of the turn in milliseconds (AR-41).
    pub duration_ms: Option<u64>,
    /// Estimated cost in USD for the turn's aggregated usage, priced at
    /// the V1 default per-million rates (AR-29/AR-41). `None` when the
    /// turn reported no usage.
    pub estimated_cost: Option<f64>,
    pub messages: Vec<AgentMessageDto>,
}

impl From<AgentTurnResult> for AgentTurnResponse {
    fn from(value: AgentTurnResult) -> Self {
        Self {
            turn_id: value.turn_id().to_string(),
            session_id: value.session_id().to_string(),
            mode: format!("{:?}", value.mode()),
            provider: value.provider().to_string(),
            model: value.model().to_string(),
            status: format!("{:?}", value.status()),
            stop_reason: format!("{:?}", value.stop_reason()),
            final_response: value
                .final_response()
                .map(|m| AgentMessageDto::from(m.clone())),
            tool_calls: value
                .tool_calls()
                .iter()
                .map(|tc| tc.clone().into())
                .collect(),
            diagnostics: value
                .diagnostics()
                .iter()
                .map(|d| DiagnosticDto::from(d.clone()))
                .collect(),
            usage: value.usage().map(|u| u.clone().into()),
            // V1 estimate default (AR-29): $3/M input, $15/M output.
            duration_ms: value.duration().map(|d| d.as_millis() as u64),
            estimated_cost: value.estimated_cost(3.0, 15.0),
            messages: value.messages().iter().map(|m| m.clone().into()).collect(),
        }
    }
}

/// Agent message DTO (projection of `reimagine_agent_harness::Message`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentMessageDto {
    pub role: String,
    pub content: String,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<AgentToolCallDto>,
}

impl From<Message> for AgentMessageDto {
    fn from(m: Message) -> Self {
        Self {
            role: m.role().to_string(),
            content: m.content().to_string(),
            tool_call_id: m.tool_call_id().map(|id| id.to_string()),
            tool_calls: Vec::new(), // Messages carry ToolCall from the provider, not ToolCallResult
        }
    }
}

/// Agent tool call DTO (projection of `reimagine_agent_harness::ToolCallResult`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolCallDto {
    pub tool_call_id: String,
    pub tool_name: String,
    pub status: String,
    pub output: Option<serde_json::Value>,
    pub effective: Option<bool>,
    /// Wall-clock duration of this invocation in milliseconds (AR-41).
    pub duration_ms: Option<u64>,
}

impl From<ToolCallResult> for AgentToolCallDto {
    fn from(tc: ToolCallResult) -> Self {
        Self {
            tool_call_id: tc.tool_call_id().to_string(),
            tool_name: tc.tool_name().to_string(),
            status: format!("{:?}", tc.status()),
            output: tc.output().cloned(),
            effective: tc.effective(),
            duration_ms: tc.duration().map(|d| d.as_millis() as u64),
        }
    }
}

/// Agent event payload for Tauri Channel streaming.
///
/// This is the event shape sent over `Channel<AgentEventPayload>` from
/// `TauriAgentEventHub`. Mirrors `RunEventPayload` in naming convention.
///
/// # Kind semantics (AR-11)
/// The backend emits two terminal milestones with canonical names:
/// - `error` — the turn ended in failure (sent by the host via
///   `TauriAgentEventHub::send_error` after the harness surfaced a provider
///   error);
/// - `turn_completed` — the turn ended normally.
///
/// In-flight kinds mirror the harness projection and are consumed directly
/// by the UI: `content_delta`, `reasoning_delta`, `tool_invoked`,
/// `tool_completed`, `tool_failed`, `session_started`, `session_stopped`,
/// `context_compacted`. The `provider_error` kind is the UI short-term
/// compatibility spelling for provider failures and is grouped with
/// `error` by [`Self::is_error`], so clients treat error semantics
/// uniformly.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentEventPayload {
    pub session_id: String,
    pub kind: String,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub code: Option<String>,
    pub message: Option<String>,
    /// Turn duration in milliseconds, populated on `turn_completed`.
    pub duration_ms: Option<u64>,
    /// Estimated USD cost for the aggregated turn usage.
    pub estimated_cost: Option<f64>,
    /// Aggregated usage, including reasoning/cache buckets.
    pub usage: Option<AgentUsageDto>,
}

impl AgentEventPayload {
    /// True for every event kind that signals a failure. The backend
    /// normalises on `error` semantics while keeping the `provider_error`
    /// spelling for UI compatibility (AR-11).
    pub fn is_error(&self) -> bool {
        matches!(
            self.kind.as_str(),
            "error" | "provider_error" | "tool_failed"
        )
    }
}

impl From<&AgentEvent> for AgentEventPayload {
    fn from(event: &AgentEvent) -> Self {
        match event {
            AgentEvent::SessionStarted {
                session_id,
                provider,
                mode,
            } => Self {
                session_id: session_id.to_string(),
                kind: "session_started".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(format!("provider={provider} mode={mode:?}")),
                ..Default::default()
            },
            AgentEvent::SessionStopped { session_id, reason } => Self {
                session_id: session_id.to_string(),
                kind: "session_stopped".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(reason.clone()),
                ..Default::default()
            },
            AgentEvent::ToolInvoked {
                session_id,
                tool,
                id,
            } => Self {
                session_id: session_id.to_string(),
                kind: "tool_invoked".to_string(),
                tool_name: Some(tool.to_string()),
                tool_call_id: id.as_ref().map(|id| id.to_string()),
                code: None,
                message: None,
                ..Default::default()
            },
            AgentEvent::ToolCompleted {
                session_id,
                tool,
                id,
            } => Self {
                session_id: session_id.to_string(),
                kind: "tool_completed".to_string(),
                tool_name: Some(tool.to_string()),
                tool_call_id: id.as_ref().map(|id| id.to_string()),
                code: None,
                message: None,
                ..Default::default()
            },
            AgentEvent::ToolFailed {
                session_id,
                tool,
                id,
                code,
                message,
            } => Self {
                session_id: session_id.to_string(),
                kind: "tool_failed".to_string(),
                tool_name: Some(tool.to_string()),
                tool_call_id: id.as_ref().map(|id| id.to_string()),
                code: Some(format!("{:?}", code)),
                message: Some(message.clone()),
                ..Default::default()
            },
            AgentEvent::ProviderError {
                session_id,
                provider: _,
                code,
                message,
            } => Self {
                session_id: session_id.to_string(),
                kind: "provider_error".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: Some(code.clone()),
                message: Some(message.clone()),
                ..Default::default()
            },
            AgentEvent::ProposalReady {
                session_id,
                proposal_id,
            } => Self {
                session_id: session_id.to_string(),
                kind: "proposal_ready".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(format!("proposal_id={proposal_id}")),
                ..Default::default()
            },
            AgentEvent::ContentDelta {
                session_id, text, ..
            } => Self {
                session_id: session_id.to_string(),
                kind: "content_delta".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(text.clone()),
                ..Default::default()
            },
            AgentEvent::ReasoningDelta {
                session_id, text, ..
            } => Self {
                session_id: session_id.to_string(),
                kind: "reasoning_delta".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(text.clone()),
                ..Default::default()
            },
            AgentEvent::ContextCompacted {
                session_id,
                summary,
                tokens_before,
                tokens_after,
            } => Self {
                session_id: session_id.to_string(),
                kind: "context_compacted".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(format!(
                    "tokens_before={tokens_before} tokens_after={tokens_after} summary={summary}"
                )),
                ..Default::default()
            },
        }
    }
}

/// Agent session info projected for IPC transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSessionInfo {
    pub session_id: String,
    pub mode: String,
    pub provider: String,
    pub started_at: String,
}

impl From<AgentSession> for AgentSessionInfo {
    fn from(session: AgentSession) -> Self {
        Self {
            session_id: session.id().to_string(),
            mode: format!("{:?}", session.mode()),
            provider: session.provider().to_string(),
            started_at: session.started_at().to_string(),
        }
    }
}

/// Agent usage DTO.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentUsageDto {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Reasoning tokens reported by the provider (AR-41).
    pub reasoning_tokens: Option<u64>,
    /// Tokens written into a provider cache, when reported.
    pub cache_creation_input_tokens: Option<u64>,
    /// Tokens read from a provider cache, when reported.
    pub cache_read_input_tokens: Option<u64>,
}

impl From<Usage> for AgentUsageDto {
    fn from(u: Usage) -> Self {
        Self {
            input_tokens: u.input_tokens(),
            output_tokens: u.output_tokens(),
            reasoning_tokens: u.reasoning_tokens(),
            cache_creation_input_tokens: u.cache_creation_input_tokens(),
            cache_read_input_tokens: u.cache_read_input_tokens(),
        }
    }
}

/// Command result re-exported from core for direct use in Tauri handlers.
///
/// `reimagine_core::command::CommandResult` already implements
/// `Serialize` / `Deserialize` so we re-export it here for IPC consumers.
#[allow(unused_imports)]
pub use reimagine_core::command::CommandResult;

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent_harness::{AgentMode, AgentSessionId, AgentTurnResult, ProviderName};

    #[test]
    fn agent_turn_projects_from_result() {
        let result = AgentTurnResult::new()
            .with_turn_id(reimagine_agent_harness::AgentTurnId::new("turn-1"))
            .with_session_id(AgentSessionId::new("sess-1"))
            .with_mode(AgentMode::Agent)
            .with_provider(ProviderName::new("openai"))
            .with_model(reimagine_agent_harness::ModelName::new("gpt-4"));

        let dto: AgentTurnResponse = result.into();
        assert_eq!(dto.turn_id, "turn-1");
        assert_eq!(dto.session_id, "sess-1");
        assert_eq!(dto.mode, "Agent");
        assert_eq!(dto.provider, "openai");
    }

    #[test]
    fn agent_turn_projects_duration_and_cost_and_per_tool_timing() {
        use reimagine_agent_harness::{AgentTurnId, ToolCallId, ToolCallResult, ToolName, Usage};
        use std::time::Duration;

        let start = std::time::Instant::now();
        let result = AgentTurnResult::new()
            .with_turn_id(AgentTurnId::new("turn-2"))
            .with_session_id(AgentSessionId::new("sess-2"))
            .with_mode(AgentMode::Agent)
            .with_provider(ProviderName::new("openai"))
            .with_model(reimagine_agent_harness::ModelName::new("gpt-4"))
            .with_usage(Usage::new(Some(1000), Some(2000)))
            .with_duration_since(start);
        let mut result = result;
        result.push_tool_call(
            ToolCallResult::new(ToolCallId::new("c1"), ToolName::new("echo"))
                .succeeded(serde_json::json!({"ok": true}))
                .with_duration(Duration::from_millis(7)),
        );

        let dto: AgentTurnResponse = result.into();
        assert!(dto.duration_ms.is_some(), "turn duration projected");
        assert!(
            dto.estimated_cost.is_some(),
            "usage present -> cost computed"
        );
        assert_eq!(dto.tool_calls.len(), 1);
        assert_eq!(dto.tool_calls[0].duration_ms, Some(7));
    }

    #[test]
    fn agent_turn_projects_no_cost_without_usage() {
        use std::time::Instant;

        let start = Instant::now();
        let result = AgentTurnResult::new().with_duration_since(start);
        let dto: AgentTurnResponse = result.into();
        assert!(dto.duration_ms.is_some());
        assert_eq!(dto.estimated_cost, None, "no usage -> no cost estimate");
    }
}
