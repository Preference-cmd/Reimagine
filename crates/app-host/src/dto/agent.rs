//! Agent DTOs for Tauri/Axum IPC.
//!
//! These shapes project host-neutral `reimagine_agent` types into stable
//! JSON forms that frontends can consume directly. The projections strip
//! backend-internal types and private fields so frontends never see
//! agent-crate internals.

use reimagine_agent::{AgentEvent, AgentSession, AgentTurnResult, Message, ToolCallResult, Usage};
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
            messages: value.messages().iter().map(|m| m.clone().into()).collect(),
        }
    }
}

/// Agent message DTO (projection of `reimagine_agent::Message`).
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

/// Agent tool call DTO (projection of `reimagine_agent::ToolCallResult`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolCallDto {
    pub tool_call_id: String,
    pub tool_name: String,
    pub status: String,
    pub output: Option<serde_json::Value>,
    pub effective: Option<bool>,
}

impl From<ToolCallResult> for AgentToolCallDto {
    fn from(tc: ToolCallResult) -> Self {
        Self {
            tool_call_id: tc.tool_call_id().to_string(),
            tool_name: tc.tool_name().to_string(),
            status: format!("{:?}", tc.status()),
            output: tc.output().cloned(),
            effective: tc.effective(),
        }
    }
}

/// Agent event payload for Tauri Channel streaming.
///
/// This is the event shape sent over `Channel<AgentEventPayload>` from
/// `TauriAgentEventHub`. Mirrors `RunEventPayload` in naming convention.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentEventPayload {
    pub session_id: String,
    pub kind: String,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub code: Option<String>,
    pub message: Option<String>,
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
            },
            AgentEvent::SessionStopped { session_id, reason } => Self {
                session_id: session_id.to_string(),
                kind: "session_stopped".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(reason.clone()),
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
}

impl From<Usage> for AgentUsageDto {
    fn from(u: Usage) -> Self {
        Self {
            input_tokens: u.input_tokens(),
            output_tokens: u.output_tokens(),
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
    use reimagine_agent::{AgentMode, AgentSessionId, AgentTurnResult, ProviderName};

    #[test]
    fn agent_turn_projects_from_result() {
        let result = AgentTurnResult::new()
            .with_turn_id(reimagine_agent::AgentTurnId::new("turn-1"))
            .with_session_id(AgentSessionId::new("sess-1"))
            .with_mode(AgentMode::Agent)
            .with_provider(ProviderName::new("openai"))
            .with_model(reimagine_agent::ModelName::new("gpt-4"));

        let dto: AgentTurnResponse = result.into();
        assert_eq!(dto.turn_id, "turn-1");
        assert_eq!(dto.session_id, "sess-1");
        assert_eq!(dto.mode, "Agent");
        assert_eq!(dto.provider, "openai");
    }
}
