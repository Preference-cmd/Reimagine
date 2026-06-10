//! Agent-local event model.
//!
//! `AgentEvent` is the canonical event type produced by the agent
//! runtime. It is projected into the host-facing `DomainEvent` stream
//! through the `AgentDomainEventAdapter` (see `event_adapter.rs`).

use serde::{Deserialize, Serialize};

use crate::ids::{AgentSessionId, ProviderName, ToolName};

/// Local event emitted by the agent runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentEvent {
    /// An agent session was started.
    SessionStarted {
        session_id: AgentSessionId,
        provider: ProviderName,
        mode: crate::mode::AgentMode,
    },
    /// An agent session was stopped.
    SessionStopped {
        session_id: AgentSessionId,
        reason: String,
    },
    /// A tool was invoked. The `id` is the tool call id supplied by the
    /// model (or the registry, for direct invocations).
    ToolInvoked {
        session_id: AgentSessionId,
        tool: ToolName,
        id: Option<crate::provider::ToolCallId>,
    },
    /// A tool invocation completed successfully.
    ToolCompleted {
        session_id: AgentSessionId,
        tool: ToolName,
        id: Option<crate::provider::ToolCallId>,
    },
    /// A tool invocation failed.
    ToolFailed {
        session_id: AgentSessionId,
        tool: ToolName,
        id: Option<crate::provider::ToolCallId>,
        code: crate::error::ToolErrorCode,
        message: String,
    },
    /// A provider produced an error. The runtime emits this so hosts
    /// can show transient provider errors and retry.
    ProviderError {
        session_id: AgentSessionId,
        provider: ProviderName,
        code: String,
        message: String,
    },
    /// A build-mode proposal is ready for human review.
    ProposalReady {
        session_id: AgentSessionId,
        proposal_id: String,
    },
}

impl AgentEvent {
    /// The agent session id this event belongs to. Every event variant
    /// is required to carry one.
    pub fn session_id(&self) -> &AgentSessionId {
        match self {
            Self::SessionStarted { session_id, .. }
            | Self::SessionStopped { session_id, .. }
            | Self::ToolInvoked { session_id, .. }
            | Self::ToolCompleted { session_id, .. }
            | Self::ToolFailed { session_id, .. }
            | Self::ProviderError { session_id, .. }
            | Self::ProposalReady { session_id, .. } => session_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode::AgentMode;

    #[test]
    fn event_session_id_extractor() {
        let sid = AgentSessionId::new("sess-1");
        let ev = AgentEvent::SessionStarted {
            session_id: sid.clone(),
            provider: ProviderName::new("openai"),
            mode: AgentMode::Agent,
        };
        assert_eq!(ev.session_id(), &sid);
    }
}
