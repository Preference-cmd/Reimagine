//! Projection of agent-local events into the common host-facing event
//! stream.
//!
//! `AgentDomainEventAdapter` implements the core
//! `DomainEventAdapter<AgentEvent>` trait. It does not own an event
//! bus, a sink, or any other transport. Hosts (app-host, tests) wire the
//! adapter to whichever transport they prefer (Tauri events, future
//! Axum SSE, etc.).
//!
//! The mapping rules are:
//!
//! - `SessionStarted` -> `DomainEvent` with kind `agent.session_started`,
//!   subject = `agent.session/<session_id>`.
//! - `SessionStopped` -> `DomainEvent` with kind `agent.session_stopped`,
//!   subject = `agent.session/<session_id>`, `path` carries the reason.
//! - `ToolInvoked` -> `DomainEvent` with kind `agent.tool_invoked`,
//!   subject = `agent.tool/<session_id>:<tool>`.
//! - `ToolCompleted` -> `DomainEvent` with kind `agent.tool_completed`,
//!   subject = `agent.tool/<session_id>:<tool>`.
//! - `ToolFailed` -> `Diagnostic` (not a `DomainEvent`), with code
//!   `AGENT/TOOL_*`, subject = `agent.tool/<session_id>:<tool>`.
//! - `ProviderError` -> `Diagnostic` with code
//!   `AGENT/PROVIDER_<code>`, subject = `agent.session/<session_id>`,
//!   `path` carries the provider name.
//! - `ProposalReady` -> `DomainEvent` with kind `agent.proposal_ready`,
//!   subject = `agent.proposal/<proposal_id>`.

use reimagine_core::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain,
};
use reimagine_core::event::{
    DomainEvent, DomainEventAdapter, DomainEventId, DomainEventKind, DomainEventSource,
    EventAdapterContext, EventReport,
};
use reimagine_core::model::DiagnosticId;

use crate::event::AgentEvent;

/// Caller-supplied timestamp newtype. We re-use core's `Timestamp` so
/// the projected event matches the host-facing event language exactly.
use reimagine_core::event::Timestamp;

/// Adapter that projects [`AgentEvent`] into the common host-facing
/// event stream.
///
/// `AgentDomainEventAdapter` is a zero-sized struct; the projection is
/// pure and stateless.
pub struct AgentDomainEventAdapter;

impl AgentDomainEventAdapter {
    pub fn new() -> Self {
        Self
    }

    fn now() -> Timestamp {
        // Core does not read the clock. Hosts (Tauri, future Axum) are
        // expected to attach their own clock when they bridge events
        // out. The adapter uses a stable placeholder so tests and
        // deterministic builds don't depend on `SystemTime`.
        Timestamp::new("1970-01-01T00:00:00Z")
    }

    fn subject_session(session_id: &str) -> DiagnosticTarget {
        DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.session"))
            .with_id(session_id.to_owned())
    }

    fn subject_tool(session_id: &str, tool: &str) -> DiagnosticTarget {
        DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool"))
            .with_id(format!("{session_id}:{tool}"))
    }

    fn subject_proposal(proposal_id: &str) -> DiagnosticTarget {
        DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.proposal"))
            .with_id(proposal_id.to_owned())
    }

    fn build_event(
        kind: &str,
        source: DomainEventSource,
        subject: DiagnosticTarget,
        correlation_id: Option<&CorrelationId>,
    ) -> DomainEvent {
        let id_string = format!("{}::{}", source.as_str(), kind);
        let id_string = format!("{id_string}::{}", subject.id().unwrap_or(""));
        let mut ev = DomainEvent::new(
            DomainEventId::new(id_string),
            source,
            DomainEventKind::new(kind),
            Self::now(),
        )
        .with_subject(subject);
        if let Some(c) = correlation_id.cloned() {
            ev = ev.with_correlation_id(c);
        }
        ev
    }
}

impl Default for AgentDomainEventAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainEventAdapter<AgentEvent> for AgentDomainEventAdapter {
    fn adapt(source: AgentEvent, context: EventAdapterContext) -> EventReport {
        let source_name = context.source().clone();
        let correlation_id = context.correlation_id().cloned();
        match source {
            AgentEvent::SessionStarted { session_id, .. } => {
                let ev = Self::build_event(
                    "agent.session_started",
                    source_name,
                    Self::subject_session(session_id.as_str()),
                    correlation_id.as_ref(),
                );
                EventReport::with_event(ev)
            }
            AgentEvent::SessionStopped { session_id, reason } => {
                let subject = Self::subject_session(session_id.as_str()).with_path(reason.clone());
                let ev = Self::build_event(
                    "agent.session_stopped",
                    source_name,
                    subject,
                    correlation_id.as_ref(),
                );
                EventReport::with_event(ev)
            }
            AgentEvent::ToolInvoked {
                session_id, tool, ..
            } => {
                let ev = Self::build_event(
                    "agent.tool_invoked",
                    source_name,
                    Self::subject_tool(session_id.as_str(), tool.as_str()),
                    correlation_id.as_ref(),
                );
                EventReport::with_event(ev)
            }
            AgentEvent::ToolCompleted {
                session_id, tool, ..
            } => {
                let ev = Self::build_event(
                    "agent.tool_completed",
                    source_name,
                    Self::subject_tool(session_id.as_str(), tool.as_str()),
                    correlation_id.as_ref(),
                );
                EventReport::with_event(ev)
            }
            AgentEvent::ToolFailed {
                session_id,
                tool,
                id: _,
                code,
                message,
            } => {
                let mut diag = Diagnostic::new(
                    DiagnosticId::new(format!("agent:{session_id}:tool:{tool}:{}", code.as_str())),
                    DiagnosticCode::new(code.as_str()),
                    DiagnosticSeverity::Error,
                    DiagnosticSourceName::new("agent"),
                    message,
                    Self::subject_tool(session_id.as_str(), tool.as_str()),
                );
                if let Some(cid) = correlation_id {
                    diag = diag.with_correlation_id(cid);
                }
                EventReport::with_diagnostic(diag)
            }
            AgentEvent::ProviderError {
                session_id,
                provider,
                code,
                message,
            } => {
                let mut subject = Self::subject_session(session_id.as_str());
                subject = subject.with_path(provider.as_str().to_owned());
                let mut diag = Diagnostic::new(
                    DiagnosticId::new(format!("agent:{session_id}:provider:{provider}:{code}")),
                    DiagnosticCode::new(format!("AGENT/PROVIDER_{code}")),
                    DiagnosticSeverity::Error,
                    DiagnosticSourceName::new("agent"),
                    message,
                    subject,
                );
                if let Some(cid) = correlation_id {
                    diag = diag.with_correlation_id(cid);
                }
                EventReport::with_diagnostic(diag)
            }
            AgentEvent::ProposalReady {
                session_id,
                proposal_id,
            } => {
                let _ = session_id;
                let ev = Self::build_event(
                    "agent.proposal_ready",
                    source_name,
                    Self::subject_proposal(&proposal_id),
                    correlation_id.as_ref(),
                );
                EventReport::with_event(ev)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ProviderName;
    use crate::mode::AgentMode;

    fn ctx() -> EventAdapterContext {
        EventAdapterContext::new(DomainEventSource::new("agent"))
    }

    fn ctx_with_correlation(cid: &str) -> EventAdapterContext {
        EventAdapterContext::new(DomainEventSource::new("agent"))
            .with_correlation_id(CorrelationId::new(cid))
    }

    #[test]
    fn session_started_becomes_event() {
        let report = <AgentDomainEventAdapter as DomainEventAdapter<AgentEvent>>::adapt(
            AgentEvent::SessionStarted {
                session_id: crate::ids::AgentSessionId::new("sess-1"),
                provider: ProviderName::new("openai"),
                mode: AgentMode::Agent,
            },
            ctx(),
        );
        assert_eq!(report.events().len(), 1);
        assert!(report.diagnostics().is_empty());
        let ev = &report.events()[0];
        assert_eq!(ev.kind().as_str(), "agent.session_started");
        assert_eq!(ev.subject().unwrap().domain().as_str(), "agent.session");
        assert_eq!(ev.subject().unwrap().id(), Some("sess-1"));
    }

    #[test]
    fn session_stopped_carries_reason_in_path() {
        let report = <AgentDomainEventAdapter as DomainEventAdapter<AgentEvent>>::adapt(
            AgentEvent::SessionStopped {
                session_id: crate::ids::AgentSessionId::new("sess-1"),
                reason: "user closed".into(),
            },
            ctx(),
        );
        let ev = &report.events()[0];
        assert_eq!(ev.kind().as_str(), "agent.session_stopped");
        assert_eq!(ev.subject().unwrap().path(), Some("user closed"));
    }

    #[test]
    fn tool_invoked_becomes_event() {
        let report = <AgentDomainEventAdapter as DomainEventAdapter<AgentEvent>>::adapt(
            AgentEvent::ToolInvoked {
                session_id: crate::ids::AgentSessionId::new("sess-1"),
                tool: crate::ids::ToolName::new("echo"),
                id: None,
            },
            ctx(),
        );
        let ev = &report.events()[0];
        assert_eq!(ev.kind().as_str(), "agent.tool_invoked");
        assert_eq!(ev.subject().unwrap().id(), Some("sess-1:echo"));
    }

    #[test]
    fn tool_completed_becomes_event() {
        let report = <AgentDomainEventAdapter as DomainEventAdapter<AgentEvent>>::adapt(
            AgentEvent::ToolCompleted {
                session_id: crate::ids::AgentSessionId::new("sess-1"),
                tool: crate::ids::ToolName::new("echo"),
                id: None,
            },
            ctx(),
        );
        let ev = &report.events()[0];
        assert_eq!(ev.kind().as_str(), "agent.tool_completed");
    }

    #[test]
    fn tool_failed_becomes_diagnostic() {
        let report = <AgentDomainEventAdapter as DomainEventAdapter<AgentEvent>>::adapt(
            AgentEvent::ToolFailed {
                session_id: crate::ids::AgentSessionId::new("sess-1"),
                tool: crate::ids::ToolName::new("echo"),
                id: None,
                code: crate::error::ToolErrorCode::ExecutionFailed,
                message: "kaboom".into(),
            },
            ctx(),
        );
        assert!(report.events().is_empty());
        assert_eq!(report.diagnostics().len(), 1);
        let diag = &report.diagnostics()[0];
        assert_eq!(diag.code().as_str(), "AGENT/TOOL_EXECUTION_FAILED");
        assert_eq!(diag.primary().id(), Some("sess-1:echo"));
    }

    #[test]
    fn provider_error_becomes_diagnostic_with_provider_path() {
        let report = <AgentDomainEventAdapter as DomainEventAdapter<AgentEvent>>::adapt(
            AgentEvent::ProviderError {
                session_id: crate::ids::AgentSessionId::new("sess-1"),
                provider: ProviderName::new("openai"),
                code: "RATE_LIMIT".into(),
                message: "slow down".into(),
            },
            ctx(),
        );
        assert!(report.events().is_empty());
        let diag = &report.diagnostics()[0];
        assert_eq!(diag.code().as_str(), "AGENT/PROVIDER_RATE_LIMIT");
        assert_eq!(diag.primary().id(), Some("sess-1"));
        assert_eq!(diag.primary().path(), Some("openai"));
    }

    #[test]
    fn proposal_ready_becomes_event_with_proposal_subject() {
        let report = <AgentDomainEventAdapter as DomainEventAdapter<AgentEvent>>::adapt(
            AgentEvent::ProposalReady {
                session_id: crate::ids::AgentSessionId::new("sess-1"),
                proposal_id: "prop-42".into(),
            },
            ctx(),
        );
        let ev = &report.events()[0];
        assert_eq!(ev.kind().as_str(), "agent.proposal_ready");
        assert_eq!(ev.subject().unwrap().id(), Some("prop-42"));
    }

    #[test]
    fn correlation_id_is_propagated_from_context() {
        let report = <AgentDomainEventAdapter as DomainEventAdapter<AgentEvent>>::adapt(
            AgentEvent::ToolInvoked {
                session_id: crate::ids::AgentSessionId::new("sess-1"),
                tool: crate::ids::ToolName::new("echo"),
                id: None,
            },
            ctx_with_correlation("corr-1"),
        );
        assert_eq!(
            report.events()[0].correlation_id().unwrap().as_str(),
            "corr-1"
        );
    }
}
