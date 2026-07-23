//! Domain event adapter trait: project service-local events into the common
//! host-facing event stream.
//!
//! Each service crate keeps its own local event enum and implements
//! [`DomainEventAdapter`] for that enum when it needs to surface its events
//! to a host (Tauri, future Axum, tests). Core owns the trait and the
//! [`EventReport`] / [`EventAdapterContext`] shapes; it does not own an event
//! bus, a sink, or a service-specific event enum.

use crate::diagnostic::{CorrelationId, Diagnostic, DiagnosticTarget};
use crate::event::domain_event::DomainEvent;
use crate::event::report::OperationReport;
use crate::event::source::DomainEventSource;

/// Context passed to a [`DomainEventAdapter`] when projecting a local
/// service event into the common host-facing event stream.
///
/// The context carries the host-known identity fields that every projected
/// [`DomainEvent`] should reflect: the subsystem that emitted it
/// ([`DomainEventSource`]), an optional correlation id for grouping, and an
/// optional subject describing what the event is about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventAdapterContext {
    source: DomainEventSource,
    correlation_id: Option<CorrelationId>,
    subject: Option<DiagnosticTarget>,
}

impl EventAdapterContext {
    /// Create a context seeded with the subsystem that emitted the event.
    pub fn new(source: DomainEventSource) -> Self {
        Self {
            source,
            correlation_id: None,
            subject: None,
        }
    }

    /// Attach a correlation id so the produced events group with other
    /// events and diagnostics from the same operation.
    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Attach a subject describing what the event is about.
    pub fn with_subject(mut self, subject: DiagnosticTarget) -> Self {
        self.subject = Some(subject);
        self
    }

    pub fn source(&self) -> &DomainEventSource {
        &self.source
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn subject(&self) -> Option<&DiagnosticTarget> {
        self.subject.as_ref()
    }
}

/// Result of adapting a local service event into common host-facing events.
///
/// `EventReport` mirrors the shape of [`OperationReport`] but is the
/// dedicated return type for [`DomainEventAdapter::adapt`]. It collects the
/// projected [`DomainEvent`] values and any [`Diagnostic`] values that did
/// not belong inside a single event.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EventReport {
    events: Vec<DomainEvent>,
    diagnostics: Vec<Diagnostic>,
}

impl EventReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a report seeded with a single domain event.
    pub fn with_event(event: DomainEvent) -> Self {
        Self {
            events: vec![event],
            diagnostics: Vec::new(),
        }
    }

    /// Create a report seeded with a single diagnostic.
    pub fn with_diagnostic(diagnostic: Diagnostic) -> Self {
        Self {
            events: Vec::new(),
            diagnostics: vec![diagnostic],
        }
    }

    /// Append a domain event.
    pub fn push_event(&mut self, event: DomainEvent) {
        self.events.push(event);
    }

    /// Append a diagnostic.
    pub fn push_diagnostic(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Merge another report into this one.
    pub fn extend(&mut self, other: EventReport) {
        self.events.extend(other.events);
        self.diagnostics.extend(other.diagnostics);
    }

    pub fn events(&self) -> &[DomainEvent] {
        &self.events
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Returns `true` when the report contains no events or diagnostics.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty() && self.diagnostics.is_empty()
    }
}

impl From<EventReport> for OperationReport {
    fn from(report: EventReport) -> Self {
        let mut op = OperationReport::new();
        for event in report.events {
            op.push_event(event);
        }
        for diagnostic in report.diagnostics {
            op.push_diagnostic(diagnostic);
        }
        op
    }
}

/// Projects a service-local event `S` into the common host-facing event
/// stream.
///
/// Implementors own the mapping from their own local event enum to one or
/// more [`DomainEvent`] values and any standalone [`Diagnostic`] values.
/// Core never knows what `S` is; the trait is generic over the local event
/// type so each service crate can adapt its own enum without leaking it
/// into core.
pub trait DomainEventAdapter<S> {
    /// Adapt a single local service event into an [`EventReport`].
    fn adapt(source: S, context: EventAdapterContext) -> EventReport;
}

#[cfg(test)]
mod tests {
    use crate::diagnostic::{
        CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
        DiagnosticTarget, DiagnosticTargetDomain,
    };
    use crate::event::{
        DomainEvent, DomainEventAdapter, DomainEventId, DomainEventKind, DomainEventSource,
        EventAdapterContext, EventReport, OperationReport, Timestamp,
    };
    use crate::model::DiagnosticId;

    // --- Config-style local event + adapter -----------------------------

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ConfigLocalEvent {
        Loaded {
            key: String,
            path: String,
        },
        Saved {
            key: String,
            bytes: usize,
        },
        ParseFailed {
            key: String,
            path: String,
            message: String,
        },
    }

    struct ConfigEventAdapter;

    impl DomainEventAdapter<ConfigLocalEvent> for ConfigEventAdapter {
        fn adapt(source: ConfigLocalEvent, context: EventAdapterContext) -> EventReport {
            let cid = context.correlation_id().cloned();
            match source {
                ConfigLocalEvent::Loaded { key, path } => {
                    let mut event = DomainEvent::new(
                        DomainEventId::new(format!("config:{key}:loaded")),
                        context.source().clone(),
                        DomainEventKind::new("config.loaded"),
                        Timestamp::new("2026-06-10T00:00:00Z"),
                    )
                    .with_subject(
                        DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                            .with_id(key)
                            .with_path(path),
                    );
                    if let Some(c) = cid {
                        event = event.with_correlation_id(c);
                    }
                    EventReport::with_event(event)
                }
                ConfigLocalEvent::Saved { key, bytes } => {
                    let mut event = DomainEvent::new(
                        DomainEventId::new(format!("config:{key}:saved")),
                        context.source().clone(),
                        DomainEventKind::new("config.saved"),
                        Timestamp::new("2026-06-10T00:00:00Z"),
                    )
                    .with_subject(
                        DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                            .with_id(key)
                            .with_path(format!("{bytes} bytes")),
                    );
                    if let Some(c) = cid {
                        event = event.with_correlation_id(c);
                    }
                    EventReport::with_event(event)
                }
                ConfigLocalEvent::ParseFailed { key, path, message } => {
                    let diagnostic = Diagnostic::new(
                        DiagnosticId::new(format!("config:{key}:parse_failed")),
                        DiagnosticCode::new("CONFIG/PARSE_FAILED"),
                        DiagnosticSeverity::Error,
                        DiagnosticSourceName::new("config"),
                        message,
                        DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                            .with_id(key)
                            .with_path(path),
                    );
                    EventReport::with_diagnostic(diagnostic)
                }
            }
        }
    }

    #[test]
    fn config_loaded_event_is_projected_with_context_source_and_subject() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("config"))
            .with_correlation_id(CorrelationId::new("corr-config"));

        let report = <ConfigEventAdapter as DomainEventAdapter<ConfigLocalEvent>>::adapt(
            ConfigLocalEvent::Loaded {
                key: "model_series".into(),
                path: "~/.reimagine/config/model_series.json".into(),
            },
            ctx,
        );

        assert_eq!(report.events().len(), 1);
        assert!(report.diagnostics().is_empty());
        let ev = &report.events()[0];
        assert_eq!(ev.source().as_str(), "config");
        assert_eq!(ev.kind().as_str(), "config.loaded");
        assert_eq!(ev.correlation_id().unwrap().as_str(), "corr-config");
        assert_eq!(ev.subject().unwrap().id(), Some("model_series"));
    }

    #[test]
    fn config_parse_failed_returns_standalone_diagnostic_not_event() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("config"));

        let report = <ConfigEventAdapter as DomainEventAdapter<ConfigLocalEvent>>::adapt(
            ConfigLocalEvent::ParseFailed {
                key: "model_series".into(),
                path: "~/.reimagine/config/model_series.json".into(),
                message: "expected `,` or `}` at line 4".into(),
            },
            ctx,
        );

        assert!(report.events().is_empty());
        assert_eq!(report.diagnostics().len(), 1);
        let diag = &report.diagnostics()[0];
        assert_eq!(diag.code().as_str(), "CONFIG/PARSE_FAILED");
        assert_eq!(diag.severity(), DiagnosticSeverity::Error);
        assert_eq!(diag.source().as_str(), "config");
        assert_eq!(diag.primary().id(), Some("model_series"));
        assert_eq!(
            diag.primary().path(),
            Some("~/.reimagine/config/model_series.json")
        );
    }

    #[test]
    fn config_saved_event_carries_subject_path() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("config"));

        let report = <ConfigEventAdapter as DomainEventAdapter<ConfigLocalEvent>>::adapt(
            ConfigLocalEvent::Saved {
                key: "model_series".into(),
                bytes: 4096,
            },
            ctx,
        );

        assert_eq!(report.events().len(), 1);
        assert!(report.diagnostics().is_empty());
        let ev = &report.events()[0];
        assert_eq!(ev.kind().as_str(), "config.saved");
        assert_eq!(ev.subject().unwrap().path(), Some("4096 bytes"));
    }

    // --- Agent-style local event + adapter ------------------------------

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum AgentLocalEvent {
        SessionStarted {
            session_id: String,
            provider: String,
        },
        ToolInvoked {
            session_id: String,
            tool: String,
        },
        ProviderError {
            session_id: String,
            message: String,
        },
        ProposalReady {
            session_id: String,
            proposal_id: String,
        },
    }

    struct AgentEventAdapter;

    impl DomainEventAdapter<AgentLocalEvent> for AgentEventAdapter {
        fn adapt(source: AgentLocalEvent, context: EventAdapterContext) -> EventReport {
            let cid = context.correlation_id().cloned();
            let subject_session = |sid: &str| {
                DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.session"))
                    .with_id(sid.to_owned())
            };
            match source {
                AgentLocalEvent::SessionStarted {
                    session_id,
                    provider: _,
                } => {
                    let mut event = DomainEvent::new(
                        DomainEventId::new(format!("agent:{session_id}:started")),
                        context.source().clone(),
                        DomainEventKind::new("agent.session_started"),
                        Timestamp::new("2026-06-10T00:00:00Z"),
                    )
                    .with_subject(subject_session(&session_id));
                    if let Some(c) = cid {
                        event = event.with_correlation_id(c);
                    }
                    EventReport::with_event(event)
                }
                AgentLocalEvent::ToolInvoked { session_id, tool } => {
                    let mut event = DomainEvent::new(
                        DomainEventId::new(format!("agent:{session_id}:tool:{tool}")),
                        context.source().clone(),
                        DomainEventKind::new("agent.tool_invoked"),
                        Timestamp::new("2026-06-10T00:00:00Z"),
                    )
                    .with_subject(
                        DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool"))
                            .with_id(format!("{session_id}:{tool}")),
                    );
                    if let Some(c) = cid.clone() {
                        event = event.with_correlation_id(c);
                    }
                    EventReport::with_event(event)
                }
                AgentLocalEvent::ProviderError {
                    session_id,
                    message,
                } => {
                    let diagnostic = Diagnostic::new(
                        DiagnosticId::new(format!("agent:{session_id}:provider_error")),
                        DiagnosticCode::new("AGENT/PROVIDER_ERROR"),
                        DiagnosticSeverity::Error,
                        DiagnosticSourceName::new("agent"),
                        message,
                        subject_session(&session_id),
                    );
                    EventReport::with_diagnostic(diagnostic)
                }
                AgentLocalEvent::ProposalReady {
                    session_id,
                    proposal_id,
                } => {
                    let mut event = DomainEvent::new(
                        DomainEventId::new(format!("agent:{session_id}:proposal:{proposal_id}")),
                        context.source().clone(),
                        DomainEventKind::new("agent.proposal_ready"),
                        Timestamp::new("2026-06-10T00:00:00Z"),
                    )
                    .with_subject(
                        DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.proposal"))
                            .with_id(proposal_id),
                    );
                    if let Some(c) = cid {
                        event = event.with_correlation_id(c);
                    }
                    EventReport::with_event(event)
                }
            }
        }
    }

    #[test]
    fn agent_session_started_event_propagates_correlation_id_from_context() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("agent"))
            .with_correlation_id(CorrelationId::new("corr-agent-1"))
            .with_subject(
                DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.session"))
                    .with_id("sess-1"),
            );

        let report = <AgentEventAdapter as DomainEventAdapter<AgentLocalEvent>>::adapt(
            AgentLocalEvent::SessionStarted {
                session_id: "sess-1".into(),
                provider: "openai".into(),
            },
            ctx,
        );

        assert_eq!(report.events().len(), 1);
        let ev = &report.events()[0];
        assert_eq!(ev.source().as_str(), "agent");
        assert_eq!(ev.kind().as_str(), "agent.session_started");
        assert_eq!(ev.correlation_id().unwrap().as_str(), "corr-agent-1");
        assert_eq!(ev.subject().unwrap().id(), Some("sess-1"));
    }

    #[test]
    fn agent_provider_error_returns_diagnostic_with_session_subject() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("agent"));

        let report = <AgentEventAdapter as DomainEventAdapter<AgentLocalEvent>>::adapt(
            AgentLocalEvent::ProviderError {
                session_id: "sess-1".into(),
                message: "rate limit exceeded".into(),
            },
            ctx,
        );

        assert!(report.events().is_empty());
        assert_eq!(report.diagnostics().len(), 1);
        let diag = &report.diagnostics()[0];
        assert_eq!(diag.code().as_str(), "AGENT/PROVIDER_ERROR");
        assert_eq!(diag.severity(), DiagnosticSeverity::Error);
        assert_eq!(diag.source().as_str(), "agent");
        assert_eq!(diag.primary().domain().as_str(), "agent.session");
        assert_eq!(diag.primary().id(), Some("sess-1"));
    }

    #[test]
    fn agent_proposal_ready_event_carries_proposal_subject() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("agent"));

        let report = <AgentEventAdapter as DomainEventAdapter<AgentLocalEvent>>::adapt(
            AgentLocalEvent::ProposalReady {
                session_id: "sess-1".into(),
                proposal_id: "prop-42".into(),
            },
            ctx,
        );

        assert_eq!(report.events().len(), 1);
        let ev = &report.events()[0];
        assert_eq!(ev.kind().as_str(), "agent.proposal_ready");
        assert_eq!(ev.subject().unwrap().domain().as_str(), "agent.proposal");
        assert_eq!(ev.subject().unwrap().id(), Some("prop-42"));
    }

    #[test]
    fn agent_tool_invoked_event_has_session_scoped_tool_subject() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("agent"));

        let report = <AgentEventAdapter as DomainEventAdapter<AgentLocalEvent>>::adapt(
            AgentLocalEvent::ToolInvoked {
                session_id: "sess-1".into(),
                tool: "list_workflows".into(),
            },
            ctx,
        );

        assert_eq!(report.events().len(), 1);
        assert!(report.diagnostics().is_empty());
        let ev = &report.events()[0];
        assert_eq!(ev.kind().as_str(), "agent.tool_invoked");
        assert_eq!(ev.subject().unwrap().domain().as_str(), "agent.tool");
        assert_eq!(ev.subject().unwrap().id(), Some("sess-1:list_workflows"));
    }

    #[test]
    fn agent_tool_invoked_event_propagates_correlation_id_from_context() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("agent"))
            .with_correlation_id(CorrelationId::new("corr-agent-tool"));

        let report = <AgentEventAdapter as DomainEventAdapter<AgentLocalEvent>>::adapt(
            AgentLocalEvent::ToolInvoked {
                session_id: "sess-1".into(),
                tool: "list_workflows".into(),
            },
            ctx,
        );

        assert_eq!(
            report.events()[0].correlation_id().unwrap().as_str(),
            "corr-agent-tool"
        );
    }

    #[test]
    fn agent_proposal_ready_event_propagates_correlation_id_from_context() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("agent"))
            .with_correlation_id(CorrelationId::new("corr-agent-proposal"));

        let report = <AgentEventAdapter as DomainEventAdapter<AgentLocalEvent>>::adapt(
            AgentLocalEvent::ProposalReady {
                session_id: "sess-1".into(),
                proposal_id: "prop-42".into(),
            },
            ctx,
        );

        assert_eq!(
            report.events()[0].correlation_id().unwrap().as_str(),
            "corr-agent-proposal"
        );
    }

    // --- EventReport behavior -------------------------------------------

    #[test]
    fn event_report_extend_merges_events_and_diagnostics() {
        let r1 = EventReport::with_event(DomainEvent::new(
            DomainEventId::new("ev-1"),
            DomainEventSource::new("config"),
            DomainEventKind::new("config.loaded"),
            Timestamp::new("2026-06-10T00:00:00Z"),
        ));
        let r2 = EventReport::with_diagnostic(Diagnostic::new(
            DiagnosticId::new("d-1"),
            DiagnosticCode::new("CONFIG/INFO"),
            DiagnosticSeverity::Info,
            DiagnosticSourceName::new("config"),
            "ok",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("config")),
        ));

        let mut combined = r1;
        combined.extend(r2);
        assert_eq!(combined.events().len(), 1);
        assert_eq!(combined.diagnostics().len(), 1);
        assert!(!combined.is_empty());
    }

    #[test]
    fn event_report_default_is_empty() {
        let r = EventReport::default();
        assert!(r.is_empty());
        assert!(r.events().is_empty());
        assert!(r.diagnostics().is_empty());
    }

    #[test]
    fn event_report_converts_into_operation_report() {
        let ev = DomainEvent::new(
            DomainEventId::new("ev-1"),
            DomainEventSource::new("config"),
            DomainEventKind::new("config.loaded"),
            Timestamp::new("2026-06-10T00:00:00Z"),
        );
        let diag = Diagnostic::new(
            DiagnosticId::new("d-1"),
            DiagnosticCode::new("CONFIG/INFO"),
            DiagnosticSeverity::Info,
            DiagnosticSourceName::new("config"),
            "ok",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("config")),
        );
        let mut report = EventReport::default();
        report.push_event(ev.clone());
        report.push_diagnostic(diag.clone());

        let op: OperationReport = report.into();
        assert_eq!(op.events().len(), 1);
        assert_eq!(op.events()[0].id().as_str(), "ev-1");
        assert_eq!(op.diagnostics().len(), 1);
        assert_eq!(op.diagnostics()[0].id().as_str(), "d-1");
    }

    // --- EventAdapterContext behavior -----------------------------------

    #[test]
    fn event_adapter_context_minimal() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("config"));
        assert_eq!(ctx.source().as_str(), "config");
        assert!(ctx.correlation_id().is_none());
        assert!(ctx.subject().is_none());
    }

    #[test]
    fn event_adapter_context_with_correlation_and_subject() {
        let ctx = EventAdapterContext::new(DomainEventSource::new("agent"))
            .with_correlation_id(CorrelationId::new("corr-1"))
            .with_subject(
                DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.session"))
                    .with_id("sess-1"),
            );
        assert_eq!(ctx.correlation_id().unwrap().as_str(), "corr-1");
        assert_eq!(ctx.subject().unwrap().id(), Some("sess-1"));
    }
}
