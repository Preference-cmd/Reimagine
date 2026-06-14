use crate::diagnostic::{CorrelationId, Diagnostic, DiagnosticTarget};

use super::kind::DomainEventKind;
use super::source::DomainEventSource;

/// Caller-supplied id for a domain event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DomainEventId(String);

impl DomainEventId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DomainEventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DomainEventId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DomainEventId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Caller-supplied timestamp (string newtype). Core does not read the clock.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Timestamp(String);

impl Timestamp {
    pub fn new(ts: impl Into<String>) -> Self {
        Self(ts.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Timestamp {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Timestamp {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// A timeline / notification envelope for something that happened.
///
/// `DomainEvent` is the smallest unit that a future event bus may carry.
/// It can optionally carry diagnostics produced alongside the event.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DomainEvent {
    id: DomainEventId,
    correlation_id: Option<CorrelationId>,
    source: DomainEventSource,
    kind: DomainEventKind,
    subject: Option<DiagnosticTarget>,
    diagnostics: Vec<Diagnostic>,
    created_at: Timestamp,
}

impl DomainEvent {
    /// Create a domain event with the required fields.
    pub fn new(
        id: DomainEventId,
        source: DomainEventSource,
        kind: DomainEventKind,
        created_at: Timestamp,
    ) -> Self {
        Self {
            id,
            correlation_id: None,
            source,
            kind,
            subject: None,
            diagnostics: Vec::new(),
            created_at,
        }
    }

    /// Set the correlation id.
    pub fn with_correlation_id(mut self, id: CorrelationId) -> Self {
        self.correlation_id = Some(id);
        self
    }

    /// Set the subject target (what the event is about).
    pub fn with_subject(mut self, subject: DiagnosticTarget) -> Self {
        self.subject = Some(subject);
        self
    }

    /// Attach a diagnostic to this event.
    pub fn with_diagnostic(mut self, diagnostic: Diagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }

    pub fn id(&self) -> &DomainEventId {
        &self.id
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn source(&self) -> &DomainEventSource {
        &self.source
    }

    pub fn kind(&self) -> &DomainEventKind {
        &self.kind
    }

    pub fn subject(&self) -> Option<&DiagnosticTarget> {
        self.subject.as_ref()
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn created_at(&self) -> &Timestamp {
        &self.created_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::{
        DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTargetDomain,
    };
    use crate::model::DiagnosticId;

    #[test]
    fn domain_event_builder_roundtrip() {
        let diag = Diagnostic::new(
            DiagnosticId::new("d-ev-001"),
            DiagnosticCode::new("MODEL/MISSING"),
            DiagnosticSeverity::Warning,
            DiagnosticSourceName::new("model-manager"),
            "model weights not found",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model")).with_id("sd-1.5"),
        );

        let ev = DomainEvent::new(
            DomainEventId::new("ev-001"),
            DomainEventSource::new("model-manager"),
            DomainEventKind::new("model.load_failed"),
            Timestamp::new("2026-06-08T12:00:00Z"),
        )
        .with_correlation_id(CorrelationId::new("corr-42"))
        .with_subject(DiagnosticTarget::new(DiagnosticTargetDomain::new("model")).with_id("sd-1.5"))
        .with_diagnostic(diag);

        assert_eq!(ev.id().as_str(), "ev-001");
        assert_eq!(ev.source().as_str(), "model-manager");
        assert_eq!(ev.kind().as_str(), "model.load_failed");
        assert_eq!(ev.created_at().as_str(), "2026-06-08T12:00:00Z");
        assert_eq!(ev.correlation_id().unwrap().as_str(), "corr-42");
        assert_eq!(ev.subject().unwrap().id(), Some("sd-1.5"));
        assert_eq!(ev.diagnostics().len(), 1);
        assert_eq!(ev.diagnostics()[0].code().as_str(), "MODEL/MISSING");
    }

    #[test]
    fn domain_event_minimal() {
        let ev = DomainEvent::new(
            DomainEventId::new("ev-min"),
            DomainEventSource::new("config"),
            DomainEventKind::new("config.loaded"),
            Timestamp::new("2026-01-01T00:00:00Z"),
        );
        assert!(ev.correlation_id().is_none());
        assert!(ev.subject().is_none());
        assert!(ev.diagnostics().is_empty());
    }
}
