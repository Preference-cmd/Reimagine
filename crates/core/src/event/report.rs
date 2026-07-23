use crate::diagnostic::Diagnostic;

use super::domain_event::DomainEvent;

/// Synchronous operation return envelope.
///
/// `OperationReport` collects diagnostics and domain events produced by one
/// synchronous operation. It is **not** the event-bus message type; if a
/// future event bus exists, callers may emit each [`DomainEvent`] from a
/// report individually.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OperationReport {
    diagnostics: Vec<Diagnostic>,
    events: Vec<DomainEvent>,
}

impl OperationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a report seeded with a single diagnostic.
    pub fn with_diagnostic(diagnostic: Diagnostic) -> Self {
        Self {
            diagnostics: vec![diagnostic],
            events: Vec::new(),
        }
    }

    /// Create a report seeded with a single event.
    pub fn with_event(event: DomainEvent) -> Self {
        Self {
            diagnostics: Vec::new(),
            events: vec![event],
        }
    }

    /// Append a diagnostic.
    pub fn push_diagnostic(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Append a domain event.
    pub fn push_event(&mut self, event: DomainEvent) {
        self.events.push(event);
    }

    /// Merge another report into this one.
    pub fn extend(&mut self, other: OperationReport) {
        self.diagnostics.extend(other.diagnostics);
        self.events.extend(other.events);
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn events(&self) -> &[DomainEvent] {
        &self.events
    }

    /// Returns `true` when the report contains no diagnostics or events.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty() && self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::{
        CorrelationId, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
        DiagnosticTargetDomain,
    };
    use crate::event::domain_event::{DomainEventId, Timestamp};
    use crate::event::{DomainEventKind, DomainEventSource};
    use crate::model::DiagnosticId;

    /// Config-style report: a warning diagnostic with no event.
    #[test]
    fn config_style_report() {
        let diag = Diagnostic::new(
            DiagnosticId::new("cfg-d-001"),
            DiagnosticCode::new("CONFIG/DEPRECATED_FIELD"),
            DiagnosticSeverity::Warning,
            DiagnosticSourceName::new("config"),
            "field 'legacy_mode' is deprecated",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                .with_path("~/.reimagine/config.toml"),
        );

        let report = OperationReport::with_diagnostic(diag);

        assert_eq!(report.diagnostics().len(), 1);
        assert!(report.events().is_empty());
        assert!(!report.is_empty());
        assert_eq!(
            report.diagnostics()[0].code().as_str(),
            "CONFIG/DEPRECATED_FIELD"
        );
    }

    /// Model-manager-style report: an event carrying a diagnostic.
    #[test]
    fn model_manager_style_report() {
        let cid = CorrelationId::new("corr-mm-1");

        let diag = Diagnostic::new(
            DiagnosticId::new("mm-d-001"),
            DiagnosticCode::new("MODEL/CORRUPT_WEIGHTS"),
            DiagnosticSeverity::Error,
            DiagnosticSourceName::new("model-manager"),
            "safetensors checksum mismatch",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model"))
                .with_id("sdxl-base")
                .with_path("/models/sdxl-base.safetensors"),
        )
        .with_correlation_id(cid.clone());

        let ev = DomainEvent::new(
            DomainEventId::new("mm-ev-001"),
            DomainEventSource::new("model-manager"),
            DomainEventKind::new("model.load_failed"),
            Timestamp::new("2026-06-08T15:30:00Z"),
        )
        .with_correlation_id(cid)
        .with_subject(
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model")).with_id("sdxl-base"),
        )
        .with_diagnostic(diag);

        let mut report = OperationReport::with_event(ev);
        // The diagnostic lives inside the event, not at the report level.
        assert_eq!(report.diagnostics().len(), 0);
        assert_eq!(report.events().len(), 1);
        assert_eq!(report.events()[0].diagnostics().len(), 1);
        assert_eq!(
            report.events()[0].diagnostics()[0].severity(),
            DiagnosticSeverity::Error
        );

        // Push an additional standalone diagnostic.
        report.push_diagnostic(Diagnostic::new(
            DiagnosticId::new("mm-d-002"),
            DiagnosticCode::new("MODEL/STALE_CACHE"),
            DiagnosticSeverity::Info,
            DiagnosticSourceName::new("model-manager"),
            "cached weights are older than the registry entry",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model")).with_id("sdxl-base"),
        ));
        assert_eq!(report.diagnostics().len(), 1);
        assert_eq!(report.events().len(), 1);
    }

    /// OperationReport::extend merges two reports.
    #[test]
    fn report_extend() {
        let r1 = OperationReport::with_diagnostic(Diagnostic::new(
            DiagnosticId::new("ext-1"),
            DiagnosticCode::new("TEST/A"),
            DiagnosticSeverity::Info,
            DiagnosticSourceName::new("test"),
            "first",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("test")),
        ));

        let r2 = OperationReport::with_event(DomainEvent::new(
            DomainEventId::new("ext-ev"),
            DomainEventSource::new("test"),
            DomainEventKind::new("test.happened"),
            Timestamp::new("2026-06-08T00:00:00Z"),
        ));

        let mut combined = r1;
        combined.extend(r2);
        assert_eq!(combined.diagnostics().len(), 1);
        assert_eq!(combined.events().len(), 1);
    }

    #[test]
    fn empty_report_is_empty() {
        let r = OperationReport::new();
        assert!(r.is_empty());
    }
}
