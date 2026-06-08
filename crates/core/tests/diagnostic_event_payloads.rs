use reimagine_core::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticError, DiagnosticFixHint,
    DiagnosticRelated, DiagnosticSeverity, DiagnosticSource, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain, IntoDiagnostic,
};
use reimagine_core::event::{
    DomainEvent, DomainEventId, DomainEventKind, DomainEventSource, OperationReport, Timestamp,
};
use reimagine_core::model::DiagnosticId;

#[test]
fn config_style_diagnostic_report_is_available_from_public_facades() {
    let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("config.file"))
        .with_path("model_series.json");

    let diagnostic = Diagnostic::new(
        DiagnosticId::new("diag-config-json-invalid"),
        DiagnosticCode::new("CONFIG_JSON_INVALID"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("config"),
        "Config file contains invalid JSON",
        target,
    )
    .with_correlation_id(CorrelationId::new("corr-config-load"))
    .with_fix(
        DiagnosticFixHint::new("Open config file")
            .with_description("Fix the JSON syntax and save the file again")
            .with_requires_confirmation(false),
    );

    let report = OperationReport::with_diagnostic(diagnostic);

    assert_eq!(report.diagnostics().len(), 1);
    assert!(report.events().is_empty());
    assert_eq!(
        report.diagnostics()[0].code().as_str(),
        "CONFIG_JSON_INVALID"
    );
    assert_eq!(
        report.diagnostics()[0].correlation_id().unwrap().as_str(),
        "corr-config-load"
    );
}

#[test]
fn model_manager_style_event_can_carry_diagnostics_from_public_facades() {
    let correlation_id = CorrelationId::new("corr-model-scan");
    let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("model"))
        .with_id("sdxl-base")
        .with_path("checkpoints/sdxl_base_1.0.safetensors");

    let diagnostic = Diagnostic::new(
        DiagnosticId::new("diag-model-stale"),
        DiagnosticCode::new("MODEL_MARKED_STALE"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("model-manager"),
        "Model file metadata changed since verification",
        target.clone(),
    )
    .with_correlation_id(correlation_id.clone())
    .with_related(DiagnosticRelated::new(
        DiagnosticTarget::new(DiagnosticTargetDomain::new("model.root")).with_id("base"),
        "Discovered during model root scan",
    ));

    let event = DomainEvent::new(
        DomainEventId::new("event-model-stale"),
        DomainEventSource::new("model-manager"),
        DomainEventKind::new("model.marked_stale"),
        Timestamp::new("2026-06-08T00:00:00Z"),
    )
    .with_correlation_id(correlation_id)
    .with_subject(target)
    .with_diagnostic(diagnostic);

    let report = OperationReport::with_event(event);

    assert!(report.diagnostics().is_empty());
    assert_eq!(report.events().len(), 1);
    assert_eq!(report.events()[0].kind().as_str(), "model.marked_stale");
    assert_eq!(report.events()[0].diagnostics().len(), 1);
    assert_eq!(
        report.events()[0].diagnostics()[0].primary().id(),
        Some("sdxl-base")
    );
}

#[derive(Debug)]
enum ExampleConfigError {
    InvalidJson,
}

impl DiagnosticSource for ExampleConfigError {
    fn diagnostic_source(&self) -> &'static str {
        "config"
    }
}

impl DiagnosticError for ExampleConfigError {
    fn user_message(&self) -> String {
        "Config JSON could not be parsed".to_owned()
    }

    fn diagnostic_code(&self) -> DiagnosticCode {
        DiagnosticCode::new("CONFIG_JSON_INVALID")
    }

    fn diagnostic_severity(&self) -> DiagnosticSeverity {
        DiagnosticSeverity::Error
    }
}

#[test]
fn service_errors_opt_into_diagnostic_conversion() {
    let diagnostic = ExampleConfigError::InvalidJson.into_diagnostic(
        DiagnosticId::new("diag-from-error"),
        DiagnosticTarget::new(DiagnosticTargetDomain::new("config.file"))
            .with_path("model_series.json"),
        Some(CorrelationId::new("corr-error")),
    );

    assert_eq!(diagnostic.source().as_str(), "config");
    assert_eq!(diagnostic.code().as_str(), "CONFIG_JSON_INVALID");
    assert_eq!(diagnostic.severity(), DiagnosticSeverity::Error);
    assert_eq!(diagnostic.primary().path(), Some("model_series.json"));
}
