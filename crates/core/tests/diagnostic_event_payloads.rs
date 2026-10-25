use reimagine_core::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticError, DiagnosticFixHint,
    DiagnosticSeverity, DiagnosticSource, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain, IntoDiagnostic,
};
use reimagine_core::model::DiagnosticId;

#[test]
fn config_style_diagnostic_payload_is_available_from_public_facades() {
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

    assert_eq!(diagnostic.code().as_str(), "CONFIG_JSON_INVALID");
    assert_eq!(
        diagnostic.correlation_id().unwrap().as_str(),
        "corr-config-load"
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
    let diagnostic = ExampleConfigError::InvalidJson.to_diagnostic_with(
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
