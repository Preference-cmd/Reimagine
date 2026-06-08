use reimagine_core::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticError, DiagnosticSeverity,
    DiagnosticSource, DiagnosticTarget, DiagnosticTargetDomain, IntoDiagnostic,
};
use reimagine_core::model::DiagnosticId;

pub type ModelManagerResult<T> = Result<T, ModelManagerError>;

/// User-facing model manager errors. Concrete variants will expand as store,
/// scan, verify, and resolve behavior lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelManagerError {
    ManifestInvalid { message: String },
}

impl ModelManagerError {
    pub fn to_diagnostic(&self, correlation_id: Option<CorrelationId>) -> Diagnostic {
        self.into_diagnostic(
            DiagnosticId::new("model_manager:manifest_invalid"),
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager")),
            correlation_id,
        )
    }
}

impl std::fmt::Display for ModelManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.user_message())
    }
}

impl std::error::Error for ModelManagerError {}

impl DiagnosticSource for ModelManagerError {
    fn diagnostic_source(&self) -> &'static str {
        "model-manager"
    }
}

impl DiagnosticError for ModelManagerError {
    fn user_message(&self) -> String {
        match self {
            Self::ManifestInvalid { message } => format!("model manifest is invalid: {message}"),
        }
    }

    fn diagnostic_code(&self) -> DiagnosticCode {
        match self {
            Self::ManifestInvalid { .. } => DiagnosticCode::new("MODEL_MANAGER/MANIFEST_INVALID"),
        }
    }

    fn diagnostic_severity(&self) -> DiagnosticSeverity {
        DiagnosticSeverity::Error
    }
}
