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
    ManifestInvalid { path: String, message: String },
    ReadFailed { path: String, message: String },
    WriteFailed { path: String, message: String },
}

impl ModelManagerError {
    pub fn to_diagnostic(&self, correlation_id: Option<CorrelationId>) -> Diagnostic {
        self.into_diagnostic(
            DiagnosticId::new(self.diagnostic_id()),
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model-manager"))
                .with_path(self.target_path()),
            correlation_id,
        )
    }

    fn diagnostic_id(&self) -> String {
        match self {
            Self::ManifestInvalid { path, .. } => {
                format!("model_manager:{path}:manifest_invalid")
            }
            Self::ReadFailed { path, .. } => format!("model_manager:{path}:read_failed"),
            Self::WriteFailed { path, .. } => format!("model_manager:{path}:write_failed"),
        }
    }

    fn target_path(&self) -> String {
        match self {
            Self::ManifestInvalid { path, .. }
            | Self::ReadFailed { path, .. }
            | Self::WriteFailed { path, .. } => path.clone(),
        }
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
            Self::ManifestInvalid { path, message } => {
                format!("model manifest `{path}` is invalid: {message}")
            }
            Self::ReadFailed { path, message } => {
                format!("failed to read model manifest `{path}`: {message}")
            }
            Self::WriteFailed { path, message } => {
                format!("failed to write model manifest `{path}`: {message}")
            }
        }
    }

    fn diagnostic_code(&self) -> DiagnosticCode {
        match self {
            Self::ManifestInvalid { .. } => DiagnosticCode::new("MODEL_MANAGER/MANIFEST_INVALID"),
            Self::ReadFailed { .. } => DiagnosticCode::new("MODEL_MANAGER/READ_FAILED"),
            Self::WriteFailed { .. } => DiagnosticCode::new("MODEL_MANAGER/WRITE_FAILED"),
        }
    }

    fn diagnostic_severity(&self) -> DiagnosticSeverity {
        DiagnosticSeverity::Error
    }
}
