use std::path::PathBuf;

use reimagine_core::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticError, DiagnosticSeverity,
    DiagnosticSource, DiagnosticTarget, DiagnosticTargetDomain, IntoDiagnostic,
};
use reimagine_core::model::DiagnosticId;

pub type ModelAcquisitionResult<T> = Result<T, ModelAcquisitionError>;

/// Errors from model acquisition operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelAcquisitionError {
    /// A configuration value is invalid.
    ConfigInvalid { key: String, reason: String },
    /// A path validation check failed.
    PathInvalid { path: String, reason: String },
    /// A staging or promote filesystem operation failed.
    Io { path: PathBuf, message: String },
    /// An hf-hub API call failed.
    Hub { repo: String, message: String },
    /// A JSON serialization or deserialization error.
    Json {
        path: Option<PathBuf>,
        message: String,
    },
    /// A request was invalid or missing required fields.
    InvalidRequest { field: String, message: String },
    /// The staging area exists and overwrite is not permitted.
    StagingExists { path: PathBuf },
    /// The target directory already exists with contents and overwrite is Skip.
    TargetExists { path: PathBuf },
}

impl ModelAcquisitionError {
    pub fn to_diagnostic(&self, correlation_id: Option<CorrelationId>) -> Diagnostic {
        let id = DiagnosticId::new(self.diagnostic_id());
        let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("model-acquisition"))
            .with_id(self.target_id())
            .with_path(self.target_path());
        self.to_diagnostic_with(id, target, correlation_id)
    }

    fn diagnostic_id(&self) -> String {
        match self {
            Self::ConfigInvalid { key, .. } => format!("model-acquisition:config:{key}"),
            Self::PathInvalid { path, .. } => format!("model-acquisition:path:{path}"),
            Self::Io { path, .. } => format!("model-acquisition:io:{}", path.display()),
            Self::Hub { repo, .. } => format!("model-acquisition:hub:{repo}"),
            Self::Json { path: Some(p), .. } => format!("model-acquisition:json:{}", p.display()),
            Self::Json { path: None, .. } => "model-acquisition:json".to_owned(),
            Self::InvalidRequest { field, .. } => format!("model-acquisition:request:{field}"),
            Self::StagingExists { path } => {
                format!("model-acquisition:staging:{}", path.display())
            }
            Self::TargetExists { path } => {
                format!("model-acquisition:target:{}", path.display())
            }
        }
    }

    fn target_id(&self) -> String {
        match self {
            Self::ConfigInvalid { key, .. } => key.clone(),
            Self::PathInvalid { path, .. } => path.clone(),
            Self::Io { path, .. } => path.display().to_string(),
            Self::Hub { repo, .. } => repo.clone(),
            Self::Json { path: Some(p), .. } => p.display().to_string(),
            Self::Json { path: None, .. } => String::new(),
            Self::InvalidRequest { field, .. } => field.clone(),
            Self::StagingExists { path } => path.display().to_string(),
            Self::TargetExists { path } => path.display().to_string(),
        }
    }

    fn target_path(&self) -> String {
        match self {
            Self::Io { path, .. } | Self::StagingExists { path } | Self::TargetExists { path } => {
                path.display().to_string()
            }
            Self::Json { path: Some(p), .. } => p.display().to_string(),
            _ => String::new(),
        }
    }
}

impl std::fmt::Display for ModelAcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.user_message())
    }
}

impl std::error::Error for ModelAcquisitionError {}

impl DiagnosticSource for ModelAcquisitionError {
    fn diagnostic_source(&self) -> &'static str {
        "model-acquisition"
    }
}

impl DiagnosticError for ModelAcquisitionError {
    fn user_message(&self) -> String {
        match self {
            Self::ConfigInvalid { key, reason } => {
                format!("model-acquisition config `{key}` is invalid: {reason}")
            }
            Self::PathInvalid { path, reason } => {
                format!("path `{path}` is invalid: {reason}")
            }
            Self::Io { path, message } => {
                format!("filesystem error at `{}`: {message}", path.display())
            }
            Self::Hub { repo, message } => {
                format!("HuggingFace hub error for `{repo}`: {message}")
            }
            Self::Json {
                path: Some(p),
                message,
            } => {
                format!("JSON error at `{}`: {message}", p.display())
            }
            Self::Json {
                path: None,
                message,
            } => format!("JSON error: {message}"),
            Self::InvalidRequest { field, message } => {
                format!("invalid request field `{field}`: {message}")
            }
            Self::StagingExists { path } => {
                format!("staging directory already exists at `{}`", path.display())
            }
            Self::TargetExists { path } => {
                format!("target directory already exists at `{}`", path.display())
            }
        }
    }

    fn diagnostic_code(&self) -> DiagnosticCode {
        match self {
            Self::ConfigInvalid { .. } => DiagnosticCode::new("MODEL_ACQUISITION/CONFIG_INVALID"),
            Self::PathInvalid { .. } => DiagnosticCode::new("MODEL_ACQUISITION/PATH_INVALID"),
            Self::Io { .. } => DiagnosticCode::new("MODEL_ACQUISITION/IO_ERROR"),
            Self::Hub { .. } => DiagnosticCode::new("MODEL_ACQUISITION/HUB_ERROR"),
            Self::Json { .. } => DiagnosticCode::new("MODEL_ACQUISITION/JSON_ERROR"),
            Self::InvalidRequest { .. } => DiagnosticCode::new("MODEL_ACQUISITION/INVALID_REQUEST"),
            Self::StagingExists { .. } => DiagnosticCode::new("MODEL_ACQUISITION/STAGING_EXISTS"),
            Self::TargetExists { .. } => DiagnosticCode::new("MODEL_ACQUISITION/TARGET_EXISTS"),
        }
    }

    fn diagnostic_severity(&self) -> DiagnosticSeverity {
        DiagnosticSeverity::Error
    }
}
