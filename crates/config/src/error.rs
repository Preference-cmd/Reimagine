use std::path::PathBuf;

use reimagine_core::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticError, DiagnosticSeverity,
    DiagnosticSource, DiagnosticTarget, DiagnosticTargetDomain, IntoDiagnostic,
};
use reimagine_core::model::DiagnosticId;

pub type ConfigResult<T> = Result<T, ConfigError>;

/// Infrastructure errors from config key, JSON, and filesystem operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    PathInvalid {
        key: String,
        reason: String,
    },
    JsonInvalid {
        key: Option<String>,
        path: PathBuf,
        message: String,
    },
    ReadFailed {
        path: PathBuf,
        message: String,
    },
    WriteFailed {
        path: PathBuf,
        message: String,
    },
}

impl ConfigError {
    pub fn to_diagnostic(&self, correlation_id: Option<CorrelationId>) -> Diagnostic {
        let id = DiagnosticId::new(self.diagnostic_id());
        let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
            .with_id(self.target_id())
            .with_path(self.target_path());
        self.to_diagnostic_with(id, target, correlation_id)
    }

    fn diagnostic_id(&self) -> String {
        match self {
            Self::PathInvalid { key, .. } => format!("config:{key}:path_invalid"),
            Self::JsonInvalid { key, path, .. } => {
                format!(
                    "config:{}:json_invalid",
                    key.as_deref()
                        .map_or_else(|| path.display().to_string(), ToOwned::to_owned)
                )
            }
            Self::ReadFailed { path, .. } => {
                format!("config:{}:read_failed", path.display())
            }
            Self::WriteFailed { path, .. } => {
                format!("config:{}:write_failed", path.display())
            }
        }
    }

    fn target_id(&self) -> String {
        match self {
            Self::PathInvalid { key, .. } => key.clone(),
            Self::JsonInvalid { key: Some(key), .. } => key.clone(),
            Self::JsonInvalid {
                key: None, path, ..
            }
            | Self::ReadFailed { path, .. }
            | Self::WriteFailed { path, .. } => path.display().to_string(),
        }
    }

    fn target_path(&self) -> String {
        match self {
            Self::PathInvalid { key, .. } => key.clone(),
            Self::JsonInvalid { path, .. }
            | Self::ReadFailed { path, .. }
            | Self::WriteFailed { path, .. } => path.display().to_string(),
        }
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.user_message())
    }
}

impl std::error::Error for ConfigError {}

impl DiagnosticSource for ConfigError {
    fn diagnostic_source(&self) -> &'static str {
        "config"
    }
}

impl DiagnosticError for ConfigError {
    fn user_message(&self) -> String {
        match self {
            Self::PathInvalid { key, reason } => {
                format!("config key `{key}` is invalid: {reason}")
            }
            Self::JsonInvalid { path, message, .. } => {
                format!(
                    "config file `{}` is not valid JSON: {message}",
                    path.display()
                )
            }
            Self::ReadFailed { path, message } => {
                format!("failed to read config file `{}`: {message}", path.display())
            }
            Self::WriteFailed { path, message } => {
                format!(
                    "failed to write config file `{}`: {message}",
                    path.display()
                )
            }
        }
    }

    fn diagnostic_code(&self) -> DiagnosticCode {
        match self {
            Self::PathInvalid { .. } => DiagnosticCode::new("CONFIG/PATH_INVALID"),
            Self::JsonInvalid { .. } => DiagnosticCode::new("CONFIG/JSON_INVALID"),
            Self::ReadFailed { .. } => DiagnosticCode::new("CONFIG/READ_FAILED"),
            Self::WriteFailed { .. } => DiagnosticCode::new("CONFIG/WRITE_FAILED"),
        }
    }

    fn diagnostic_severity(&self) -> DiagnosticSeverity {
        DiagnosticSeverity::Error
    }
}
