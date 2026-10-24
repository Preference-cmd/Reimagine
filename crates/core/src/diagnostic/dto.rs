//! Diagnostic transfer object.

use super::Diagnostic;
use serde::{Deserialize, Serialize};

/// Diagnostic JSON projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticDto {
    pub id: String,
    pub code: String,
    pub severity: String,
    pub source: String,
    pub message: String,
    pub target: String,
}

impl From<&Diagnostic> for DiagnosticDto {
    fn from(value: &Diagnostic) -> Self {
        let target = value.primary();
        Self {
            id: value.id().as_str().to_string(),
            code: value.code().as_str().to_string(),
            severity: value.severity().to_string(),
            source: value.source().to_string(),
            message: value.message().to_string(),
            target: format!("{}:{}", target.domain(), target.path().unwrap_or("")),
        }
    }
}

impl From<Diagnostic> for DiagnosticDto {
    fn from(value: Diagnostic) -> Self {
        Self::from(&value)
    }
}
