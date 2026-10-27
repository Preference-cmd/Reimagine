//! Subsystem health and availability.

use crate::diagnostic::Diagnostic;

/// Whether a subsystem can serve requests. The reason it cannot is a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum Health {
    Available,
    Unavailable(Diagnostic),
}

impl Health {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    pub fn diagnostic(&self) -> Option<&Diagnostic> {
        match self {
            Self::Available => None,
            Self::Unavailable(diag) => Some(diag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::{
        DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
        DiagnosticTargetDomain,
    };
    use crate::model::DiagnosticId;

    #[test]
    fn health_available() {
        let health = Health::Available;
        assert!(health.is_available());
        assert!(health.diagnostic().is_none());
    }

    #[test]
    fn health_unavailable() {
        let diag = Diagnostic::new(
            DiagnosticId::new("test:diag"),
            DiagnosticCode::new("TEST/UNAVAILABLE"),
            DiagnosticSeverity::Error,
            DiagnosticSourceName::new("test"),
            "subsystem down",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("test")),
        );
        let health = Health::Unavailable(diag.clone());
        assert!(!health.is_available());
        assert_eq!(health.diagnostic(), Some(&diag));
    }
}
