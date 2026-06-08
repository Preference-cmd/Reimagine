//! Diagnostic payloads: host-neutral, actionable explanations for humans and agents.

use crate::model::DiagnosticId;

#[path = "diagnostic/code.rs"]
mod code;
#[path = "diagnostic/fix.rs"]
mod fix;
#[path = "diagnostic/related.rs"]
mod related;
#[path = "diagnostic/severity.rs"]
mod severity;
#[path = "diagnostic/target.rs"]
mod target;

pub use code::DiagnosticCode;
pub use fix::DiagnosticFixHint;
pub use related::DiagnosticRelated;
pub use severity::DiagnosticSeverity;
pub use target::{DiagnosticTarget, DiagnosticTargetDomain};

/// Extensible newtype for the name of the subsystem that produced a diagnostic
/// (e.g. "config", "model-manager", "runtime").
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticSourceName(String);

impl DiagnosticSourceName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DiagnosticSourceName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DiagnosticSourceName {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DiagnosticSourceName {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Caller-supplied correlation id, shared between diagnostics and domain events.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CorrelationId(String);

impl CorrelationId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CorrelationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for CorrelationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for CorrelationId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// An actionable diagnostic that a host (Tauri, Axum, CLI) can present to a
/// user or agent.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    id: DiagnosticId,
    correlation_id: Option<CorrelationId>,
    trace_span_id: Option<String>,
    code: DiagnosticCode,
    severity: DiagnosticSeverity,
    source: DiagnosticSourceName,
    message: String,
    primary: DiagnosticTarget,
    related: Vec<DiagnosticRelated>,
    fixes: Vec<DiagnosticFixHint>,
}

impl Diagnostic {
    /// Build a diagnostic with the required fields.
    pub fn new(
        id: DiagnosticId,
        code: DiagnosticCode,
        severity: DiagnosticSeverity,
        source: DiagnosticSourceName,
        message: impl Into<String>,
        primary: DiagnosticTarget,
    ) -> Self {
        Self {
            id,
            correlation_id: None,
            trace_span_id: None,
            code,
            severity,
            source,
            message: message.into(),
            primary,
            related: Vec::new(),
            fixes: Vec::new(),
        }
    }

    /// Attach a correlation id so this diagnostic can be grouped with domain
    /// events from the same operation.
    pub fn with_correlation_id(mut self, id: CorrelationId) -> Self {
        self.correlation_id = Some(id);
        self
    }

    /// Attach an optional trace span id for observability tooling.
    pub fn with_trace_span_id(mut self, span_id: impl Into<String>) -> Self {
        self.trace_span_id = Some(span_id.into());
        self
    }

    /// Add a related target.
    pub fn with_related(mut self, related: DiagnosticRelated) -> Self {
        self.related.push(related);
        self
    }

    /// Add a fix hint.
    pub fn with_fix(mut self, fix: DiagnosticFixHint) -> Self {
        self.fixes.push(fix);
        self
    }

    pub fn id(&self) -> &DiagnosticId {
        &self.id
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn trace_span_id(&self) -> Option<&str> {
        self.trace_span_id.as_deref()
    }

    pub fn code(&self) -> &DiagnosticCode {
        &self.code
    }

    pub fn severity(&self) -> DiagnosticSeverity {
        self.severity
    }

    pub fn source(&self) -> &DiagnosticSourceName {
        &self.source
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn primary(&self) -> &DiagnosticTarget {
        &self.primary
    }

    pub fn related(&self) -> &[DiagnosticRelated] {
        &self.related
    }

    pub fn fixes(&self) -> &[DiagnosticFixHint] {
        &self.fixes
    }
}

/// Trait for types that identify the subsystem they come from.
pub trait DiagnosticSource {
    fn diagnostic_source(&self) -> &'static str;
}

/// Trait for errors that can be presented as user/agent-facing diagnostics.
pub trait DiagnosticError: DiagnosticSource {
    fn user_message(&self) -> String;
    fn diagnostic_code(&self) -> DiagnosticCode;
    fn diagnostic_severity(&self) -> DiagnosticSeverity;
}

/// Convenience conversion from a `DiagnosticError` into a concrete
/// [`Diagnostic`], supplying the id, target, and optional correlation id that
/// the caller owns.
pub trait IntoDiagnostic {
    fn into_diagnostic(
        &self,
        id: DiagnosticId,
        target: DiagnosticTarget,
        correlation_id: Option<CorrelationId>,
    ) -> Diagnostic;
}

/// Blanket implementation for any type that implements `DiagnosticError`.
impl<T: DiagnosticError> IntoDiagnostic for T {
    fn into_diagnostic(
        &self,
        id: DiagnosticId,
        target: DiagnosticTarget,
        correlation_id: Option<CorrelationId>,
    ) -> Diagnostic {
        let mut diag = Diagnostic::new(
            id,
            self.diagnostic_code(),
            self.diagnostic_severity(),
            DiagnosticSourceName::new(self.diagnostic_source()),
            self.user_message(),
            target,
        );
        if let Some(cid) = correlation_id {
            diag = diag.with_correlation_id(cid);
        }
        diag
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DiagnosticId;

    #[test]
    fn diagnostic_builder_roundtrip() {
        let diag = Diagnostic::new(
            DiagnosticId::new("d-001"),
            DiagnosticCode::new("CONFIG/MISSING"),
            DiagnosticSeverity::Warning,
            DiagnosticSourceName::new("config"),
            "config file not found",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
                .with_path("~/.reimagine/config.toml"),
        )
        .with_correlation_id(CorrelationId::new("corr-1"))
        .with_trace_span_id("span-42")
        .with_related(DiagnosticRelated::new(
            DiagnosticTarget::new(DiagnosticTargetDomain::new("model")).with_id("sd-1.5"),
            "required by active workflow",
        ))
        .with_fix(
            DiagnosticFixHint::new("create default config")
                .with_description("Generates a starter config.toml")
                .with_requires_confirmation(false),
        );

        assert_eq!(diag.id().as_str(), "d-001");
        assert_eq!(diag.code().as_str(), "CONFIG/MISSING");
        assert_eq!(diag.severity(), DiagnosticSeverity::Warning);
        assert_eq!(diag.source().as_str(), "config");
        assert_eq!(diag.message(), "config file not found");
        assert_eq!(
            diag.primary().path(),
            Some("~/.reimagine/config.toml")
        );
        assert_eq!(diag.correlation_id().unwrap().as_str(), "corr-1");
        assert_eq!(diag.trace_span_id(), Some("span-42"));
        assert_eq!(diag.related().len(), 1);
        assert_eq!(diag.related()[0].message(), "required by active workflow");
        assert_eq!(diag.fixes().len(), 1);
        assert_eq!(diag.fixes()[0].label(), "create default config");
    }

    // Simulated config-style error.

    #[derive(Debug)]
    enum ConfigError {
        FileNotFound(String),
    }

    impl std::fmt::Display for ConfigError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::FileNotFound(p) => write!(f, "config file not found: {p}"),
            }
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
                Self::FileNotFound(p) => format!("Configuration file not found at {p}"),
            }
        }

        fn diagnostic_code(&self) -> DiagnosticCode {
            DiagnosticCode::new("CONFIG/FILE_NOT_FOUND")
        }

        fn diagnostic_severity(&self) -> DiagnosticSeverity {
            DiagnosticSeverity::Error
        }
    }

    #[test]
    fn config_error_into_diagnostic() {
        let err = ConfigError::FileNotFound("~/.reimagine/config.toml".into());
        let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("config"))
            .with_path("~/.reimagine/config.toml");
        let diag = err.into_diagnostic(
            DiagnosticId::new("cfg-001"),
            target,
            Some(CorrelationId::new("corr-config")),
        );

        assert_eq!(diag.code().as_str(), "CONFIG/FILE_NOT_FOUND");
        assert_eq!(diag.severity(), DiagnosticSeverity::Error);
        assert_eq!(diag.source().as_str(), "config");
        assert!(diag.message().contains("not found"));
        assert_eq!(
            diag.correlation_id().unwrap().as_str(),
            "corr-config"
        );
    }

    // Simulated model-manager-style error.

    #[derive(Debug)]
    enum ModelManagerError {
        LoadFailed { model_id: String, reason: String },
    }

    impl std::fmt::Display for ModelManagerError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::LoadFailed { model_id, reason } => {
                    write!(f, "failed to load model {model_id}: {reason}")
                }
            }
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
                Self::LoadFailed { model_id, reason } => {
                    format!("Could not load model \"{model_id}\": {reason}")
                }
            }
        }

        fn diagnostic_code(&self) -> DiagnosticCode {
            DiagnosticCode::new("MODEL/LOAD_FAILED")
        }

        fn diagnostic_severity(&self) -> DiagnosticSeverity {
            DiagnosticSeverity::Error
        }
    }

    #[test]
    fn model_manager_error_into_diagnostic() {
        let err = ModelManagerError::LoadFailed {
            model_id: "sd-1.5".into(),
            reason: "weights file missing".into(),
        };
        let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("model"))
            .with_id("sd-1.5")
            .with_path("/models/sd-1.5.safetensors");
        let diag = err.into_diagnostic(
            DiagnosticId::new("mm-001"),
            target,
            Some(CorrelationId::new("corr-model")),
        );

        assert_eq!(diag.code().as_str(), "MODEL/LOAD_FAILED");
        assert_eq!(diag.source().as_str(), "model-manager");
        assert!(diag.message().contains("sd-1.5"));
        assert!(diag.message().contains("weights file missing"));
    }

    // Ordinary infra errors are NOT diagnostics.

    #[test]
    fn ordinary_io_error_is_not_a_diagnostic() {
        // An ordinary std::io::Error does not implement DiagnosticError and
        // therefore cannot be converted via IntoDiagnostic. Only errors that
        // explicitly opt in become diagnostics.
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        // We can still inspect it as a normal error.
        assert!(io_err.to_string().contains("gone"));
        // But there is no blanket impl for std::error::Error to Diagnostic.
    }
}
