//! Agent error types and tool error codes.

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticError, DiagnosticSeverity, DiagnosticSource,
    DiagnosticTarget, DiagnosticTargetDomain, IntoDiagnostic,
};
use reimagine_core::model::DiagnosticId;

use crate::ids::{AgentSessionId, ToolName};

/// Stable, namespaced tool error codes. These are surfaced through the
/// diagnostic bridge so hosts and agents can switch on the code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ToolErrorCode {
    /// Tool name was not present in the registry.
    UnknownTool,
    /// Tool input failed to deserialize into the expected shape.
    InvalidInput,
    /// Tool was invoked in a mode that policy does not allow.
    ModeDenied,
    /// The session did not carry the permission required by the tool.
    PermissionDenied,
    /// The tool is registered as external-risk and a human/host approval
    /// is required before invocation.
    ApprovalRequired,
    /// The tool was invoked without going through policy.
    PolicyBypass,
    /// The concrete tool returned an error during execution.
    ExecutionFailed,
    /// The tool was invoked with a workspace scope that does not match the
    /// session's bound scope.
    WorkspaceMismatch,
}

impl ToolErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnknownTool => "AGENT/TOOL_UNKNOWN",
            Self::InvalidInput => "AGENT/TOOL_INVALID_INPUT",
            Self::ModeDenied => "AGENT/TOOL_MODE_DENIED",
            Self::PermissionDenied => "AGENT/TOOL_PERMISSION_DENIED",
            Self::ApprovalRequired => "AGENT/TOOL_APPROVAL_REQUIRED",
            Self::PolicyBypass => "AGENT/TOOL_POLICY_BYPASS",
            Self::ExecutionFailed => "AGENT/TOOL_EXECUTION_FAILED",
            Self::WorkspaceMismatch => "AGENT/TOOL_WORKSPACE_MISMATCH",
        }
    }
}

impl std::fmt::Display for ToolErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tool-level error returned by the registry when policy or execution
/// fails. The error keeps the tool name and the stable code so hosts can
/// project the failure into a `Diagnostic`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolError {
    code: ToolErrorCode,
    tool: Option<ToolName>,
    message: String,
}

impl ToolError {
    pub fn new(code: ToolErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            tool: None,
            message: message.into(),
        }
    }

    pub fn with_tool(mut self, tool: ToolName) -> Self {
        self.tool = Some(tool);
        self
    }

    pub fn code(&self) -> ToolErrorCode {
        self.code
    }

    pub fn tool(&self) -> Option<&ToolName> {
        self.tool.as_ref()
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn diagnostic_id(&self) -> String {
        match &self.tool {
            Some(name) => format!("agent:tool:{}:{}", name.as_str(), self.code.as_str()),
            None => format!("agent:tool:{}", self.code.as_str()),
        }
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.tool {
            Some(name) => write!(f, "[{}] {}: {}", self.code, name, self.message),
            None => write!(f, "[{}] {}", self.code, self.message),
        }
    }
}

impl std::error::Error for ToolError {}

impl DiagnosticSource for ToolError {
    fn diagnostic_source(&self) -> &'static str {
        "agent"
    }
}

impl DiagnosticError for ToolError {
    fn user_message(&self) -> String {
        self.message.clone()
    }

    fn diagnostic_code(&self) -> DiagnosticCode {
        DiagnosticCode::new(self.code.as_str())
    }

    fn diagnostic_severity(&self) -> DiagnosticSeverity {
        match self.code {
            ToolErrorCode::UnknownTool
            | ToolErrorCode::ModeDenied
            | ToolErrorCode::PermissionDenied
            | ToolErrorCode::PolicyBypass
            | ToolErrorCode::WorkspaceMismatch => DiagnosticSeverity::Error,
            ToolErrorCode::ApprovalRequired => DiagnosticSeverity::Warning,
            ToolErrorCode::InvalidInput | ToolErrorCode::ExecutionFailed => {
                DiagnosticSeverity::Error
            }
        }
    }
}

impl ToolError {
    /// Project this tool error into a core `Diagnostic` with a
    /// session-scoped identity (AC-26). The diagnostic id and target
    /// match the `AgentDomainEventAdapter`'s `ToolFailed` projection
    /// exactly, so both projection paths produce the same identity for
    /// the same failure:
    ///
    /// - id: `agent:{session}:tool:{tool}:{code}` (tool unknown →
    ///   `agent:{session}:tool:{code}`)
    /// - target: `agent.tool` with id `{session}:{tool}`
    pub fn to_diagnostic(
        &self,
        session_id: &AgentSessionId,
        correlation_id: Option<reimagine_core::diagnostic::CorrelationId>,
    ) -> Diagnostic {
        let id = match &self.tool {
            Some(name) => format!(
                "agent:{session_id}:tool:{}:{}",
                name.as_str(),
                self.code.as_str()
            ),
            None => format!("agent:{session_id}:tool:{}", self.code.as_str()),
        };
        let target = match &self.tool {
            Some(name) => DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool"))
                .with_id(format!("{session_id}:{}", name.as_str())),
            None => DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool")),
        };
        self.to_diagnostic_with(DiagnosticId::new(id), target, correlation_id)
    }
}

/// Provider-level error returned by `AgentProvider` implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError {
    provider: Option<crate::ids::ProviderName>,
    code: String,
    message: String,
}

impl ProviderError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            provider: None,
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn with_provider(mut self, provider: crate::ids::ProviderName) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn provider(&self) -> Option<&crate::ids::ProviderName> {
        self.provider.as_ref()
    }

    /// Whether the failure is transient and may succeed on retry
    /// (AC-24). `true` for the transport and timeout codes and for
    /// parseable numeric codes in the retryable HTTP ranges (429 rate
    /// limit, 5xx server errors) — mirroring `ai-protocol`'s
    /// `ProviderAdapterError::is_retryable`, now classified at the
    /// harness level. `code` stays a free string; the classification
    /// is a pure function of it (documented decision).
    pub fn is_transient(&self) -> bool {
        self.code == "TRANSPORT"
            || self.code == "TIMEOUT"
            || self
                .code
                .parse::<u16>()
                .map(|c| c == 429 || (500..600).contains(&c))
                .unwrap_or(false)
    }

    /// Project this provider error into a core `Diagnostic` with a
    /// session-scoped identity (AC-26). The diagnostic id matches the
    /// `AgentDomainEventAdapter`'s `ProviderError` projection exactly:
    ///
    /// - id: `agent:{session}:provider:{provider}:{code}` (provider
    ///   unknown → `agent:{session}:provider:{code}`)
    /// - target: `agent.provider` with id `{session}:{provider}`
    pub fn to_diagnostic(
        &self,
        session_id: &AgentSessionId,
        correlation_id: Option<reimagine_core::diagnostic::CorrelationId>,
    ) -> Diagnostic {
        let id = match &self.provider {
            Some(p) => format!("agent:{session_id}:provider:{}:{}", p.as_str(), self.code),
            None => format!("agent:{session_id}:provider:{}", self.code),
        };
        let target = match &self.provider {
            Some(p) => DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.provider"))
                .with_id(format!("{session_id}:{}", p.as_str())),
            None => DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.provider")),
        };
        let mut diag = Diagnostic::new(
            DiagnosticId::new(id),
            DiagnosticCode::new(format!("AGENT/PROVIDER_{}", self.code)),
            DiagnosticSeverity::Error,
            reimagine_core::diagnostic::DiagnosticSourceName::new("agent"),
            self.message.clone(),
            target,
        );
        if let Some(cid) = correlation_id {
            diag = diag.with_correlation_id(cid);
        }
        diag
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.provider {
            Some(p) => write!(f, "[provider:{}] {}: {}", p, self.code, self.message),
            None => write!(f, "[{}] {}", self.code, self.message),
        }
    }
}

impl std::error::Error for ProviderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_error_display_includes_code_and_tool() {
        let err = ToolError::new(ToolErrorCode::UnknownTool, "not registered")
            .with_tool(ToolName::new("workflow.run"));
        let s = format!("{err}");
        assert!(s.contains("AGENT/TOOL_UNKNOWN"));
        assert!(s.contains("workflow.run"));
    }

    #[test]
    fn tool_error_projects_to_diagnostic() {
        let err = ToolError::new(ToolErrorCode::PermissionDenied, "missing workflow.write")
            .with_tool(ToolName::new("workflow.apply_commands"));
        let session_id = AgentSessionId::new("sess-1");
        let diag = err.to_diagnostic(&session_id, None);
        assert_eq!(diag.code().as_str(), "AGENT/TOOL_PERMISSION_DENIED");
        assert_eq!(diag.severity(), DiagnosticSeverity::Error);
        assert_eq!(diag.source().as_str(), "agent");
        // Session-scoped identity (AC-26): id and target match the
        // `AgentDomainEventAdapter`'s ToolFailed projection exactly.
        assert_eq!(
            diag.id().as_str(),
            "agent:sess-1:tool:workflow.apply_commands:AGENT/TOOL_PERMISSION_DENIED"
        );
        assert_eq!(diag.primary().domain().as_str(), "agent.tool");
        assert_eq!(diag.primary().id(), Some("sess-1:workflow.apply_commands"));
    }

    #[test]
    fn tool_error_without_tool_projects_session_scoped_diagnostic() {
        let err = ToolError::new(ToolErrorCode::UnknownTool, "not registered");
        let session_id = AgentSessionId::new("sess-2");
        let diag = err.to_diagnostic(&session_id, None);
        assert_eq!(diag.id().as_str(), "agent:sess-2:tool:AGENT/TOOL_UNKNOWN");
        assert_eq!(diag.primary().domain().as_str(), "agent.tool");
        assert_eq!(diag.primary().id(), None);
    }

    #[test]
    fn provider_error_projects_to_diagnostic() {
        let err = ProviderError::new("RATE_LIMIT", "slow down")
            .with_provider(crate::ids::ProviderName::new("openai"));
        let session_id = AgentSessionId::new("sess-1");
        let diag = err.to_diagnostic(&session_id, None);
        assert_eq!(diag.code().as_str(), "AGENT/PROVIDER_RATE_LIMIT");
        assert_eq!(diag.source().as_str(), "agent");
        // Session-scoped identity (AC-26): the id matches the
        // `AgentDomainEventAdapter`'s ProviderError projection exactly.
        assert_eq!(
            diag.id().as_str(),
            "agent:sess-1:provider:openai:RATE_LIMIT"
        );
        assert_eq!(diag.primary().domain().as_str(), "agent.provider");
        assert_eq!(diag.primary().id(), Some("sess-1:openai"));
    }

    #[test]
    fn provider_error_transient_classification() {
        // Transport and timeout codes are transient (AC-24).
        assert!(ProviderError::new("TRANSPORT", "connection reset").is_transient());
        assert!(ProviderError::new("TIMEOUT", "slow").is_transient());
        // Parseable numeric codes in the retryable HTTP ranges.
        assert!(ProviderError::new("429", "rate limited").is_transient());
        assert!(ProviderError::new("500", "server error").is_transient());
        assert!(ProviderError::new("503", "unavailable").is_transient());
        assert!(ProviderError::new("599", "proxy error").is_transient());
        // Everything else is permanent.
        assert!(!ProviderError::new("RATE_LIMIT", "slow down").is_transient());
        assert!(!ProviderError::new("404", "not found").is_transient());
        assert!(!ProviderError::new("400", "bad request").is_transient());
        assert!(!ProviderError::new("CONFIGURATION", "bad config").is_transient());
        assert!(!ProviderError::new("SERIALIZATION", "bad payload").is_transient());
        assert!(!ProviderError::new("not_a_number", "nonsense").is_transient());
    }

    #[test]
    fn approval_required_is_warning_severity() {
        let err = ToolError::new(
            ToolErrorCode::ApprovalRequired,
            "build mode requires approval",
        );
        assert_eq!(err.diagnostic_severity(), DiagnosticSeverity::Warning);
    }
}
