//! Agent error types and tool error codes.

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticError, DiagnosticSeverity, DiagnosticSource,
    DiagnosticTarget, DiagnosticTargetDomain, IntoDiagnostic,
};
use reimagine_core::model::DiagnosticId;

use crate::ids::ToolName;

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
    /// Project this tool error into a core `Diagnostic`. Uses the
    /// blanket `IntoDiagnostic` impl from `reimagine-core`.
    pub fn to_diagnostic(
        &self,
        correlation_id: Option<reimagine_core::diagnostic::CorrelationId>,
    ) -> Diagnostic {
        let target = match &self.tool {
            Some(name) => DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool"))
                .with_id(name.as_str()),
            None => DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool")),
        };
        self.to_diagnostic_with(
            DiagnosticId::new(self.diagnostic_id()),
            target,
            correlation_id,
        )
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

    pub fn to_diagnostic(
        &self,
        correlation_id: Option<reimagine_core::diagnostic::CorrelationId>,
    ) -> Diagnostic {
        let id = match &self.provider {
            Some(p) => format!("agent:provider:{}:{}", p.as_str(), self.code),
            None => format!("agent:provider:{}", self.code),
        };
        let target = match &self.provider {
            Some(p) => DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.provider"))
                .with_id(p.as_str()),
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

/// Top-level Agent error, used for orchestration errors that do not fit
/// the tool or provider error categories (for example, invalid session
/// construction).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentError {
    code: String,
    message: String,
}

impl AgentError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for AgentError {}

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
        let diag = err.to_diagnostic(None);
        assert_eq!(diag.code().as_str(), "AGENT/TOOL_PERMISSION_DENIED");
        assert_eq!(diag.severity(), DiagnosticSeverity::Error);
        assert_eq!(diag.source().as_str(), "agent");
        assert_eq!(diag.primary().id(), Some("workflow.apply_commands"));
    }

    #[test]
    fn provider_error_projects_to_diagnostic() {
        let err = ProviderError::new("RATE_LIMIT", "slow down")
            .with_provider(crate::ids::ProviderName::new("openai"));
        let diag = err.to_diagnostic(None);
        assert_eq!(diag.code().as_str(), "AGENT/PROVIDER_RATE_LIMIT");
        assert_eq!(diag.source().as_str(), "agent");
        assert_eq!(diag.primary().id(), Some("openai"));
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
