//! Reports for tool and policy outcomes.

use reimagine_core::diagnostic::Diagnostic;

use crate::ids::{AgentSessionId, ToolName};

/// Report for a single tool invocation. Carries the session id, tool
/// name, and any diagnostics produced during the invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolInvocationReport {
    session_id: Option<AgentSessionId>,
    tool: Option<ToolName>,
    diagnostics: Vec<Diagnostic>,
}

impl ToolInvocationReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_session(mut self, session_id: AgentSessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn with_tool(mut self, tool: ToolName) -> Self {
        self.tool = Some(tool);
        self
    }

    pub fn push_diagnostic(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn session_id(&self) -> Option<&AgentSessionId> {
        self.session_id.as_ref()
    }

    pub fn tool(&self) -> Option<&ToolName> {
        self.tool.as_ref()
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

/// Aggregate agent report. Collects per-tool invocation reports and
/// standalone diagnostics produced by orchestration.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AgentReport {
    invocations: Vec<ToolInvocationReport>,
    diagnostics: Vec<Diagnostic>,
}

impl AgentReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_invocation(&mut self, report: ToolInvocationReport) {
        self.invocations.push(report);
    }

    pub fn push_diagnostic(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn extend(&mut self, other: AgentReport) {
        self.invocations.extend(other.invocations);
        self.diagnostics.extend(other.diagnostics);
    }

    pub fn invocations(&self) -> &[ToolInvocationReport] {
        &self.invocations
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn is_empty(&self) -> bool {
        self.invocations.is_empty() && self.diagnostics.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::diagnostic::{
        Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
        DiagnosticTargetDomain,
    };
    use reimagine_core::model::DiagnosticId;

    #[test]
    fn invocation_report_collects_diagnostics() {
        let mut report = ToolInvocationReport::new()
            .with_session(AgentSessionId::new("sess-1"))
            .with_tool(ToolName::new("echo"));
        let diag = Diagnostic::new(
            DiagnosticId::new("d-1"),
            DiagnosticCode::new("AGENT/INFO"),
            DiagnosticSeverity::Info,
            DiagnosticSourceName::new("agent"),
            "ok",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool")).with_id("sess-1:echo"),
        );
        report.push_diagnostic(diag);
        assert_eq!(report.diagnostics().len(), 1);
        assert!(!report.is_empty());
    }

    #[test]
    fn agent_report_extend_merges() {
        let mut a = AgentReport::new();
        a.push_invocation(ToolInvocationReport::new());
        let mut b = AgentReport::new();
        b.push_diagnostic(Diagnostic::new(
            DiagnosticId::new("d-2"),
            DiagnosticCode::new("AGENT/INFO"),
            DiagnosticSeverity::Info,
            DiagnosticSourceName::new("agent"),
            "ok",
            DiagnosticTarget::new(DiagnosticTargetDomain::new("agent")),
        ));
        a.extend(b);
        assert_eq!(a.invocations().len(), 1);
        assert_eq!(a.diagnostics().len(), 1);
    }
}
