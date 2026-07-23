use crate::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use crate::model::{DiagnosticId, EdgeId, NodeId, WorkflowId};

const SOURCE: &str = "core";

pub(super) fn workflow_diagnostic(
    suffix: &str,
    workflow_id: &WorkflowId,
    code: &str,
    message: &str,
    path: Option<String>,
) -> Diagnostic {
    let target = target("workflow", workflow_id.as_str(), path);
    diagnostic(suffix, code, message, target)
}

pub(super) fn node_diagnostic(
    suffix: &str,
    node_id: &NodeId,
    code: &str,
    message: &str,
    path: Option<String>,
) -> Diagnostic {
    let target = target("workflow.node", node_id.as_str(), path);
    diagnostic(suffix, code, message, target)
}

pub(super) fn edge_diagnostic(
    suffix: &str,
    edge_id: &EdgeId,
    code: &str,
    message: &str,
    path: Option<String>,
) -> Diagnostic {
    let target = target("workflow.edge", edge_id.as_str(), path);
    diagnostic(suffix, code, message, target)
}

fn target(domain: &str, id: &str, path: Option<String>) -> DiagnosticTarget {
    let target = DiagnosticTarget::new(DiagnosticTargetDomain::new(domain)).with_id(id.to_owned());
    if let Some(path) = path {
        target.with_path(path)
    } else {
        target
    }
}

fn diagnostic(suffix: &str, code: &str, message: &str, target: DiagnosticTarget) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("core:workflow:{suffix}")),
        DiagnosticCode::new(code),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new(SOURCE),
        message,
        target,
    )
}
