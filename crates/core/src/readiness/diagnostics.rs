use crate::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use crate::model::{DiagnosticId, NodeId, SlotId, WorkflowId};

use super::external::{ExternalReadinessContext, ExternalReadinessSubject};

const SOURCE: &str = "core";

pub fn no_target(workflow_id: &WorkflowId) -> Diagnostic {
    workflow_diagnostic(
        "no_target",
        workflow_id,
        "CORE/WORKFLOW_NO_TARGET",
        "workflow has no executable run target",
        None,
    )
}

pub fn non_contributing_pure_graph(workflow_id: &WorkflowId) -> Diagnostic {
    workflow_diagnostic(
        "non_contributing_pure_graph",
        workflow_id,
        "CORE/WORKFLOW_NON_CONTRIBUTING_PURE_GRAPH",
        "workflow contains pure nodes that do not contribute to any run target",
        None,
    )
}

pub fn target_invalid(node_id: &NodeId) -> Diagnostic {
    node_diagnostic(
        "target_invalid",
        node_id,
        "CORE/WORKFLOW_TARGET_INVALID",
        "selected run target is not valid for execution",
        None,
    )
}

pub fn workflow_target_invalid(workflow_id: &WorkflowId, path: impl Into<String>) -> Diagnostic {
    workflow_diagnostic(
        "target_invalid",
        workflow_id,
        "CORE/WORKFLOW_TARGET_INVALID",
        "selected run target is not valid for execution",
        Some(path.into()),
    )
}

pub fn required_input_missing(node_id: &NodeId, slot_id: &SlotId) -> Diagnostic {
    node_diagnostic(
        "required_input_missing",
        node_id,
        "CORE/WORKFLOW_REQUIRED_INPUT_MISSING",
        "required effective input is missing",
        Some(format!("inputs.{}", slot_id.as_str())),
    )
}

pub fn executable_cycle(workflow_id: &WorkflowId) -> Diagnostic {
    workflow_diagnostic(
        "executable_cycle",
        workflow_id,
        "CORE/WORKFLOW_EXECUTABLE_CYCLE",
        "execution subgraph contains a cycle",
        None,
    )
}

pub fn external_readiness_missing(
    context: &ExternalReadinessContext,
    _subject: &ExternalReadinessSubject,
) -> Diagnostic {
    let mut target = DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow"))
        .with_id(context.workflow_id().as_str().to_owned())
        .with_path(context.path().to_owned());

    if let Some(node_id) = context.node_id() {
        target = DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow.node"))
            .with_id(node_id.as_str().to_owned())
            .with_path(context.path().to_owned());
    }

    Diagnostic::new(
        DiagnosticId::new("core:workflow:external_readiness_missing"),
        DiagnosticCode::new("CORE/WORKFLOW_EXTERNAL_READINESS_MISSING"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new(SOURCE),
        "external readiness subject is required but missing from the provider snapshot",
        target,
    )
}

fn workflow_diagnostic(
    suffix: &str,
    workflow_id: &WorkflowId,
    code: &str,
    message: &str,
    path: Option<String>,
) -> Diagnostic {
    diagnostic(
        suffix,
        code,
        message,
        target("workflow", workflow_id.as_str(), path),
    )
}

fn node_diagnostic(
    suffix: &str,
    node_id: &NodeId,
    code: &str,
    message: &str,
    path: Option<String>,
) -> Diagnostic {
    diagnostic(
        suffix,
        code,
        message,
        target("workflow.node", node_id.as_str(), path),
    )
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
