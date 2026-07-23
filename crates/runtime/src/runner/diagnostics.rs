use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::{NodeId, RunId};

pub(super) fn make_diagnostic(run_id: &RunId, node_id: &NodeId, message: &str) -> Diagnostic {
    let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow.node"))
        .with_id(node_id.as_str())
        .with_path(run_id.as_str());
    Diagnostic::new(
        reimagine_core::model::DiagnosticId::new(format!(
            "runtime-{}-{}",
            run_id.as_str(),
            node_id.as_str()
        )),
        DiagnosticCode::new("RUNTIME/NODE_EXECUTION_FAILED"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("runtime"),
        message,
        target,
    )
}

pub(super) fn make_run_diagnostic(run_id: &RunId, message: &str) -> Diagnostic {
    let target =
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow.run")).with_id(run_id.as_str());
    Diagnostic::new(
        reimagine_core::model::DiagnosticId::new(format!("runtime-{}", run_id.as_str())),
        DiagnosticCode::new("RUNTIME/RUN_EXECUTION_FAILED"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("runtime"),
        message,
        target,
    )
}
