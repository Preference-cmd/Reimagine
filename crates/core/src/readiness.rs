//! Executable readiness and execution-plan construction.

mod diagnostics;
mod external;
mod inputs;
mod planner;
mod targets;

pub use crate::execution_plan::{
    ExecutionEdge, ExecutionInputBinding, ExecutionInputSource, ExecutionNode, ExecutionPlan,
    ExecutionPlanResult, ExecutionStage, ExecutionWorkflowOutput, ExecutionWorkflowOutputSource,
    RunTarget, RunTargetSelection,
};
pub use external::{ExternalReadinessContext, ExternalReadinessProvider, ExternalReadinessSubject};

use crate::diagnostic::DiagnosticSeverity;
use crate::model::NodeCatalog;
use crate::workflow::Workflow;

use diagnostics::external_readiness_missing;
use external::check_external_readiness;
use inputs::validate_effective_inputs;
use planner::{build_plan, trace_execution_subgraph};
use targets::resolve_targets;

pub fn build_execution_plan(
    workflow: &Workflow,
    node_catalog: &impl NodeCatalog,
    target_selection: RunTargetSelection,
    external_provider: Option<&dyn ExternalReadinessProvider>,
) -> ExecutionPlanResult {
    let mut diagnostics = Vec::new();

    let Some(resolved_targets) =
        resolve_targets(workflow, node_catalog, &target_selection, &mut diagnostics)
    else {
        return ExecutionPlanResult::new(None, diagnostics);
    };

    let Some(planning_graph) =
        trace_execution_subgraph(workflow, node_catalog, &resolved_targets, &mut diagnostics)
    else {
        return ExecutionPlanResult::new(None, diagnostics);
    };

    validate_effective_inputs(
        workflow,
        node_catalog,
        &planning_graph,
        &mut diagnostics,
        |context, subject| {
            if let Some(provider) = external_provider {
                check_external_readiness(provider, context, subject)
            } else {
                Some(vec![external_readiness_missing(context, subject)])
            }
        },
    );

    let has_errors = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity() == DiagnosticSeverity::Error);
    if has_errors {
        return ExecutionPlanResult::new(None, diagnostics);
    }

    let plan = build_plan(
        workflow,
        node_catalog,
        target_selection,
        resolved_targets,
        planning_graph,
    );

    ExecutionPlanResult::new(Some(plan), diagnostics)
}
