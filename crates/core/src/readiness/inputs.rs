use std::collections::HashMap;

use crate::diagnostic::{DiagnosticTarget, DiagnosticTargetDomain};
use crate::event::OperationReport;
use crate::model::{InputSlotDef, NodeCatalog, ParamValue, SlotId};
use crate::workflow::{Endpoint, Workflow, WorkflowEdge, WorkflowNode};

use super::diagnostics::{external_readiness_missing, required_input_missing};
use super::external::{
    ExternalReadinessContext, ExternalReadinessSubject, project_external_diagnostic,
};
use super::planner::PlanningGraph;
use super::{ExecutionInputBinding, ExecutionInputSource};

pub fn validate_effective_inputs(
    workflow: &Workflow,
    node_catalog: &impl NodeCatalog,
    planning_graph: &PlanningGraph,
    report: &mut OperationReport,
    external_lookup: impl Fn(
        &ExternalReadinessContext,
        &ExternalReadinessSubject,
    ) -> Option<Vec<crate::diagnostic::Diagnostic>>,
) {
    let incoming = incoming_edges_by_node(workflow);

    for node in workflow
        .nodes()
        .iter()
        .filter(|node| planning_graph.node_ids.contains(node.id()))
    {
        let Some(node_def) = node_catalog.get(node.type_id()) else {
            continue;
        };

        for input_slot in node_def.input_slots() {
            let binding = effective_binding_for_slot(workflow, node, input_slot, &incoming);
            if input_slot.is_required() && binding.is_none() {
                report.push_diagnostic(required_input_missing(node.id(), input_slot.id()));
                continue;
            }

            if let Some(subject) = external_subject_for_slot(node, input_slot, binding.as_ref()) {
                let context = ExternalReadinessContext::new(
                    workflow.id().clone(),
                    workflow.version(),
                    format!(
                        "nodes.{}.params.{}",
                        node.id().as_str(),
                        input_slot.id().as_str()
                    ),
                )
                .with_node(node.id().clone())
                .with_slot(input_slot.id().clone());

                let primary = DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow.node"))
                    .with_id(node.id().as_str().to_owned())
                    .with_path(format!("params.{}", input_slot.id().as_str()));

                match external_lookup(&context, &subject) {
                    Some(diagnostics) => {
                        for diagnostic in diagnostics {
                            report.push_diagnostic(project_external_diagnostic(
                                &diagnostic,
                                primary.clone(),
                            ));
                        }
                    }
                    None => report.push_diagnostic(external_readiness_missing(&context, &subject)),
                }
            }
        }
    }
}

pub fn node_input_bindings(
    workflow: &Workflow,
    node: &WorkflowNode,
    node_def: &crate::model::NodeDef,
    incoming: &HashMap<(crate::model::NodeId, SlotId), &WorkflowEdge>,
) -> Vec<ExecutionInputBinding> {
    node_def
        .input_slots()
        .iter()
        .filter_map(|slot| {
            effective_binding_for_slot(workflow, node, slot, incoming)
                .map(|binding| ExecutionInputBinding::new(slot.id().clone(), binding))
        })
        .collect()
}

pub fn effective_binding_for_slot(
    _workflow: &Workflow,
    node: &WorkflowNode,
    input_slot: &InputSlotDef,
    incoming: &HashMap<(crate::model::NodeId, SlotId), &WorkflowEdge>,
) -> Option<ExecutionInputSource> {
    if let Some(edge) = incoming.get(&(node.id().clone(), input_slot.id().clone())) {
        match edge.from() {
            Endpoint::NodeSlot {
                node: from_node,
                slot: from_slot,
            } => {
                return Some(ExecutionInputSource::Edge {
                    edge_id: edge.id().clone(),
                    from_node_id: from_node.clone(),
                    from_slot_id: from_slot.clone(),
                });
            }
            Endpoint::WorkflowInput { workflow_input } => {
                return Some(ExecutionInputSource::WorkflowInput {
                    edge_id: edge.id().clone(),
                    workflow_input_id: workflow_input.clone(),
                });
            }
            Endpoint::WorkflowOutput { .. } => {}
        }
    }

    if !input_slot.is_dynamic() && node.params().contains_key(input_slot.id()) {
        return Some(ExecutionInputSource::Param {
            slot_id: input_slot.id().clone(),
        });
    }

    input_slot
        .default_value()
        .map(|_| ExecutionInputSource::Default {
            slot_id: input_slot.id().clone(),
        })
}

fn external_subject_for_slot(
    node: &WorkflowNode,
    input_slot: &InputSlotDef,
    binding: Option<&ExecutionInputSource>,
) -> Option<ExternalReadinessSubject> {
    let value = match binding {
        Some(ExecutionInputSource::Param { .. }) => node.params().get(input_slot.id()),
        Some(ExecutionInputSource::Default { .. }) => input_slot.default_value(),
        Some(ExecutionInputSource::Edge { .. } | ExecutionInputSource::WorkflowInput { .. }) => {
            None
        }
        None => None,
    }?;

    match value {
        ParamValue::ModelRef(model_ref) => {
            Some(ExternalReadinessSubject::ModelRef(model_ref.clone()))
        }
        _ => None,
    }
}

fn incoming_edges_by_node(
    workflow: &Workflow,
) -> HashMap<(crate::model::NodeId, SlotId), &WorkflowEdge> {
    workflow
        .edges()
        .iter()
        .filter_map(|edge| {
            if let Endpoint::NodeSlot { node, slot } = edge.to() {
                Some(((node.clone(), slot.clone()), edge))
            } else {
                None
            }
        })
        .collect()
}
