use std::collections::{BTreeSet, HashSet};

use crate::event::OperationReport;
use crate::execution_plan::{ExecutionWorkflowOutput, ExecutionWorkflowOutputSource};
use crate::model::{NodeCatalog, NodeId};
use crate::workflow::{Endpoint, Workflow};

use super::RunTarget;
use super::RunTargetSelection;
use super::diagnostics::{
    no_target, non_contributing_pure_graph, target_invalid, workflow_target_invalid,
};

#[derive(Debug, Clone)]
pub struct ResolvedTargets {
    pub targets: Vec<RunTarget>,
    pub target_node_ids: BTreeSet<NodeId>,
    pub workflow_outputs: Vec<ExecutionWorkflowOutput>,
}

pub fn resolve_targets(
    workflow: &Workflow,
    node_catalog: &impl NodeCatalog,
    target_selection: &RunTargetSelection,
    report: &mut OperationReport,
) -> Option<ResolvedTargets> {
    let target_capable_nodes: HashSet<NodeId> = workflow
        .nodes()
        .iter()
        .filter(|node| {
            node_catalog
                .get(node.type_id())
                .map(|node_def| {
                    node_def
                        .output_slots()
                        .iter()
                        .all(|slot| !slot.is_required())
                })
                .unwrap_or(false)
        })
        .map(|node| node.id().clone())
        .collect();

    match target_selection {
        RunTargetSelection::AllDefaultTargets => {
            let targets: Vec<RunTarget> = workflow
                .nodes()
                .iter()
                .filter(|node| target_capable_nodes.contains(node.id()))
                .map(|node| RunTarget::Node {
                    node_id: node.id().clone(),
                })
                .collect();

            if targets.is_empty() {
                report.push_diagnostic(no_target(workflow.id()));
                if workflow
                    .nodes()
                    .iter()
                    .any(|node| node_catalog.get(node.type_id()).is_some())
                {
                    report.push_diagnostic(non_contributing_pure_graph(workflow.id()));
                }
                return None;
            }

            Some(ResolvedTargets {
                target_node_ids: targets
                    .iter()
                    .filter_map(|target| match target {
                        RunTarget::Node { node_id } => Some(node_id.clone()),
                        _ => None,
                    })
                    .collect(),
                workflow_outputs: Vec::new(),
                targets,
            })
        }
        RunTargetSelection::ExplicitTargets(targets) => {
            if targets.is_empty() {
                report.push_diagnostic(workflow_target_invalid(workflow.id(), "targets"));
                return None;
            }

            let mut target_node_ids = BTreeSet::new();
            let mut workflow_outputs = Vec::new();

            for target in targets {
                match target {
                    RunTarget::Node { node_id } => {
                        if !target_capable_nodes.contains(node_id) {
                            report.push_diagnostic(target_invalid(node_id));
                        } else {
                            target_node_ids.insert(node_id.clone());
                        }
                    }
                    RunTarget::NodeOutput { node_id, slot_id } => {
                        let Some(node) = workflow.nodes().iter().find(|node| node.id() == node_id)
                        else {
                            report.push_diagnostic(target_invalid(node_id));
                            continue;
                        };
                        let Some(node_def) = node_catalog.get(node.type_id()) else {
                            report.push_diagnostic(target_invalid(node_id));
                            continue;
                        };
                        if node_def.output_slot(slot_id).is_none() {
                            report.push_diagnostic(target_invalid(node_id));
                            continue;
                        }
                        target_node_ids.insert(node_id.clone());
                    }
                    RunTarget::WorkflowOutput { output_id } => {
                        if workflow.interface().output(output_id).is_none() {
                            report.push_diagnostic(workflow_target_invalid(
                                workflow.id(),
                                format!("interface.outputs.{}", output_id.as_str()),
                            ));
                            continue;
                        }

                        let Some(edge) = workflow.edges().iter().find(|edge| {
                            matches!(
                                edge.to(),
                                Endpoint::WorkflowOutput { workflow_output }
                                    if workflow_output == output_id
                            )
                        }) else {
                            report.push_diagnostic(workflow_target_invalid(
                                workflow.id(),
                                format!("interface.outputs.{}", output_id.as_str()),
                            ));
                            continue;
                        };

                        match edge.from() {
                            Endpoint::NodeSlot { node, slot } => {
                                target_node_ids.insert(node.clone());
                                workflow_outputs.push(ExecutionWorkflowOutput::new(
                                    output_id.clone(),
                                    ExecutionWorkflowOutputSource::NodeOutput {
                                        node_id: node.clone(),
                                        slot_id: slot.clone(),
                                    },
                                ));
                            }
                            Endpoint::WorkflowInput { workflow_input } => {
                                workflow_outputs.push(ExecutionWorkflowOutput::new(
                                    output_id.clone(),
                                    ExecutionWorkflowOutputSource::WorkflowInput {
                                        workflow_input_id: workflow_input.clone(),
                                    },
                                ));
                            }
                            Endpoint::WorkflowOutput { .. } => {
                                report.push_diagnostic(workflow_target_invalid(
                                    workflow.id(),
                                    format!("interface.outputs.{}", output_id.as_str()),
                                ));
                            }
                        }
                    }
                }
            }

            if target_node_ids.is_empty() && workflow_outputs.is_empty() {
                return None;
            }

            Some(ResolvedTargets {
                targets: targets.clone(),
                target_node_ids,
                workflow_outputs,
            })
        }
    }
}
