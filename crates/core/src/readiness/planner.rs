use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::event::OperationReport;
use crate::execution_plan::{
    ExecutionEdge, ExecutionInputSource, ExecutionNode, ExecutionPlan, ExecutionStage,
    ExecutionWorkflowOutputSource,
};
use crate::model::{EdgeId, NodeCatalog, NodeEffect, NodeId, SlotId};
use crate::workflow::{Endpoint, Workflow};

use super::RunTargetSelection;
use super::diagnostics::{executable_cycle, non_contributing_pure_graph};
use super::inputs::{effective_binding_for_slot, node_input_bindings};
use super::targets::ResolvedTargets;

#[derive(Debug, Clone)]
pub struct PlanningGraph {
    pub node_ids: BTreeSet<NodeId>,
    pub edge_ids: BTreeSet<EdgeId>,
}

pub fn trace_execution_subgraph(
    workflow: &Workflow,
    node_catalog: &impl NodeCatalog,
    resolved_targets: &ResolvedTargets,
    report: &mut OperationReport,
) -> Option<PlanningGraph> {
    let incoming = incoming_edges_by_node(workflow);
    let mut node_ids = BTreeSet::new();
    let mut edge_ids = BTreeSet::new();
    let mut stack: Vec<NodeId> = resolved_targets.target_node_ids.iter().cloned().collect();

    while let Some(node_id) = stack.pop() {
        if !node_ids.insert(node_id.clone()) {
            continue;
        }

        let Some(node) = workflow.nodes().iter().find(|node| node.id() == &node_id) else {
            continue;
        };
        let Some(node_def) = node_catalog.get(node.type_id()) else {
            continue;
        };

        for input_slot in node_def.input_slots() {
            let Some(binding) = effective_binding_for_slot(workflow, node, input_slot, &incoming)
            else {
                continue;
            };
            if let ExecutionInputSource::Edge {
                edge_id,
                from_node_id,
                ..
            } = binding
            {
                edge_ids.insert(edge_id);
                stack.push(from_node_id);
            }
        }
    }

    if has_cycle(workflow, &node_ids, &edge_ids) {
        report.push_diagnostic(executable_cycle(workflow.id()));
        return None;
    }

    let planning_graph = PlanningGraph { node_ids, edge_ids };
    validate_required_outputs(workflow, node_catalog, resolved_targets, &planning_graph, report);

    Some(planning_graph)
}

fn validate_required_outputs(
    workflow: &Workflow,
    node_catalog: &impl NodeCatalog,
    resolved_targets: &ResolvedTargets,
    planning_graph: &PlanningGraph,
    report: &mut OperationReport,
) {
    let exposed_outputs: BTreeSet<(NodeId, SlotId)> = resolved_targets
        .targets
        .iter()
        .filter_map(|target| match target {
            super::RunTarget::NodeOutput { node_id, slot_id } => {
                Some((node_id.clone(), slot_id.clone()))
            }
            super::RunTarget::Node { .. } | super::RunTarget::WorkflowOutput { .. } => None,
        })
        .chain(
            resolved_targets
                .workflow_outputs
                .iter()
                .filter_map(|output| match output.source() {
                    ExecutionWorkflowOutputSource::NodeOutput { node_id, slot_id } => {
                        Some((node_id.clone(), slot_id.clone()))
                    }
                    ExecutionWorkflowOutputSource::WorkflowInput { .. } => None,
                }),
        )
        .collect();

    for node in workflow
        .nodes()
        .iter()
        .filter(|node| planning_graph.node_ids.contains(node.id()))
    {
        let Some(node_def) = node_catalog.get(node.type_id()) else {
            continue;
        };
        if node_def.effect() != NodeEffect::Pure {
            continue;
        }

        for output_slot in node_def
            .output_slots()
            .iter()
            .filter(|slot| slot.is_required())
        {
            let output_key = (node.id().clone(), output_slot.id().clone());
            let consumed = workflow.edges().iter().any(|edge| {
                planning_graph.edge_ids.contains(edge.id())
                    && matches!(
                        edge.from(),
                        Endpoint::NodeSlot { node: from_node, slot }
                            if from_node == node.id() && slot == output_slot.id()
                    )
            });

            if !consumed && !exposed_outputs.contains(&output_key) {
                report.push_diagnostic(non_contributing_pure_graph(workflow.id()));
            }
        }
    }
}

pub fn build_plan(
    workflow: &Workflow,
    node_catalog: &impl NodeCatalog,
    target_selection: RunTargetSelection,
    resolved_targets: ResolvedTargets,
    planning_graph: PlanningGraph,
) -> ExecutionPlan {
    let incoming = incoming_edges_by_node(workflow);

    let nodes: Vec<ExecutionNode> = workflow
        .nodes()
        .iter()
        .filter(|node| planning_graph.node_ids.contains(node.id()))
        .filter_map(|node| {
            let node_def = node_catalog.get(node.type_id())?;
            Some(ExecutionNode::new(
                node.id().clone(),
                node.type_id().clone(),
                node_input_bindings(workflow, node, node_def, &incoming),
                node_def
                    .output_slots()
                    .iter()
                    .map(|slot| slot.id().clone())
                    .collect(),
            ))
        })
        .collect();

    let edges: Vec<ExecutionEdge> = workflow
        .edges()
        .iter()
        .filter(|edge| planning_graph.edge_ids.contains(edge.id()))
        .filter_map(|edge| match (edge.from(), edge.to()) {
            (
                Endpoint::NodeSlot {
                    node: from_node,
                    slot: from_slot,
                },
                Endpoint::NodeSlot {
                    node: to_node,
                    slot: to_slot,
                },
            ) => Some(ExecutionEdge::new(
                edge.id().clone(),
                from_node.clone(),
                from_slot.clone(),
                to_node.clone(),
                to_slot.clone(),
            )),
            _ => None,
        })
        .collect();

    let stages = build_stages(workflow, &planning_graph);

    ExecutionPlan::new(
        workflow.id().clone(),
        workflow.version(),
        target_selection,
        resolved_targets.targets,
        nodes,
        edges,
        resolved_targets.workflow_outputs,
        stages,
    )
}

fn build_stages(workflow: &Workflow, planning_graph: &PlanningGraph) -> Vec<ExecutionStage> {
    let node_order: HashMap<NodeId, usize> = workflow
        .nodes()
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id().clone(), index))
        .collect();
    let mut indegree: BTreeMap<NodeId, usize> = planning_graph
        .node_ids
        .iter()
        .cloned()
        .map(|node_id| (node_id, 0))
        .collect();
    let mut outgoing: BTreeMap<NodeId, Vec<NodeId>> = planning_graph
        .node_ids
        .iter()
        .cloned()
        .map(|node_id| (node_id, Vec::new()))
        .collect();

    for edge in workflow
        .edges()
        .iter()
        .filter(|edge| planning_graph.edge_ids.contains(edge.id()))
    {
        let (Endpoint::NodeSlot { node: from, .. }, Endpoint::NodeSlot { node: to, .. }) =
            (edge.from(), edge.to())
        else {
            continue;
        };
        *indegree.get_mut(to).expect("to node indegree") += 1;
        outgoing
            .get_mut(from)
            .expect("from node outgoing")
            .push(to.clone());
    }

    let mut ready: VecDeque<NodeId> = indegree
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(node_id, _)| node_id.clone())
        .collect();
    let mut scheduled = BTreeSet::new();
    let mut stages = Vec::new();
    let mut stage_index = 0;

    while !ready.is_empty() {
        let mut current_stage: Vec<NodeId> = ready.drain(..).collect();
        current_stage.sort_by_key(|node_id| node_order.get(node_id).copied().unwrap_or(usize::MAX));
        for node_id in &current_stage {
            scheduled.insert(node_id.clone());
        }

        for node_id in &current_stage {
            if let Some(next_nodes) = outgoing.get(node_id) {
                for next in next_nodes {
                    let count = indegree.get_mut(next).expect("downstream indegree");
                    *count -= 1;
                }
            }
        }

        let mut next_ready: Vec<NodeId> = indegree
            .iter()
            .filter(|(node_id, count)| **count == 0 && !scheduled.contains(*node_id))
            .map(|(node_id, _)| node_id.clone())
            .collect();
        next_ready.sort_by_key(|node_id| node_order.get(node_id).copied().unwrap_or(usize::MAX));
        ready = next_ready.into();

        stages.push(ExecutionStage::new(stage_index, current_stage));
        stage_index += 1;
    }

    stages
}

fn incoming_edges_by_node(
    workflow: &Workflow,
) -> HashMap<(NodeId, SlotId), &crate::workflow::WorkflowEdge> {
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

fn has_cycle(
    workflow: &Workflow,
    node_ids: &BTreeSet<NodeId>,
    edge_ids: &BTreeSet<EdgeId>,
) -> bool {
    let mut indegree: HashMap<NodeId, usize> = node_ids
        .iter()
        .cloned()
        .map(|node_id| (node_id, 0))
        .collect();
    let mut outgoing: HashMap<NodeId, Vec<NodeId>> = node_ids
        .iter()
        .cloned()
        .map(|node_id| (node_id, Vec::new()))
        .collect();

    for edge in workflow
        .edges()
        .iter()
        .filter(|edge| edge_ids.contains(edge.id()))
    {
        let (Endpoint::NodeSlot { node: from, .. }, Endpoint::NodeSlot { node: to, .. }) =
            (edge.from(), edge.to())
        else {
            continue;
        };
        *indegree.get_mut(to).expect("to indegree") += 1;
        outgoing
            .get_mut(from)
            .expect("from outgoing")
            .push(to.clone());
    }

    let mut queue: VecDeque<NodeId> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(node_id, _)| node_id.clone())
        .collect();
    let mut visited = 0;

    while let Some(node_id) = queue.pop_front() {
        visited += 1;
        if let Some(next_nodes) = outgoing.get(&node_id) {
            for next in next_nodes {
                let degree = indegree.get_mut(next).expect("downstream indegree");
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(next.clone());
                }
            }
        }
    }

    visited != node_ids.len()
}
