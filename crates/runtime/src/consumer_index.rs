//! Per-plan edge-sourced consumer index used to enforce producer-declared
//! retention in the runtime.
//!
//! The index is built from `ExecutionPlan::edges()` (the active execution
//! plan only) and keyed by [`OutputKey`] (producer `(node_id, slot_id)`).
//! Fan-out is defined strictly as the number of edge-sourced consumers
//! in the active plan. Workflow outputs, target markers, and edges
//! outside the active plan do not count.
//!
//! The runtime uses the index to:
//!
//! - diagnose `SingleUse` outputs whose fan-out is greater than one
//!   before any downstream consumer receives the value;
//! - find the unique consumer of a `SingleUse` output so the runtime
//!   can drop the producer value from `RunValueStore` once that
//!   consumer's execution attempt completes (success / failure /
//!   cancel);
//! - leave `RunScoped` and `WorkspaceScoped` values alone — only
//!   `SingleUse` fan-out is interpreted from the index.

use std::collections::HashMap;

use reimagine_core::model::{NodeId, SlotId};
use reimagine_core::readiness::ExecutionPlan;

use crate::value_store::OutputKey;

/// Edge-sourced consumer recorded in the index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConsumerBinding {
    pub to_node_id: NodeId,
    pub to_slot_id: SlotId,
}

/// Per-plan edge-sourced consumer index built from
/// `ExecutionPlan::edges()`.
///
/// The index is constructed once when the runner starts a run and then
/// queried by the runner as it inserts and drops `SingleUse` values.
#[derive(Debug, Default, Clone)]
pub struct PlanConsumerIndex {
    consumers: HashMap<OutputKey, Vec<ConsumerBinding>>,
}

impl PlanConsumerIndex {
    /// Build the index from the active execution plan. Edges are
    /// collected as-is; `OutputKey` collapses on `(from_node_id,
    /// from_slot_id)` which is the V1 fan-out key.
    pub fn from_plan(plan: &ExecutionPlan) -> Self {
        let mut consumers: HashMap<OutputKey, Vec<ConsumerBinding>> = HashMap::new();
        for edge in plan.edges() {
            let key = OutputKey::new(edge.from_node_id().clone(), edge.from_slot_id().clone());
            consumers.entry(key).or_default().push(ConsumerBinding {
                to_node_id: edge.to_node_id().clone(),
                to_slot_id: edge.to_slot_id().clone(),
            });
        }
        Self { consumers }
    }

    /// Return the edge-sourced consumer list for the given producer key.
    pub fn consumers(&self, key: &OutputKey) -> &[ConsumerBinding] {
        self.consumers.get(key).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Number of edge-sourced consumers for the given producer key.
    pub fn fan_out(&self, key: &OutputKey) -> usize {
        self.consumers.get(key).map(|v| v.len()).unwrap_or(0)
    }

    /// Returns `true` if the producer key has exactly one edge-sourced
    /// consumer in the active plan.
    pub fn has_unique_consumer(&self, key: &OutputKey) -> bool {
        self.fan_out(key) == 1
    }

    /// Returns the unique consumer of the given producer key, if and
    /// only if the active plan has exactly one edge-sourced consumer for
    /// it.
    pub fn unique_consumer(&self, key: &OutputKey) -> Option<&ConsumerBinding> {
        self.consumers
            .get(key)
            .and_then(|v| v.first())
            .filter(|_| self.has_unique_consumer(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::{EdgeId, NodeId, SlotId, WorkflowId, WorkflowVersion};
    use reimagine_core::readiness::{
        ExecutionEdge, ExecutionNode, ExecutionPlan, ExecutionStage, RunTarget, RunTargetSelection,
    };

    fn node(node_id: &str, slot: &str) -> ExecutionNode {
        ExecutionNode::new(
            NodeId::new(node_id),
            reimagine_core::model::NodeTypeId::new("mock"),
            Vec::new(),
            vec![SlotId::new(slot)],
        )
    }

    fn edge(from: &str, from_slot: &str, to: &str, to_slot: &str) -> ExecutionEdge {
        ExecutionEdge::new(
            EdgeId::new(format!("e-{from}-{to}")),
            NodeId::new(from),
            SlotId::new(from_slot),
            NodeId::new(to),
            SlotId::new(to_slot),
        )
    }

    fn plan(nodes: Vec<ExecutionNode>, edges: Vec<ExecutionEdge>) -> ExecutionPlan {
        ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: nodes
                    .last()
                    .map(|n| n.node_id().clone())
                    .unwrap_or_else(|| NodeId::new("a")),
            }],
            nodes,
            edges,
            Vec::new(),
            vec![ExecutionStage::new(0, Vec::new())],
        )
    }

    #[test]
    fn fan_out_is_zero_when_no_edge_targets_the_output() {
        let plan = plan(vec![node("a", "out")], Vec::new());
        let index = PlanConsumerIndex::from_plan(&plan);
        let key = OutputKey::new(NodeId::new("a"), SlotId::new("out"));
        assert_eq!(index.fan_out(&key), 0);
        assert!(index.unique_consumer(&key).is_none());
    }

    #[test]
    fn fan_out_counts_edge_sourced_consumers_in_the_active_plan() {
        let nodes = vec![node("a", "out"), node("b", "in"), node("c", "in")];
        let edges = vec![edge("a", "out", "b", "in"), edge("a", "out", "c", "in")];
        let plan = plan(nodes, edges);
        let index = PlanConsumerIndex::from_plan(&plan);
        let key = OutputKey::new(NodeId::new("a"), SlotId::new("out"));
        assert_eq!(index.fan_out(&key), 2);
        assert!(index.unique_consumer(&key).is_none());
    }

    #[test]
    fn unique_consumer_is_exposed_when_fan_out_is_one() {
        let nodes = vec![node("a", "out"), node("b", "in")];
        let edges = vec![edge("a", "out", "b", "in")];
        let plan = plan(nodes, edges);
        let index = PlanConsumerIndex::from_plan(&plan);
        let key = OutputKey::new(NodeId::new("a"), SlotId::new("out"));
        assert!(index.has_unique_consumer(&key));
        let unique = index.unique_consumer(&key).expect("unique");
        assert_eq!(unique.to_node_id, NodeId::new("b"));
        assert_eq!(unique.to_slot_id, SlotId::new("in"));
    }

    #[test]
    fn edges_outside_the_active_plan_are_not_recorded() {
        // Simulate an explicit-target plan that excludes a branch of the
        // saved workflow: the plan contains only `a` and `c`; the
        // `a -> b` edge from the saved workflow is NOT in this plan.
        let nodes = vec![node("a", "out"), node("c", "in")];
        let edges = vec![edge("a", "out", "c", "in")];
        let plan = plan(nodes, edges);
        let index = PlanConsumerIndex::from_plan(&plan);
        let key = OutputKey::new(NodeId::new("a"), SlotId::new("out"));
        assert_eq!(index.fan_out(&key), 1);
        let unique = index.unique_consumer(&key).expect("unique");
        assert_eq!(unique.to_node_id, NodeId::new("c"));
    }
}
