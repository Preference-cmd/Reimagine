//! Dynamic ready-set scheduler that replaces the sequential stage loop.
//!
//! Instead of executing stages in order, the ready-set scheduler
//! maintains a set of nodes whose inputs are all satisfied and
//! dispatches them as soon as resources are available.

use std::collections::{HashMap, HashSet};

use reimagine_core::model::NodeId;
use reimagine_core::readiness::{ExecutionEdge, ExecutionInputSource, ExecutionNode};

use crate::value_store::OutputKey;

/// Tracks which nodes are ready to execute based on input availability.
///
/// The scheduler is built from an `ExecutionPlan` and updated as nodes
/// complete. It replaces the sequential stage loop with a dynamic
/// dispatch model where nodes start as soon as their inputs are ready.
#[derive(Debug)]
pub struct ReadySetScheduler {
    /// For each node, the set of upstream output keys it depends on.
    node_dependencies: HashMap<NodeId, Vec<OutputKey>>,
    /// Output keys that have been produced.
    satisfied_outputs: HashSet<OutputKey>,
    /// Nodes that have been dispatched or completed.
    dispatched: HashSet<NodeId>,
    /// Nodes that have completed successfully.
    completed: HashSet<NodeId>,
    /// Nodes that have failed (for downstream skip propagation).
    failed: HashSet<NodeId>,
    /// Total number of nodes in the plan.
    total_nodes: usize,
}

impl ReadySetScheduler {
    /// Build a ready-set scheduler from plan nodes and edges.
    pub fn from_plan(nodes: &[ExecutionNode], edges: &[ExecutionEdge]) -> Self {
        let total_nodes = nodes.len();

        // Build dependency map: for each node, which output keys does it depend on?
        let mut node_dependencies: HashMap<NodeId, Vec<OutputKey>> = HashMap::new();

        // Initialize all nodes with empty dependency lists
        for node in nodes {
            node_dependencies.entry(node.node_id().clone()).or_default();
        }

        // Collect edge-sourced dependencies
        for edge in edges {
            let dep = OutputKey::new(edge.from_node_id().clone(), edge.from_slot_id().clone());
            node_dependencies
                .entry(edge.to_node_id().clone())
                .or_default()
                .push(dep);
        }

        // Also collect input binding dependencies (for workflow inputs and params,
        // which are always satisfied at init time, so we only track edge deps)
        for node in nodes {
            for binding in node.input_bindings() {
                if let ExecutionInputSource::Edge {
                    from_node_id,
                    from_slot_id,
                    ..
                } = binding.source()
                {
                    let dep = OutputKey::new(from_node_id.clone(), from_slot_id.clone());
                    let deps = node_dependencies.entry(node.node_id().clone()).or_default();
                    if !deps.contains(&dep) {
                        deps.push(dep);
                    }
                }
            }
        }

        Self {
            node_dependencies,
            satisfied_outputs: HashSet::new(),
            dispatched: HashSet::new(),
            completed: HashSet::new(),
            failed: HashSet::new(),
            total_nodes,
        }
    }

    /// Check if a node's inputs are all satisfied.
    fn is_ready(&self, node_id: &NodeId) -> bool {
        if self.dispatched.contains(node_id) {
            return false;
        }
        if let Some(deps) = self.node_dependencies.get(node_id) {
            deps.iter().all(|dep| self.satisfied_outputs.contains(dep))
        } else {
            // No dependencies => ready immediately
            true
        }
    }

    /// Get all nodes that are ready to execute.
    pub fn ready_nodes(&self) -> Vec<NodeId> {
        self.node_dependencies
            .keys()
            .filter(|id| self.is_ready(id))
            .cloned()
            .collect()
    }

    /// Take up to `max` ready nodes, marking them as dispatched.
    pub fn take_ready(&mut self, max: usize) -> Vec<NodeId> {
        let ready: Vec<NodeId> = self
            .node_dependencies
            .keys()
            .filter(|id| self.is_ready(id))
            .cloned()
            .take(max)
            .collect();

        for id in &ready {
            self.dispatched.insert(id.clone());
        }

        ready
    }

    /// Mark an output as produced. Returns any newly-ready nodes.
    pub fn mark_output(&mut self, key: OutputKey) -> Vec<NodeId> {
        self.satisfied_outputs.insert(key);
        self.find_newly_ready()
    }

    /// Mark a node as completed. Returns newly-ready nodes.
    pub fn mark_completed(&mut self, node_id: &NodeId) -> Vec<NodeId> {
        self.completed.insert(node_id.clone());
        self.find_newly_ready()
    }

    /// Mark a node as failed. Returns newly-ready nodes (for skip propagation).
    pub fn mark_failed(&mut self, node_id: &NodeId) -> Vec<NodeId> {
        self.failed.insert(node_id.clone());
        self.find_newly_ready()
    }

    /// Find nodes that are now ready but haven't been dispatched yet.
    fn find_newly_ready(&self) -> Vec<NodeId> {
        self.node_dependencies
            .keys()
            .filter(|id| self.is_ready(id))
            .cloned()
            .collect()
    }

    /// Check if all nodes have completed (successfully or otherwise).
    pub fn is_complete(&self) -> bool {
        self.completed.len() + self.failed.len() >= self.total_nodes
    }

    /// Number of nodes completed successfully.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Number of nodes that have been dispatched (running or completed).
    pub fn dispatched_count(&self) -> usize {
        self.dispatched.len()
    }

    /// Total number of nodes in the plan.
    pub fn total_nodes(&self) -> usize {
        self.total_nodes
    }

    /// Check if a specific node has completed.
    pub fn is_completed(&self, node_id: &NodeId) -> bool {
        self.completed.contains(node_id)
    }

    /// Check if a specific node has failed.
    pub fn is_failed(&self, node_id: &NodeId) -> bool {
        self.failed.contains(node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::{EdgeId, NodeTypeId, SlotId};

    fn node(node_id: &str, slots: &[&str]) -> ExecutionNode {
        ExecutionNode::new(
            NodeId::new(node_id),
            NodeTypeId::new("mock"),
            Vec::new(),
            slots.iter().map(|s| SlotId::new(*s)).collect(),
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

    #[test]
    fn single_node_is_ready_immediately() {
        let nodes = vec![node("a", &["out"])];
        let mut sched = ReadySetScheduler::from_plan(&nodes, &[]);
        assert_eq!(sched.total_nodes(), 1);
        let ready = sched.take_ready(10);
        assert_eq!(ready, vec![NodeId::new("a")]);
    }

    #[test]
    fn dependent_node_not_ready_until_dependency_satisfied() {
        let nodes = vec![node("a", &["out"]), node("b", &["in"])];
        let edges = vec![edge("a", "out", "b", "in")];
        let mut sched = ReadySetScheduler::from_plan(&nodes, &edges);

        // Initially only "a" is ready
        let ready = sched.take_ready(10);
        assert_eq!(ready, vec![NodeId::new("a")]);

        // Mark a's output as satisfied
        sched.mark_output(OutputKey::new(NodeId::new("a"), SlotId::new("out")));

        // Now "b" should be ready
        let ready = sched.take_ready(10);
        assert_eq!(ready, vec![NodeId::new("b")]);
    }

    #[test]
    fn diamond_dependency_allows_parallel_execution() {
        // a -> b, a -> c, b -> d, c -> d
        let nodes = vec![
            node("a", &["out"]),
            node("b", &["in", "out"]),
            node("c", &["in", "out"]),
            node("d", &["in"]),
        ];
        let edges = vec![
            edge("a", "out", "b", "in"),
            edge("a", "out", "c", "in"),
            edge("b", "out", "d", "in"),
            edge("c", "out", "d", "in"),
        ];
        let mut sched = ReadySetScheduler::from_plan(&nodes, &edges);

        // Initially only "a" is ready
        let ready = sched.take_ready(10);
        assert_eq!(ready, vec![NodeId::new("a")]);

        // Mark a's output
        sched.mark_output(OutputKey::new(NodeId::new("a"), SlotId::new("out")));

        // Both "b" and "c" should be ready now
        let mut ready = sched.take_ready(10);
        ready.sort();
        assert_eq!(ready, vec![NodeId::new("b"), NodeId::new("c")]);

        // Mark both outputs
        sched.mark_output(OutputKey::new(NodeId::new("b"), SlotId::new("out")));
        sched.mark_output(OutputKey::new(NodeId::new("c"), SlotId::new("out")));

        // Now "d" should be ready
        let ready = sched.take_ready(10);
        assert_eq!(ready, vec![NodeId::new("d")]);
    }

    #[test]
    fn completion_tracks_count() {
        let nodes = vec![node("a", &["out"]), node("b", &["in"])];
        let mut sched = ReadySetScheduler::from_plan(&nodes, &[]);

        assert_eq!(sched.completed_count(), 0);
        assert!(!sched.is_complete());

        sched.mark_completed(&NodeId::new("a"));
        assert_eq!(sched.completed_count(), 1);
        assert!(!sched.is_complete());

        sched.mark_completed(&NodeId::new("b"));
        assert_eq!(sched.completed_count(), 2);
        assert!(sched.is_complete());
    }

    #[test]
    fn take_ready_respects_max_limit() {
        let nodes = vec![
            node("a", &["out"]),
            node("b", &["out"]),
            node("c", &["out"]),
        ];
        let mut sched = ReadySetScheduler::from_plan(&nodes, &[]);

        let ready = sched.take_ready(2);
        assert_eq!(ready.len(), 2);

        // One more should be available
        let ready = sched.take_ready(10);
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn failed_node_does_not_block_unrelated_nodes() {
        // a -> b, c -> d (two independent chains)
        let nodes = vec![
            node("a", &["out"]),
            node("b", &["in"]),
            node("c", &["out"]),
            node("d", &["in"]),
        ];
        let edges = vec![edge("a", "out", "b", "in"), edge("c", "out", "d", "in")];
        let mut sched = ReadySetScheduler::from_plan(&nodes, &edges);

        // Mark a as failed
        sched.mark_failed(&NodeId::new("a"));

        // c should still be ready (independent chain)
        let ready = sched.take_ready(10);
        assert!(ready.contains(&NodeId::new("c")));
    }
}
