//! Per-node execution deadlines tracked by the runtime scheduler.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reimagine_core::model::NodeId;

/// Per-node execution deadlines tracked by the runner and stage reducer.
#[derive(Debug, Default, Clone)]
pub struct NodeDeadlines {
    deadlines: Arc<Mutex<HashMap<NodeId, Instant>>>,
}

impl NodeDeadlines {
    pub fn new() -> Self {
        Self::default()
    }

    /// Arm a deadline for `node_id` starting now: the node must complete
    /// within `timeout`, or the stage reducer fails it with a timeout
    /// diagnostic.
    pub fn set_deadline(&self, node_id: &NodeId, timeout: Duration) {
        self.set_deadline_at(node_id, Instant::now() + timeout);
    }

    /// Arm an absolute deadline for `node_id`.
    pub fn set_deadline_at(&self, node_id: &NodeId, deadline: Instant) {
        self.deadlines
            .lock()
            .expect("deadline map poisoned")
            .insert(node_id.clone(), deadline);
    }

    /// The deadline currently armed for `node_id`, if any.
    pub fn node_deadline(&self, node_id: &NodeId) -> Option<Instant> {
        self.deadlines
            .lock()
            .expect("deadline map poisoned")
            .get(node_id)
            .copied()
    }

    /// Returns `true` when `node_id` has a deadline in the past.
    pub fn is_node_expired(&self, node_id: &NodeId) -> bool {
        self.node_deadline(node_id)
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Disarm any deadline for `node_id`.
    pub fn clear_deadline(&self, node_id: &NodeId) {
        self.deadlines
            .lock()
            .expect("deadline map poisoned")
            .remove(node_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_expiry_is_time_based() {
        let deadlines = NodeDeadlines::new();
        let node = NodeId::new("a");

        deadlines.set_deadline(&node, Duration::from_secs(3600));
        assert!(!deadlines.is_node_expired(&node));

        deadlines.set_deadline_at(&node, Instant::now() - Duration::from_secs(1));
        assert!(deadlines.is_node_expired(&node));

        deadlines.clear_deadline(&node);
        assert!(!deadlines.is_node_expired(&node));
    }

    #[test]
    fn deadlines_are_per_node() {
        let deadlines = NodeDeadlines::new();
        let a = NodeId::new("a");
        let b = NodeId::new("b");

        deadlines.set_deadline_at(&a, Instant::now() - Duration::from_secs(1));
        assert!(deadlines.is_node_expired(&a));
        assert!(!deadlines.is_node_expired(&b));
        assert_eq!(deadlines.node_deadline(&b), None);
    }
}
