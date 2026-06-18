//! Per-node state machine used by the scheduler and exposed via
//! `RunSnapshot.node_states`.

use reimagine_core::model::NodeId;

/// State of an individual node within a running or completed run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum NodeState {
    /// Node has been registered in the run but not yet started.
    Queued,
    /// Node is currently executing.
    Running,
    /// Node finished successfully.
    Completed,
    /// Node failed; the run will skip downstream nodes.
    Failed,
    /// Node was skipped because an upstream node failed or another
    /// readiness condition prevented it.
    Skipped,
    /// Node execution was prevented by cancellation.
    Cancelled,
}

impl NodeState {
    /// Returns `true` once the node has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Skipped | Self::Cancelled
        )
    }
}

/// Decision for a single workflow node in the current stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageNodeDecision {
    /// The runner should invoke the node executor.
    Execute,
    /// The runner should mark the node skipped with the given reason.
    Skip { reason: String },
}

/// Scheduler-owned fail-fast policy over workflow node invocations.
///
/// This deliberately does not know value stores, artifacts, backend
/// operations, or model resources. It only captures the V1 policy that once
/// the run has observed a node failure, remaining nodes are skipped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StageExecutionPolicy {
    failed_node: Option<NodeId>,
    failed_message: Option<String>,
}

impl StageExecutionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_failure(&mut self, node_id: NodeId, message: String) {
        if self.failed_node.is_none() {
            self.failed_node = Some(node_id);
            self.failed_message = Some(message);
        }
    }

    pub fn decision_for(&self, _node_id: &NodeId) -> StageNodeDecision {
        match &self.failed_node {
            Some(failing_node) => StageNodeDecision::Skip {
                reason: format!("upstream node {failing_node} failed"),
            },
            None => StageNodeDecision::Execute,
        }
    }

    pub fn failed_message(&self) -> Option<&str> {
        self.failed_message.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use reimagine_core::model::NodeId;

    use super::{StageExecutionPolicy, StageNodeDecision};

    #[test]
    fn stage_policy_executes_when_no_node_has_failed() {
        let policy = StageExecutionPolicy::new();

        assert_eq!(
            policy.decision_for(&NodeId::new("node_a")),
            StageNodeDecision::Execute
        );
    }

    #[test]
    fn stage_policy_skips_after_first_failure() {
        let mut policy = StageExecutionPolicy::new();
        policy.record_failure(NodeId::new("node_a"), "kaboom".to_owned());
        policy.record_failure(NodeId::new("node_b"), "ignored".to_owned());

        assert_eq!(policy.failed_message(), Some("kaboom"));
        assert_eq!(
            policy.decision_for(&NodeId::new("node_c")),
            StageNodeDecision::Skip {
                reason: "upstream node node_a failed".to_owned()
            }
        );
    }
}
