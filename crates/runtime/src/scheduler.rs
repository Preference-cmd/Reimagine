//! Per-node state machine used by the scheduler and exposed via
//! `RunSnapshot.node_states`.

mod ready_set;

use std::sync::atomic::{AtomicUsize, Ordering};

use reimagine_core::model::NodeId;

pub use ready_set::ReadySetScheduler;

/// Default per-node VRAM estimate (2 GiB) the memory budget falls back
/// to when a node carries no resource requirements (BE-36).
///
/// The runner holds no `NodeCatalog`, so
/// `NodeResourceRequirements::estimated_vram_bytes` is not reachable
/// from `ExecutionNode` today; the budget is advisory and this constant
/// is the V1 fallback. Wire catalog-backed estimates through
/// `estimate_node_vram_bytes` when the runtime gains a catalog.
pub const DEFAULT_NODE_ESTIMATED_VRAM_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Admission decision produced by [`MemoryBudget::admit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// The estimated allocation fits; the caller may proceed.
    Allow,
    /// The estimated allocation crosses the soft watermark; the caller
    /// should wait for an in-flight node to complete before retrying.
    Backpressure,
    /// The estimated allocation would cross the hard ceiling; the
    /// caller must fail the node with an out-of-memory diagnostic.
    Reject,
}

/// Best-effort run memory budget used to gate node admission (BE-36).
///
/// Advisory: the budget tracks *estimates* of per-node VRAM usage
/// (backend-declared where available, [`DEFAULT_NODE_ESTIMATED_VRAM_BYTES`]
/// otherwise) and is not a substitute for backend-level memory
/// management. The runtime uses it to add backpressure once in-flight
/// estimates cross the soft watermark and to fail nodes that would
/// cross the hard ceiling.
///
/// `current_usage` is shared across concurrent stage runners via
/// `Arc<MemoryBudget>`, so one budget gates nodes admitted from
/// parallel stages.
#[derive(Debug)]
pub struct MemoryBudget {
    soft_limit: usize,
    hard_limit: usize,
    current_usage: AtomicUsize,
}

impl MemoryBudget {
    /// Build a budget with the given soft/hard ceilings in bytes.
    ///
    /// The hard ceiling is the maximum the run may ever overcommit
    /// beyond; the soft watermark is where admission starts waiting.
    /// Callers should validate `hard_limit >= soft_limit` before
    /// construction (the runtime does this in
    /// `RuntimeService::run`).
    pub fn new(soft_limit: usize, hard_limit: usize) -> Self {
        Self {
            soft_limit,
            hard_limit,
            current_usage: AtomicUsize::new(0),
        }
    }

    /// Soft watermark in bytes.
    pub fn soft_limit(&self) -> usize {
        self.soft_limit
    }

    /// Hard ceiling in bytes.
    pub fn hard_limit(&self) -> usize {
        self.hard_limit
    }

    /// Currently claimed (in-flight) bytes.
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Bytes the budget can still admit without crossing the hard
    /// ceiling.
    pub fn available_bytes(&self) -> usize {
        self.hard_limit.saturating_sub(self.current_usage())
    }

    /// Decide whether `estimated` bytes may be admitted.
    pub fn admit(&self, estimated: usize) -> AdmissionDecision {
        let current = self.current_usage.load(Ordering::Relaxed);
        match current.saturating_add(estimated) {
            x if x > self.hard_limit => AdmissionDecision::Reject,
            x if x > self.soft_limit => AdmissionDecision::Backpressure,
            _ => AdmissionDecision::Allow,
        }
    }

    /// Record that `estimated` bytes were admitted and are in-flight.
    pub fn claim(&self, estimated: usize) {
        self.current_usage.fetch_add(estimated, Ordering::Relaxed);
    }

    /// Record that `estimated` in-flight bytes completed.
    ///
    /// Saturating: the budget is advisory and must never wrap if a
    /// release races with a claim.
    pub fn release(&self, estimated: usize) {
        let _ = self
            .current_usage
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(estimated))
            });
    }
}

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

    use super::{AdmissionDecision, MemoryBudget, StageExecutionPolicy, StageNodeDecision};

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

    #[test]
    fn memory_budget_allows_while_usage_is_below_soft_limit() {
        let budget = MemoryBudget::new(100, 200);
        assert_eq!(budget.admit(40), AdmissionDecision::Allow);
        budget.claim(40);
        assert_eq!(budget.admit(60), AdmissionDecision::Allow);
        // At the soft watermark the decision is still Allow (only
        // strictly-above triggers backpressure).
        assert_eq!(budget.admit(0), AdmissionDecision::Allow);
    }

    #[test]
    fn memory_budget_backpressures_between_soft_and_hard_limits() {
        let budget = MemoryBudget::new(100, 200);
        budget.claim(100);
        assert_eq!(budget.admit(1), AdmissionDecision::Backpressure);
        assert_eq!(budget.admit(100), AdmissionDecision::Backpressure);
        assert_eq!(budget.admit(99), AdmissionDecision::Backpressure);
        // Back to exactly the soft watermark: Allow again.
        budget.release(1);
        assert_eq!(budget.admit(1), AdmissionDecision::Allow);
    }

    #[test]
    fn memory_budget_rejects_above_hard_limit() {
        let budget = MemoryBudget::new(100, 200);
        budget.claim(100);
        assert_eq!(budget.admit(100), AdmissionDecision::Backpressure);
        assert_eq!(budget.admit(101), AdmissionDecision::Reject);
    }

    #[test]
    fn memory_budget_tracks_claim_and_release() {
        let budget = MemoryBudget::new(100, 200);
        assert_eq!(budget.current_usage(), 0);
        budget.claim(80);
        assert_eq!(budget.current_usage(), 80);
        budget.claim(40);
        assert_eq!(budget.current_usage(), 120);
        budget.release(90);
        assert_eq!(budget.current_usage(), 30);
        assert_eq!(budget.available_bytes(), 170);
    }

    #[test]
    fn memory_budget_release_saturates_at_zero() {
        let budget = MemoryBudget::new(100, 200);
        budget.claim(10);
        budget.release(100);
        assert_eq!(budget.current_usage(), 0);
        budget.release(5);
        assert_eq!(budget.current_usage(), 0);
    }
}
