//! Per-node state machine used by the scheduler and exposed via
//! `RunSnapshot.node_states`.

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
