//! Task execution state.

/// High-level execution state of a task or run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TaskState {
    /// Task has been registered and queued, but has not yet started execution.
    Queued,
    /// Task is actively executing.
    Running,
    /// Task finished successfully.
    Completed,
    /// Task failed with an error or diagnostic.
    Failed,
    /// Task was cancelled by a host request or cancellation token.
    Cancelled,
}

impl TaskState {
    /// Returns `true` once the task has reached a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_state_classification() {
        assert!(!TaskState::Queued.is_terminal());
        assert!(!TaskState::Running.is_terminal());
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Failed.is_terminal());
        assert!(TaskState::Cancelled.is_terminal());
    }

    #[test]
    fn serde_representation_matches_run_state() {
        let serialized = serde_json::to_string(&TaskState::Queued).expect("serialize");
        assert_eq!(serialized, "\"Queued\"");
        let deserialized: TaskState = serde_json::from_str("\"Cancelled\"").expect("deserialize");
        assert_eq!(deserialized, TaskState::Cancelled);
    }
}
