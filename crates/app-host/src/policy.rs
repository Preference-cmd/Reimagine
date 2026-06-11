use reimagine_core::command::{CommandActorKind, WorkflowCommand};

/// Agent/app-host auto-apply policy over `WorkflowCommand` batches.
///
/// V1: `Agent` mode may auto-apply only low-risk, reversible,
/// editor-only commands. `Build` mode never auto-applies through agent
/// tools; human/host approval is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkflowCommandPolicy;

impl WorkflowCommandPolicy {
    pub fn new() -> Self {
        Self
    }

    /// Returns `true` if the command batch is allowed to be auto-applied
    /// in `Agent` mode.
    ///
    /// V1 rules:
    /// - All commands must be low-risk, reversible, editor-only changes.
    /// - `MoveNode`, `ApplyLayout`, `SetNodeLabel`, and
    ///   `SetWorkflowMetadata` are allowed.
    /// - Graph/data semantic changes must go through proposals in V1.
    pub fn allows_auto_apply(&self, commands: &[WorkflowCommand]) -> bool {
        commands.iter().all(|cmd| Self::is_editor_only(cmd))
    }

    fn is_editor_only(command: &WorkflowCommand) -> bool {
        matches!(
            command,
            WorkflowCommand::MoveNode { .. }
                | WorkflowCommand::ApplyLayout { .. }
                | WorkflowCommand::SetNodeLabel { .. }
                | WorkflowCommand::SetWorkflowMetadata { .. }
        )
    }

    /// Returns `true` if the given `CommandActorKind` is permitted to
    /// apply commands through the agent tool path.
    ///
    /// V1: only `Agent` actor kind is permitted for auto-apply;
    /// human approval applies proposals through a host API, not a tool.
    pub fn allowed_actor_kind(&self, kind: CommandActorKind) -> bool {
        matches!(kind, CommandActorKind::Agent)
    }
}

impl Default for WorkflowCommandPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::command::WorkflowCommand;
    use reimagine_core::model::{NodeId, NodeTypeId};

    #[test]
    fn allows_low_risk_editor_commands() {
        let policy = WorkflowCommandPolicy::new();
        let commands = vec![WorkflowCommand::SetNodeLabel {
            node_id: NodeId::new("n1"),
            label: Some("label".into()),
        }];
        assert!(policy.allows_auto_apply(&commands));
    }

    #[test]
    fn rejects_graph_and_data_semantic_commands() {
        let policy = WorkflowCommandPolicy::new();
        let commands = vec![WorkflowCommand::AddNode {
            node_id: NodeId::new("n1"),
            type_id: NodeTypeId::new("t1"),
            label: None,
            params: Default::default(),
            position: None,
        }];
        assert!(!policy.allows_auto_apply(&commands));
    }

    #[test]
    fn empty_batch_is_allowed() {
        let policy = WorkflowCommandPolicy::new();
        assert!(policy.allows_auto_apply(&[]));
    }
}
