use crate::command::{CommandActor, CommandBatch, CommandProvenance, WorkflowChange};
use crate::event::Timestamp;
use crate::model::HistoryEntryId;
use crate::workflow::Workflow;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HistoryEntry {
    id: HistoryEntryId,
    actor: CommandActor,
    provenance: CommandProvenance,
    command_batch: CommandBatch,
    before: Workflow,
    after: Workflow,
    forward_changes: Vec<WorkflowChange>,
    inverse_changes: Vec<WorkflowChange>,
    created_at: Timestamp,
}

impl HistoryEntry {
    pub fn new(
        id: HistoryEntryId,
        command_batch: CommandBatch,
        before: Workflow,
        after: Workflow,
        forward_changes: Vec<WorkflowChange>,
        inverse_changes: Vec<WorkflowChange>,
        created_at: Timestamp,
    ) -> Self {
        Self {
            id,
            actor: command_batch.actor().clone(),
            provenance: command_batch.provenance().clone(),
            command_batch,
            before,
            after,
            forward_changes,
            inverse_changes,
            created_at,
        }
    }

    pub fn id(&self) -> &HistoryEntryId {
        &self.id
    }

    pub fn actor(&self) -> &CommandActor {
        &self.actor
    }

    pub fn provenance(&self) -> &CommandProvenance {
        &self.provenance
    }

    pub fn command_batch(&self) -> &CommandBatch {
        &self.command_batch
    }

    pub fn before(&self) -> &Workflow {
        &self.before
    }

    pub fn after(&self) -> &Workflow {
        &self.after
    }

    pub fn forward_changes(&self) -> &[WorkflowChange] {
        &self.forward_changes
    }

    pub fn inverse_changes(&self) -> &[WorkflowChange] {
        &self.inverse_changes
    }

    pub fn created_at(&self) -> &Timestamp {
        &self.created_at
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowHistory {
    entries: Vec<HistoryEntry>,
    cursor: usize,
}

impl WorkflowHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub(crate) fn truncate_to_cursor(&mut self) {
        self.entries.truncate(self.cursor);
    }

    pub(crate) fn push(&mut self, entry: HistoryEntry) {
        self.entries.push(entry);
        self.cursor = self.entries.len();
    }

    pub(crate) fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub(crate) fn can_redo(&self) -> bool {
        self.cursor < self.entries.len()
    }

    pub(crate) fn entry_to_undo(&self) -> Option<&HistoryEntry> {
        self.can_undo().then(|| &self.entries[self.cursor - 1])
    }

    pub(crate) fn entry_to_redo(&self) -> Option<&HistoryEntry> {
        self.can_redo().then(|| &self.entries[self.cursor])
    }

    pub(crate) fn move_cursor_back(&mut self) {
        self.cursor -= 1;
    }

    pub(crate) fn move_cursor_forward(&mut self) {
        self.cursor += 1;
    }
}
