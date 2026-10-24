//! Generic history tracking for document sessions.

use crate::command::CommandBatch;
use crate::model::HistoryEntryId;

use super::traits::Document;

/// History entry for a command batch applied to a document.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEntry<D: Document> {
    id: HistoryEntryId,
    command_batch: CommandBatch<D::Command>,
    before: D,
    after: D,
    forward_changes: Vec<D::Change>,
    inverse_changes: Vec<D::Change>,
}

impl<D: Document> HistoryEntry<D> {
    pub fn new(
        id: HistoryEntryId,
        command_batch: CommandBatch<D::Command>,
        before: D,
        after: D,
        forward_changes: Vec<D::Change>,
        inverse_changes: Vec<D::Change>,
    ) -> Self {
        Self {
            id,
            command_batch,
            before,
            after,
            forward_changes,
            inverse_changes,
        }
    }

    pub fn id(&self) -> &HistoryEntryId {
        &self.id
    }

    pub fn command_batch(&self) -> &CommandBatch<D::Command> {
        &self.command_batch
    }

    pub fn before(&self) -> &D {
        &self.before
    }

    pub fn after(&self) -> &D {
        &self.after
    }

    pub fn forward_changes(&self) -> &[D::Change] {
        &self.forward_changes
    }

    pub fn inverse_changes(&self) -> &[D::Change] {
        &self.inverse_changes
    }
}

/// Generic document undo/redo history.
#[derive(Debug, Clone, PartialEq)]
pub struct History<D: Document> {
    entries: Vec<HistoryEntry<D>>,
    cursor: usize,
}

impl<D: Document> Default for History<D> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            cursor: 0,
        }
    }
}

impl<D: Document> History<D> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[HistoryEntry<D>] {
        &self.entries
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        self.cursor < self.entries.len()
    }

    pub(crate) fn truncate_to_cursor(&mut self) {
        self.entries.truncate(self.cursor);
    }

    pub(crate) fn push(&mut self, entry: HistoryEntry<D>) {
        self.entries.push(entry);
        self.cursor = self.entries.len();
    }

    pub(crate) fn entry_to_undo(&self) -> Option<&HistoryEntry<D>> {
        self.can_undo().then(|| &self.entries[self.cursor - 1])
    }

    pub(crate) fn entry_to_redo(&self) -> Option<&HistoryEntry<D>> {
        self.can_redo().then(|| &self.entries[self.cursor])
    }

    pub(crate) fn move_cursor_back(&mut self) {
        self.cursor -= 1;
    }

    pub(crate) fn move_cursor_forward(&mut self) {
        self.cursor += 1;
    }
}
