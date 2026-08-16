//! Board commands: operations that manipulate the board document.

use crate::command::{CommandActor, CommandProvenance};
use crate::diagnostic::{CorrelationId, Diagnostic};
use crate::event::Timestamp;
use crate::model::{BoardItemId, BoardVersion, CommandBatchId, HistoryEntryId};

use super::document::BoardDocument;
use super::item::{BoardItemPosition, BoardItemSize};
use super::kind::BoardItemKind;

/// A command that manipulates the board.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BoardCommand {
    /// Add a new item to the board.
    AddItem {
        item_id: BoardItemId,
        kind: BoardItemKind,
        position: BoardItemPosition,
        size: BoardItemSize,
    },
    /// Remove an item from the board.
    RemoveItem { item_id: BoardItemId },
    /// Move an item to a new position.
    MoveItem {
        item_id: BoardItemId,
        position: BoardItemPosition,
    },
    /// Resize an item.
    ResizeItem {
        item_id: BoardItemId,
        size: BoardItemSize,
    },
    /// Set the z-index of an item.
    SetZ { item_id: BoardItemId, z: i32 },
    /// Update the content of a note item.
    UpdateNote {
        item_id: BoardItemId,
        content: String,
    },
    /// Lock or unlock an item.
    LockItem { item_id: BoardItemId, locked: bool },
}

/// A batch of board commands.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BoardCommandBatch {
    id: CommandBatchId,
    actor: CommandActor,
    base_version: BoardVersion,
    provenance: CommandProvenance,
    created_at: Timestamp,
    correlation_id: Option<CorrelationId>,
    commands: Vec<BoardCommand>,
}

impl BoardCommandBatch {
    /// Create a new command batch.
    pub fn new(
        id: CommandBatchId,
        actor: CommandActor,
        base_version: BoardVersion,
        provenance: CommandProvenance,
        created_at: Timestamp,
        commands: Vec<BoardCommand>,
    ) -> Self {
        Self {
            id,
            actor,
            base_version,
            provenance,
            created_at,
            correlation_id: None,
            commands,
        }
    }

    /// Attach a correlation id.
    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Get the batch id.
    pub fn id(&self) -> &CommandBatchId {
        &self.id
    }

    /// Get the actor.
    pub fn actor(&self) -> &CommandActor {
        &self.actor
    }

    /// Get the base version.
    pub fn base_version(&self) -> BoardVersion {
        self.base_version
    }

    /// Get the provenance.
    pub fn provenance(&self) -> &CommandProvenance {
        &self.provenance
    }

    /// Get the creation timestamp.
    pub fn created_at(&self) -> &Timestamp {
        &self.created_at
    }

    /// Get the correlation id.
    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    /// Get the commands.
    pub fn commands(&self) -> &[BoardCommand] {
        &self.commands
    }
}

/// Status of a board command result.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum BoardCommandResultStatus {
    Applied,
    Rejected,
    NoOp,
}

/// A change record for board operations.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BoardChange {
    ItemAdded {
        item_id: BoardItemId,
        kind: BoardItemKind,
        position: BoardItemPosition,
        size: BoardItemSize,
        z: i32,
    },
    ItemRemoved {
        item_id: BoardItemId,
        kind: BoardItemKind,
        position: BoardItemPosition,
        size: BoardItemSize,
        z: i32,
        locked: bool,
    },
    ItemMoved {
        item_id: BoardItemId,
        before: BoardItemPosition,
        after: BoardItemPosition,
    },
    ItemResized {
        item_id: BoardItemId,
        before: BoardItemSize,
        after: BoardItemSize,
    },
    ItemZChanged {
        item_id: BoardItemId,
        before: i32,
        after: i32,
    },
    NoteUpdated {
        item_id: BoardItemId,
        before: String,
        after: String,
    },
    ItemLocked {
        item_id: BoardItemId,
        before: bool,
        after: bool,
    },
    VersionAdvanced {
        before: BoardVersion,
        after: BoardVersion,
    },
}

/// Result of applying a board command batch.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BoardCommandResult {
    status: BoardCommandResultStatus,
    board_version: BoardVersion,
    changes: Vec<BoardChange>,
    diagnostics: Vec<Diagnostic>,
    history_entry_id: Option<HistoryEntryId>,
}

impl BoardCommandResult {
    /// Create a new command result.
    pub fn new(
        status: BoardCommandResultStatus,
        board_version: BoardVersion,
        changes: Vec<BoardChange>,
        diagnostics: Vec<Diagnostic>,
        history_entry_id: Option<HistoryEntryId>,
    ) -> Self {
        Self {
            status,
            board_version,
            changes,
            diagnostics,
            history_entry_id,
        }
    }

    /// Get the status.
    pub fn status(&self) -> BoardCommandResultStatus {
        self.status.clone()
    }

    /// Get the board version.
    pub fn board_version(&self) -> BoardVersion {
        self.board_version
    }

    /// Get the changes.
    pub fn changes(&self) -> &[BoardChange] {
        &self.changes
    }

    /// Get the diagnostics.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Get the history entry id.
    pub fn history_entry_id(&self) -> Option<&HistoryEntryId> {
        self.history_entry_id.as_ref()
    }
}

/// History entry for a board command batch.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BoardHistoryEntry {
    id: HistoryEntryId,
    actor: CommandActor,
    provenance: CommandProvenance,
    command_batch: BoardCommandBatch,
    before: BoardDocument,
    after: BoardDocument,
    forward_changes: Vec<BoardChange>,
    inverse_changes: Vec<BoardChange>,
    created_at: Timestamp,
}

impl BoardHistoryEntry {
    /// Create a new history entry.
    pub fn new(
        id: HistoryEntryId,
        command_batch: BoardCommandBatch,
        before: BoardDocument,
        after: BoardDocument,
        forward_changes: Vec<BoardChange>,
        inverse_changes: Vec<BoardChange>,
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

    /// Get the entry id.
    pub fn id(&self) -> &HistoryEntryId {
        &self.id
    }

    /// Get the actor.
    pub fn actor(&self) -> &CommandActor {
        &self.actor
    }

    /// Get the provenance.
    pub fn provenance(&self) -> &CommandProvenance {
        &self.provenance
    }

    /// Get the command batch.
    pub fn command_batch(&self) -> &BoardCommandBatch {
        &self.command_batch
    }

    /// Get the before state.
    pub fn before(&self) -> &BoardDocument {
        &self.before
    }

    /// Get the after state.
    pub fn after(&self) -> &BoardDocument {
        &self.after
    }

    /// Get the forward changes.
    pub fn forward_changes(&self) -> &[BoardChange] {
        &self.forward_changes
    }

    /// Get the inverse changes.
    pub fn inverse_changes(&self) -> &[BoardChange] {
        &self.inverse_changes
    }

    /// Get the creation timestamp.
    pub fn created_at(&self) -> &Timestamp {
        &self.created_at
    }
}

/// Board history with undo/redo cursor.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BoardHistory {
    entries: Vec<BoardHistoryEntry>,
    cursor: usize,
}

impl BoardHistory {
    /// Create a new empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the entries.
    pub fn entries(&self) -> &[BoardHistoryEntry] {
        &self.entries
    }

    /// Get the cursor position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Check if undo is possible.
    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    /// Check if redo is possible.
    pub fn can_redo(&self) -> bool {
        self.cursor < self.entries.len()
    }

    pub(crate) fn truncate_to_cursor(&mut self) {
        self.entries.truncate(self.cursor);
    }

    pub(crate) fn push(&mut self, entry: BoardHistoryEntry) {
        self.entries.push(entry);
        self.cursor = self.entries.len();
    }

    pub(crate) fn entry_to_undo(&self) -> Option<&BoardHistoryEntry> {
        self.can_undo().then(|| &self.entries[self.cursor - 1])
    }

    pub(crate) fn entry_to_redo(&self) -> Option<&BoardHistoryEntry> {
        self.can_redo().then(|| &self.entries[self.cursor])
    }

    pub(crate) fn move_cursor_back(&mut self) {
        self.cursor -= 1;
    }

    pub(crate) fn move_cursor_forward(&mut self) {
        self.cursor += 1;
    }
}
