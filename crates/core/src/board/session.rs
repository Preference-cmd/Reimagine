//! Board session: preview, apply, undo, and redo board commands.

use crate::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain,
};
use crate::model::{BoardVersion, DiagnosticId, HistoryEntryId};

use super::command::{
    BoardChange, BoardCommand, BoardCommandBatch, BoardCommandResult, BoardCommandResultStatus,
    BoardHistory, BoardHistoryEntry,
};
use super::document::BoardDocument;
use super::item::BoardItem;
use super::kind::BoardItemKind;

/// A board session that manages preview, apply, undo, and redo operations.
pub struct BoardSession {
    board: BoardDocument,
    history: BoardHistory,
}

impl BoardSession {
    /// Create a new board session.
    pub fn new(board: BoardDocument) -> Self {
        Self {
            board,
            history: BoardHistory::new(),
        }
    }

    /// Get a reference to the board.
    pub fn board(&self) -> &BoardDocument {
        &self.board
    }

    /// Get the current board version.
    pub fn version(&self) -> BoardVersion {
        self.board.version()
    }

    /// Get a reference to the history.
    pub fn history(&self) -> &BoardHistory {
        &self.history
    }

    /// Preview a command batch without applying it.
    pub fn preview_batch(&self, batch: BoardCommandBatch) -> BoardCommandResult {
        match self.evaluate_batch(&batch) {
            BatchEvaluation::Rejected { diagnostics } => BoardCommandResult::new(
                BoardCommandResultStatus::Rejected,
                self.version(),
                Vec::new(),
                diagnostics,
                None,
            ),
            BatchEvaluation::NoOp => BoardCommandResult::new(
                BoardCommandResultStatus::NoOp,
                self.version(),
                Vec::new(),
                Vec::new(),
                None,
            ),
            BatchEvaluation::Applied {
                projected_version,
                forward_changes,
                ..
            } => BoardCommandResult::new(
                BoardCommandResultStatus::Applied,
                projected_version,
                forward_changes,
                Vec::new(),
                None,
            ),
        }
    }

    /// Apply a command batch to the board.
    pub fn apply_batch(&mut self, batch: BoardCommandBatch) -> BoardCommandResult {
        match self.evaluate_batch(&batch) {
            BatchEvaluation::Rejected { diagnostics } => BoardCommandResult::new(
                BoardCommandResultStatus::Rejected,
                self.version(),
                Vec::new(),
                diagnostics,
                None,
            ),
            BatchEvaluation::NoOp => BoardCommandResult::new(
                BoardCommandResultStatus::NoOp,
                self.version(),
                Vec::new(),
                Vec::new(),
                None,
            ),
            BatchEvaluation::Applied {
                board,
                projected_version,
                forward_changes,
                inverse_changes,
            } => {
                let before = self.board.clone();
                self.board = board;
                self.board.set_version(projected_version);

                let history_entry_id = history_entry_id(batch.id().as_str());
                let history_entry = BoardHistoryEntry::new(
                    history_entry_id.clone(),
                    batch.clone(),
                    before,
                    self.board.clone(),
                    forward_changes.clone(),
                    inverse_changes,
                    batch.created_at().clone(),
                );

                self.history.truncate_to_cursor();
                self.history.push(history_entry);

                BoardCommandResult::new(
                    BoardCommandResultStatus::Applied,
                    projected_version,
                    forward_changes,
                    Vec::new(),
                    Some(history_entry_id),
                )
            }
        }
    }

    /// Undo the last applied command batch.
    pub fn undo(&mut self) -> Option<BoardCommandResult> {
        let entry = self.history.entry_to_undo()?.clone();
        let current_version = self.version();
        let next_version = increment_version(current_version);
        let mut restored = entry.before().clone();
        restored.set_version(next_version);

        self.board = restored;
        self.history.move_cursor_back();

        let mut changes = entry.inverse_changes().to_vec();
        changes.push(BoardChange::VersionAdvanced {
            before: current_version,
            after: next_version,
        });

        Some(BoardCommandResult::new(
            BoardCommandResultStatus::Applied,
            next_version,
            changes,
            Vec::new(),
            None,
        ))
    }

    /// Redo the last undone command batch.
    pub fn redo(&mut self) -> Option<BoardCommandResult> {
        let entry = self.history.entry_to_redo()?.clone();
        let current_version = self.version();
        let next_version = increment_version(current_version);
        let mut restored = entry.after().clone();
        restored.set_version(next_version);

        self.board = restored;
        self.history.move_cursor_forward();

        let mut changes = without_version_change(entry.forward_changes());
        changes.push(BoardChange::VersionAdvanced {
            before: current_version,
            after: next_version,
        });

        Some(BoardCommandResult::new(
            BoardCommandResultStatus::Applied,
            next_version,
            changes,
            Vec::new(),
            None,
        ))
    }

    fn evaluate_batch(&self, batch: &BoardCommandBatch) -> BatchEvaluation {
        let current_version = self.version();
        if batch.base_version() != current_version {
            return BatchEvaluation::Rejected {
                diagnostics: vec![version_conflict_diagnostic(
                    self.board.id().as_str(),
                    current_version,
                    batch.base_version(),
                    batch.correlation_id(),
                )],
            };
        }

        let mut working = self.board.clone();
        let mut forward_changes = Vec::new();
        let mut inverse_steps = Vec::<Vec<BoardChange>>::new();
        let mut diagnostics = Vec::new();

        for command in batch.commands() {
            apply_command(
                &mut working,
                command,
                &mut forward_changes,
                &mut inverse_steps,
                &mut diagnostics,
                batch.correlation_id(),
            );
        }

        if !diagnostics.is_empty() {
            return BatchEvaluation::Rejected { diagnostics };
        }

        if forward_changes.is_empty() {
            return BatchEvaluation::NoOp;
        }

        let projected_version = increment_version(current_version);
        forward_changes.push(BoardChange::VersionAdvanced {
            before: current_version,
            after: projected_version,
        });

        let inverse_changes = inverse_steps.into_iter().rev().flatten().collect();

        BatchEvaluation::Applied {
            board: working,
            projected_version,
            forward_changes,
            inverse_changes,
        }
    }
}

enum BatchEvaluation {
    Rejected {
        diagnostics: Vec<Diagnostic>,
    },
    NoOp,
    Applied {
        board: BoardDocument,
        projected_version: BoardVersion,
        forward_changes: Vec<BoardChange>,
        inverse_changes: Vec<BoardChange>,
    },
}

fn apply_command(
    board: &mut BoardDocument,
    command: &BoardCommand,
    forward_changes: &mut Vec<BoardChange>,
    inverse_steps: &mut Vec<Vec<BoardChange>>,
    diagnostics: &mut Vec<Diagnostic>,
    correlation_id: Option<&CorrelationId>,
) {
    match command {
        BoardCommand::AddItem {
            item_id,
            kind,
            position,
            size,
        } => {
            if board.item(item_id).is_some() {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-duplicate-{item_id}"),
                    "CORE/BOARD_ITEM_DUPLICATE",
                    "board item id already exists",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            }

            let item = BoardItem::new(item_id.clone(), kind.clone(), *position, *size);
            board.add_item(item);

            forward_changes.push(BoardChange::ItemAdded {
                item_id: item_id.clone(),
                kind: kind.clone(),
                position: *position,
                size: *size,
                z: 0,
            });

            inverse_steps.push(vec![BoardChange::ItemRemoved {
                item_id: item_id.clone(),
                kind: kind.clone(),
                position: *position,
                size: *size,
                z: 0,
                locked: false,
            }]);
        }
        BoardCommand::RemoveItem { item_id } => {
            let Some(item) = board.item(item_id).cloned() else {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-missing-{item_id}"),
                    "CORE/BOARD_ITEM_MISSING",
                    "board item does not exist",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            };

            board.remove_item(item_id);

            forward_changes.push(BoardChange::ItemRemoved {
                item_id: item_id.clone(),
                kind: item.kind().clone(),
                position: *item.position(),
                size: *item.size(),
                z: item.z(),
                locked: item.is_locked(),
            });

            inverse_steps.push(vec![BoardChange::ItemAdded {
                item_id: item_id.clone(),
                kind: item.kind().clone(),
                position: *item.position(),
                size: *item.size(),
                z: item.z(),
            }]);
        }
        BoardCommand::MoveItem { item_id, position } => {
            let Some(item) = board.item_mut(item_id) else {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-missing-{item_id}"),
                    "CORE/BOARD_ITEM_MISSING",
                    "board item does not exist",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            };

            if item.is_locked() {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-locked-{item_id}"),
                    "CORE/BOARD_ITEM_LOCKED",
                    "cannot modify locked board item",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            }

            let before = *item.position();
            if before == *position {
                return;
            }

            item.position_mut().x = position.x;
            item.position_mut().y = position.y;

            forward_changes.push(BoardChange::ItemMoved {
                item_id: item_id.clone(),
                before,
                after: *position,
            });

            inverse_steps.push(vec![BoardChange::ItemMoved {
                item_id: item_id.clone(),
                before: *position,
                after: before,
            }]);
        }
        BoardCommand::ResizeItem { item_id, size } => {
            let Some(item) = board.item_mut(item_id) else {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-missing-{item_id}"),
                    "CORE/BOARD_ITEM_MISSING",
                    "board item does not exist",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            };

            if item.is_locked() {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-locked-{item_id}"),
                    "CORE/BOARD_ITEM_LOCKED",
                    "cannot modify locked board item",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            }

            let before = *item.size();
            if before == *size {
                return;
            }

            item.size_mut().width = size.width;
            item.size_mut().height = size.height;

            forward_changes.push(BoardChange::ItemResized {
                item_id: item_id.clone(),
                before,
                after: *size,
            });

            inverse_steps.push(vec![BoardChange::ItemResized {
                item_id: item_id.clone(),
                before: *size,
                after: before,
            }]);
        }
        BoardCommand::SetZ { item_id, z } => {
            let Some(item) = board.item_mut(item_id) else {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-missing-{item_id}"),
                    "CORE/BOARD_ITEM_MISSING",
                    "board item does not exist",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            };

            let before = item.z();
            if before == *z {
                return;
            }

            item.set_z(*z);

            forward_changes.push(BoardChange::ItemZChanged {
                item_id: item_id.clone(),
                before,
                after: *z,
            });

            inverse_steps.push(vec![BoardChange::ItemZChanged {
                item_id: item_id.clone(),
                before: *z,
                after: before,
            }]);
        }
        BoardCommand::UpdateNote { item_id, content } => {
            let Some(item) = board.item_mut(item_id) else {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-missing-{item_id}"),
                    "CORE/BOARD_ITEM_MISSING",
                    "board item does not exist",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            };

            if item.is_locked() {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-locked-{item_id}"),
                    "CORE/BOARD_ITEM_LOCKED",
                    "cannot modify locked board item",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            }

            let BoardItemKind::Note {
                content: old_content,
            } = item.kind_mut()
            else {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-not-note-{item_id}"),
                    "CORE/BOARD_ITEM_NOT_NOTE",
                    "board item is not a note",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            };

            if *old_content == *content {
                return;
            }

            let before = old_content.clone();
            *old_content = content.clone();

            forward_changes.push(BoardChange::NoteUpdated {
                item_id: item_id.clone(),
                before: before.clone(),
                after: content.clone(),
            });

            inverse_steps.push(vec![BoardChange::NoteUpdated {
                item_id: item_id.clone(),
                before: content.clone(),
                after: before,
            }]);
        }
        BoardCommand::LockItem { item_id, locked } => {
            let Some(item) = board.item_mut(item_id) else {
                diagnostics.push(simple_diagnostic(
                    format!("board-item-missing-{item_id}"),
                    "CORE/BOARD_ITEM_MISSING",
                    "board item does not exist",
                    DiagnosticTarget::new(DiagnosticTargetDomain::new("board"))
                        .with_id(item_id.as_str()),
                    correlation_id,
                ));
                return;
            };

            let before = item.is_locked();
            if before == *locked {
                return;
            }

            item.set_locked(*locked);

            forward_changes.push(BoardChange::ItemLocked {
                item_id: item_id.clone(),
                before,
                after: *locked,
            });

            inverse_steps.push(vec![BoardChange::ItemLocked {
                item_id: item_id.clone(),
                before: *locked,
                after: before,
            }]);
        }
    }
}

fn without_version_change(changes: &[BoardChange]) -> Vec<BoardChange> {
    changes
        .iter()
        .filter(|change| !matches!(change, BoardChange::VersionAdvanced { .. }))
        .cloned()
        .collect()
}

fn increment_version(version: BoardVersion) -> BoardVersion {
    BoardVersion::new(version.get() + 1)
}

fn history_entry_id(batch_id: &str) -> HistoryEntryId {
    HistoryEntryId::new(format!("board-history:{batch_id}"))
}

fn version_conflict_diagnostic(
    board_id: &str,
    current_version: BoardVersion,
    batch_version: BoardVersion,
    correlation_id: Option<&CorrelationId>,
) -> Diagnostic {
    let mut diagnostic = Diagnostic::new(
        DiagnosticId::new(format!(
            "board-version-conflict-{}-{}",
            board_id,
            batch_version.get()
        )),
        DiagnosticCode::new("CORE/BOARD_VERSION_CONFLICT"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("core"),
        format!(
            "board version conflict: expected {}, got {}",
            current_version, batch_version
        ),
        DiagnosticTarget::new(DiagnosticTargetDomain::new("board")).with_id(board_id),
    );
    if let Some(correlation_id) = correlation_id.cloned() {
        diagnostic = diagnostic.with_correlation_id(correlation_id);
    }
    diagnostic
}

fn simple_diagnostic(
    id: String,
    code: &str,
    message: &str,
    primary: DiagnosticTarget,
    correlation_id: Option<&CorrelationId>,
) -> Diagnostic {
    let mut diagnostic = Diagnostic::new(
        DiagnosticId::new(id),
        DiagnosticCode::new(code),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("core"),
        message,
        primary,
    );
    if let Some(correlation_id) = correlation_id.cloned() {
        diagnostic = diagnostic.with_correlation_id(correlation_id);
    }
    diagnostic
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::item::{BoardItemPosition, BoardItemSize};
    use crate::board::kind::BoardItemKind;
    use crate::command::{CommandActor, CommandProvenance};
    use crate::event::Timestamp;
    use crate::model::{BoardId, BoardItemId, CommandBatchId, ProjectId, WorkflowId};

    fn test_board() -> BoardDocument {
        BoardDocument::new(
            BoardId::new("test-board"),
            ProjectId::new("test-project"),
            BoardVersion::new(0),
        )
    }

    fn test_batch(board_version: BoardVersion, commands: Vec<BoardCommand>) -> BoardCommandBatch {
        BoardCommandBatch::new(
            CommandBatchId::new("batch-1"),
            CommandActor::new(crate::command::CommandActorKind::Human),
            board_version,
            CommandProvenance::Direct,
            Timestamp::new("2026-01-01T00:00:00Z"),
            commands,
        )
    }

    #[test]
    fn preview_does_not_modify_state() {
        let session = BoardSession::new(test_board());
        let batch = test_batch(
            BoardVersion::new(0),
            vec![BoardCommand::AddItem {
                item_id: BoardItemId::new("item-1"),
                kind: BoardItemKind::Note {
                    content: "Test".to_string(),
                },
                position: BoardItemPosition::new(0, 0),
                size: BoardItemSize::new(100, 100),
            }],
        );

        let result = session.preview_batch(batch);
        assert_eq!(result.status(), BoardCommandResultStatus::Applied);
        assert!(session.board().items().is_empty());
        assert_eq!(session.version(), BoardVersion::new(0));
    }

    #[test]
    fn apply_add_item() {
        let mut session = BoardSession::new(test_board());
        let batch = test_batch(
            BoardVersion::new(0),
            vec![BoardCommand::AddItem {
                item_id: BoardItemId::new("item-1"),
                kind: BoardItemKind::Note {
                    content: "Test".to_string(),
                },
                position: BoardItemPosition::new(10, 20),
                size: BoardItemSize::new(100, 100),
            }],
        );

        let result = session.apply_batch(batch);
        assert_eq!(result.status(), BoardCommandResultStatus::Applied);
        assert_eq!(session.version(), BoardVersion::new(1));
        assert_eq!(session.board().items().len(), 1);

        let item = session.board().item(&BoardItemId::new("item-1")).unwrap();
        assert_eq!(item.position().x, 10);
        assert_eq!(item.position().y, 20);
    }

    #[test]
    fn version_conflict_rejection() {
        let mut session = BoardSession::new(test_board());
        let batch = test_batch(
            BoardVersion::new(5), // Wrong version
            vec![BoardCommand::AddItem {
                item_id: BoardItemId::new("item-1"),
                kind: BoardItemKind::Note {
                    content: "Test".to_string(),
                },
                position: BoardItemPosition::new(0, 0),
                size: BoardItemSize::new(100, 100),
            }],
        );

        let result = session.apply_batch(batch);
        assert_eq!(result.status(), BoardCommandResultStatus::Rejected);
        assert_eq!(result.diagnostics().len(), 1);
        assert_eq!(
            result.diagnostics()[0].code().as_str(),
            "CORE/BOARD_VERSION_CONFLICT"
        );
    }

    #[test]
    fn undo_redo() {
        let mut session = BoardSession::new(test_board());

        // Apply
        let batch1 = test_batch(
            BoardVersion::new(0),
            vec![BoardCommand::AddItem {
                item_id: BoardItemId::new("item-1"),
                kind: BoardItemKind::Note {
                    content: "Test".to_string(),
                },
                position: BoardItemPosition::new(0, 0),
                size: BoardItemSize::new(100, 100),
            }],
        );
        session.apply_batch(batch1);
        assert_eq!(session.board().items().len(), 1);

        // Undo
        let undo_result = session.undo();
        assert!(undo_result.is_some());
        assert_eq!(session.board().items().len(), 0);

        // Redo
        let redo_result = session.redo();
        assert!(redo_result.is_some());
        assert_eq!(session.board().items().len(), 1);
    }

    #[test]
    fn move_item() {
        let mut session = BoardSession::new(test_board());

        // Add item
        let batch1 = test_batch(
            BoardVersion::new(0),
            vec![BoardCommand::AddItem {
                item_id: BoardItemId::new("item-1"),
                kind: BoardItemKind::Note {
                    content: "Test".to_string(),
                },
                position: BoardItemPosition::new(0, 0),
                size: BoardItemSize::new(100, 100),
            }],
        );
        session.apply_batch(batch1);

        // Move item
        let batch2 = test_batch(
            BoardVersion::new(1),
            vec![BoardCommand::MoveItem {
                item_id: BoardItemId::new("item-1"),
                position: BoardItemPosition::new(50, 60),
            }],
        );
        session.apply_batch(batch2);

        let item = session.board().item(&BoardItemId::new("item-1")).unwrap();
        assert_eq!(item.position().x, 50);
        assert_eq!(item.position().y, 60);

        // Undo move
        session.undo();
        let item = session.board().item(&BoardItemId::new("item-1")).unwrap();
        assert_eq!(item.position().x, 0);
        assert_eq!(item.position().y, 0);
    }

    #[test]
    fn lock_item_prevents_modification() {
        let mut session = BoardSession::new(test_board());

        // Add item
        let batch1 = test_batch(
            BoardVersion::new(0),
            vec![BoardCommand::AddItem {
                item_id: BoardItemId::new("item-1"),
                kind: BoardItemKind::Note {
                    content: "Test".to_string(),
                },
                position: BoardItemPosition::new(0, 0),
                size: BoardItemSize::new(100, 100),
            }],
        );
        session.apply_batch(batch1);

        // Lock item
        let batch2 = test_batch(
            BoardVersion::new(1),
            vec![BoardCommand::LockItem {
                item_id: BoardItemId::new("item-1"),
                locked: true,
            }],
        );
        session.apply_batch(batch2);

        // Try to move locked item
        let batch3 = test_batch(
            BoardVersion::new(2),
            vec![BoardCommand::MoveItem {
                item_id: BoardItemId::new("item-1"),
                position: BoardItemPosition::new(50, 50),
            }],
        );
        let result = session.apply_batch(batch3);
        assert_eq!(result.status(), BoardCommandResultStatus::Rejected);
        assert_eq!(
            result.diagnostics()[0].code().as_str(),
            "CORE/BOARD_ITEM_LOCKED"
        );
    }

    #[test]
    fn update_note() {
        let mut session = BoardSession::new(test_board());

        // Add note
        let batch1 = test_batch(
            BoardVersion::new(0),
            vec![BoardCommand::AddItem {
                item_id: BoardItemId::new("note-1"),
                kind: BoardItemKind::Note {
                    content: "Original".to_string(),
                },
                position: BoardItemPosition::new(0, 0),
                size: BoardItemSize::new(100, 100),
            }],
        );
        session.apply_batch(batch1);

        // Update note
        let batch2 = test_batch(
            BoardVersion::new(1),
            vec![BoardCommand::UpdateNote {
                item_id: BoardItemId::new("note-1"),
                content: "Updated content".to_string(),
            }],
        );
        session.apply_batch(batch2);

        let item = session.board().item(&BoardItemId::new("note-1")).unwrap();
        match item.kind() {
            BoardItemKind::Note { content } => assert_eq!(content, "Updated content"),
            _ => panic!("Expected note"),
        }

        // Undo update
        session.undo();
        let item = session.board().item(&BoardItemId::new("note-1")).unwrap();
        match item.kind() {
            BoardItemKind::Note { content } => assert_eq!(content, "Original"),
            _ => panic!("Expected note"),
        }
    }

    #[test]
    fn update_non_note_fails() {
        let mut session = BoardSession::new(test_board());

        // Add workflow ref
        let batch1 = test_batch(
            BoardVersion::new(0),
            vec![BoardCommand::AddItem {
                item_id: BoardItemId::new("wf-1"),
                kind: BoardItemKind::WorkflowRef {
                    workflow_id: WorkflowId::new("wf-test"),
                },
                position: BoardItemPosition::new(0, 0),
                size: BoardItemSize::new(100, 100),
            }],
        );
        session.apply_batch(batch1);

        // Try to update as note
        let batch2 = test_batch(
            BoardVersion::new(1),
            vec![BoardCommand::UpdateNote {
                item_id: BoardItemId::new("wf-1"),
                content: "Should fail".to_string(),
            }],
        );
        let result = session.apply_batch(batch2);
        assert_eq!(result.status(), BoardCommandResultStatus::Rejected);
        assert_eq!(
            result.diagnostics()[0].code().as_str(),
            "CORE/BOARD_ITEM_NOT_NOTE"
        );
    }
}
