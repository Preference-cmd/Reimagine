//! Board document: the persisted project canvas.

use std::collections::BTreeMap;

use crate::board::command::{BoardChange, BoardCommand};
use crate::board::item::BoardItem;
use crate::board::kind::BoardItemKind;
use crate::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain,
};
use crate::document::{CommandApplication, Document};
use crate::model::{
    BoardId, BoardItemId, DiagnosticId, DocumentVersion, HistoryEntryId, ProjectId,
};

pub const BOARD_SCHEMA_VERSION: &str = "reimagine.board.v1";

/// Board schema version marker.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BoardSchemaVersion(String);

impl BoardSchemaVersion {
    /// Create a new schema version.
    pub fn new(version: impl Into<String>) -> Self {
        Self(version.into())
    }

    /// Get the schema version as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for BoardSchemaVersion {
    fn default() -> Self {
        Self(BOARD_SCHEMA_VERSION.to_owned())
    }
}

/// The board document: a persisted project canvas with spatial items.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BoardDocument {
    schema_version: BoardSchemaVersion,
    id: BoardId,
    project_id: ProjectId,
    version: DocumentVersion,
    items: BTreeMap<BoardItemId, BoardItem>,
}

impl BoardDocument {
    /// Create a new empty board document.
    pub fn new(
        id: impl Into<BoardId>,
        project_id: impl Into<ProjectId>,
        version: DocumentVersion,
    ) -> Self {
        Self {
            schema_version: BoardSchemaVersion::default(),
            id: id.into(),
            project_id: project_id.into(),
            version,
            items: BTreeMap::new(),
        }
    }

    /// Get the schema version.
    pub fn schema_version(&self) -> &BoardSchemaVersion {
        &self.schema_version
    }

    /// Get the board id.
    pub fn id(&self) -> &BoardId {
        &self.id
    }

    /// Get the project id.
    pub fn project_id(&self) -> &ProjectId {
        &self.project_id
    }

    /// Get the board version.
    pub fn version(&self) -> DocumentVersion {
        self.version
    }

    /// Get all items.
    pub fn items(&self) -> &BTreeMap<BoardItemId, BoardItem> {
        &self.items
    }

    /// Get an item by id.
    pub fn item(&self, item_id: &BoardItemId) -> Option<&BoardItem> {
        self.items.get(item_id)
    }

    /// Get a mutable reference to an item by id.
    pub fn item_mut(&mut self, item_id: &BoardItemId) -> Option<&mut BoardItem> {
        self.items.get_mut(item_id)
    }

    /// Add an item to the board.
    pub(crate) fn add_item(&mut self, item: BoardItem) {
        self.items.insert(item.id().clone(), item);
    }

    /// Remove an item from the board, returning it if it existed.
    pub(crate) fn remove_item(&mut self, item_id: &BoardItemId) -> Option<BoardItem> {
        self.items.remove(item_id)
    }

    /// Set the board version.
    pub fn set_version(&mut self, version: DocumentVersion) {
        self.version = version;
    }
}

impl Document for BoardDocument {
    type Command = BoardCommand;
    type Change = BoardChange;
    type Context = ();

    fn version(&self) -> DocumentVersion {
        self.version
    }

    fn set_version(&mut self, version: DocumentVersion) {
        self.version = version;
    }

    fn apply_command(
        &mut self,
        command: &Self::Command,
        _context: &Self::Context,
        correlation_id: Option<&CorrelationId>,
    ) -> Result<CommandApplication<Self::Change>, Vec<Diagnostic>> {
        let mut forward_changes = Vec::new();
        let mut inverse_steps = Vec::new();
        let mut diagnostics = Vec::new();

        apply_board_command(
            self,
            command,
            &mut forward_changes,
            &mut inverse_steps,
            &mut diagnostics,
            correlation_id,
        );

        if !diagnostics.is_empty() {
            Err(diagnostics)
        } else {
            let inverse = inverse_steps.into_iter().flatten().collect();
            Ok(CommandApplication::new(forward_changes, inverse))
        }
    }

    fn version_conflict_diagnostic(
        &self,
        current_version: DocumentVersion,
        base_version: DocumentVersion,
        correlation_id: Option<&CorrelationId>,
    ) -> Diagnostic {
        let mut diagnostic = Diagnostic::new(
            DiagnosticId::new(format!(
                "board-version-conflict-{}-{}",
                self.id.as_str(),
                base_version.get()
            )),
            DiagnosticCode::new("CORE/BOARD_VERSION_CONFLICT"),
            DiagnosticSeverity::Error,
            DiagnosticSourceName::new("core"),
            format!(
                "board version conflict: expected {}, got {}",
                current_version, base_version
            ),
            DiagnosticTarget::new(DiagnosticTargetDomain::new("board")).with_id(self.id.as_str()),
        );
        if let Some(correlation_id) = correlation_id.cloned() {
            diagnostic = diagnostic.with_correlation_id(correlation_id);
        }
        diagnostic
    }

    fn history_entry_id(&self, batch_id: &str) -> HistoryEntryId {
        HistoryEntryId::new(format!("board-history:{batch_id}"))
    }
}

fn apply_board_command(
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
                diagnostics.push(item_missing(item_id, correlation_id));
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
                diagnostics.push(item_missing(item_id, correlation_id));
                return;
            };

            if item.is_locked() {
                diagnostics.push(item_locked(item_id, correlation_id));
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
                diagnostics.push(item_missing(item_id, correlation_id));
                return;
            };

            if item.is_locked() {
                diagnostics.push(item_locked(item_id, correlation_id));
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
                diagnostics.push(item_missing(item_id, correlation_id));
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
                diagnostics.push(item_missing(item_id, correlation_id));
                return;
            };

            if item.is_locked() {
                diagnostics.push(item_locked(item_id, correlation_id));
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
                diagnostics.push(item_missing(item_id, correlation_id));
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

fn item_missing(item_id: &BoardItemId, correlation_id: Option<&CorrelationId>) -> Diagnostic {
    item_diagnostic(
        item_id,
        "missing",
        "CORE/BOARD_ITEM_MISSING",
        "board item does not exist",
        correlation_id,
    )
}

fn item_locked(item_id: &BoardItemId, correlation_id: Option<&CorrelationId>) -> Diagnostic {
    item_diagnostic(
        item_id,
        "locked",
        "CORE/BOARD_ITEM_LOCKED",
        "cannot modify locked board item",
        correlation_id,
    )
}

fn item_diagnostic(
    item_id: &BoardItemId,
    kind: &str,
    code: &str,
    message: &str,
    correlation_id: Option<&CorrelationId>,
) -> Diagnostic {
    simple_diagnostic(
        format!("board-item-{kind}-{item_id}"),
        code,
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("board")).with_id(item_id.as_str()),
        correlation_id,
    )
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

    #[test]
    fn board_creation() {
        let id = BoardId::new("board-1");
        let project_id = ProjectId::new("proj-1");
        let version = DocumentVersion::new(1);

        let board = BoardDocument::new(id.clone(), project_id.clone(), version);
        assert_eq!(board.id(), &id);
        assert_eq!(board.project_id(), &project_id);
        assert_eq!(board.version(), version);
        assert!(board.items().is_empty());
    }

    #[test]
    fn board_item_add_remove() {
        let mut board = BoardDocument::new(
            BoardId::new("board-2"),
            ProjectId::new("proj-1"),
            DocumentVersion::new(1),
        );

        let item = BoardItem::new(
            BoardItemId::new("item-1"),
            BoardItemKind::Note {
                content: "Test note".to_string(),
            },
            BoardItemPosition::new(0, 0),
            BoardItemSize::new(100, 100),
        );

        board.add_item(item.clone());
        assert_eq!(board.items().len(), 1);
        assert!(board.item(&BoardItemId::new("item-1")).is_some());

        let removed = board.remove_item(&BoardItemId::new("item-1"));
        assert!(removed.is_some());
        assert_eq!(removed.unwrap(), item);
        assert!(board.items().is_empty());
    }

    #[test]
    fn board_serde_roundtrip() {
        let mut board = BoardDocument::new(
            BoardId::new("board-serde"),
            ProjectId::new("proj-serde"),
            DocumentVersion::new(1),
        );

        let item = BoardItem::new(
            BoardItemId::new("item-serde"),
            BoardItemKind::WorkflowRef {
                workflow_id: crate::model::WorkflowId::new("wf-1"),
            },
            BoardItemPosition::new(100, 200),
            BoardItemSize::new(300, 150),
        );

        board.add_item(item);

        let json = serde_json::to_string(&board).expect("serialize");
        let back: BoardDocument = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(board, back);
    }
}
