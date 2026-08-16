//! Board document: the persisted project canvas.

use std::collections::BTreeMap;

use crate::model::{BoardId, BoardItemId, BoardVersion, ProjectId};

use super::item::BoardItem;

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
    version: BoardVersion,
    items: BTreeMap<BoardItemId, BoardItem>,
}

impl BoardDocument {
    /// Create a new empty board document.
    pub fn new(
        id: impl Into<BoardId>,
        project_id: impl Into<ProjectId>,
        version: BoardVersion,
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
    pub fn version(&self) -> BoardVersion {
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
    pub(crate) fn set_version(&mut self, version: BoardVersion) {
        self.version = version;
    }
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
        let version = BoardVersion::new(1);

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
            BoardVersion::new(1),
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
            BoardVersion::new(1),
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
