//! Board commands: operations that manipulate the board document.

use crate::model::BoardItemId;

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
}

impl BoardChange {
    /// The item this change applies to.
    pub fn item_id(&self) -> &BoardItemId {
        match self {
            Self::ItemAdded { item_id, .. }
            | Self::ItemRemoved { item_id, .. }
            | Self::ItemMoved { item_id, .. }
            | Self::ItemResized { item_id, .. }
            | Self::ItemZChanged { item_id, .. }
            | Self::NoteUpdated { item_id, .. }
            | Self::ItemLocked { item_id, .. } => item_id,
        }
    }
}
