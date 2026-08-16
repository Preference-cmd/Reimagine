//! Board item: a spatial element on the project canvas.

use crate::model::BoardItemId;

use super::kind::BoardItemKind;

/// Position of a board item in 2D canvas space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BoardItemPosition {
    pub x: i64,
    pub y: i64,
}

impl BoardItemPosition {
    /// Create a new position.
    pub fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }
}

/// Size of a board item in canvas space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BoardItemSize {
    pub width: u64,
    pub height: u64,
}

impl BoardItemSize {
    /// Create a new size.
    pub fn new(width: u64, height: u64) -> Self {
        Self { width, height }
    }
}

/// A single item on the board canvas.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BoardItem {
    id: BoardItemId,
    kind: BoardItemKind,
    position: BoardItemPosition,
    size: BoardItemSize,
    z: i32,
    locked: bool,
}

impl BoardItem {
    /// Create a new board item.
    pub fn new(
        id: BoardItemId,
        kind: BoardItemKind,
        position: BoardItemPosition,
        size: BoardItemSize,
    ) -> Self {
        Self {
            id,
            kind,
            position,
            size,
            z: 0,
            locked: false,
        }
    }

    /// Set the z-index.
    pub fn with_z(mut self, z: i32) -> Self {
        self.z = z;
        self
    }

    /// Set the locked state.
    pub fn with_locked(mut self, locked: bool) -> Self {
        self.locked = locked;
        self
    }

    /// Get the item id.
    pub fn id(&self) -> &BoardItemId {
        &self.id
    }

    /// Get the item kind.
    pub fn kind(&self) -> &BoardItemKind {
        &self.kind
    }

    /// Get a mutable reference to the item kind.
    pub fn kind_mut(&mut self) -> &mut BoardItemKind {
        &mut self.kind
    }

    /// Get the item position.
    pub fn position(&self) -> &BoardItemPosition {
        &self.position
    }

    /// Get a mutable reference to the item position.
    pub fn position_mut(&mut self) -> &mut BoardItemPosition {
        &mut self.position
    }

    /// Get the item size.
    pub fn size(&self) -> &BoardItemSize {
        &self.size
    }

    /// Get a mutable reference to the item size.
    pub fn size_mut(&mut self) -> &mut BoardItemSize {
        &mut self.size
    }

    /// Get the z-index.
    pub fn z(&self) -> i32 {
        self.z
    }

    /// Set the z-index.
    pub fn set_z(&mut self, z: i32) {
        self.z = z;
    }

    /// Check if the item is locked.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Set the locked state.
    pub fn set_locked(&mut self, locked: bool) {
        self.locked = locked;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::kind::BoardItemKind;
    use crate::model::BoardItemId;

    #[test]
    fn item_creation() {
        let id = BoardItemId::new("item-1");
        let kind = BoardItemKind::Note {
            content: "Test note".to_string(),
        };
        let pos = BoardItemPosition::new(100, 200);
        let size = BoardItemSize::new(300, 150);

        let item = BoardItem::new(id.clone(), kind.clone(), pos, size);
        assert_eq!(item.id(), &id);
        assert_eq!(item.kind(), &kind);
        assert_eq!(item.position().x, 100);
        assert_eq!(item.position().y, 200);
        assert_eq!(item.size().width, 300);
        assert_eq!(item.size().height, 150);
        assert_eq!(item.z(), 0);
        assert!(!item.is_locked());
    }

    #[test]
    fn item_builder_methods() {
        let id = BoardItemId::new("item-2");
        let kind = BoardItemKind::Note {
            content: "Test".to_string(),
        };
        let pos = BoardItemPosition::new(0, 0);
        let size = BoardItemSize::new(100, 100);

        let item = BoardItem::new(id, kind, pos, size)
            .with_z(5)
            .with_locked(true);
        assert_eq!(item.z(), 5);
        assert!(item.is_locked());
    }

    #[test]
    fn item_mutation() {
        let id = BoardItemId::new("item-3");
        let kind = BoardItemKind::Note {
            content: "Original".to_string(),
        };
        let pos = BoardItemPosition::new(10, 20);
        let size = BoardItemSize::new(100, 100);

        let mut item = BoardItem::new(id, kind, pos, size);

        // Mutate position
        item.position_mut().x = 50;
        item.position_mut().y = 60;
        assert_eq!(item.position().x, 50);
        assert_eq!(item.position().y, 60);

        // Mutate size
        item.size_mut().width = 200;
        item.size_mut().height = 200;
        assert_eq!(item.size().width, 200);
        assert_eq!(item.size().height, 200);

        // Mutate z and locked
        item.set_z(10);
        item.set_locked(true);
        assert_eq!(item.z(), 10);
        assert!(item.is_locked());
    }

    #[test]
    fn item_serde_roundtrip() {
        let id = BoardItemId::new("item-serde");
        let kind = BoardItemKind::WorkflowRef {
            workflow_id: crate::model::WorkflowId::new("wf-1"),
        };
        let pos = BoardItemPosition::new(100, 200);
        let size = BoardItemSize::new(300, 150);

        let item = BoardItem::new(id, kind, pos, size)
            .with_z(3)
            .with_locked(false);
        let json = serde_json::to_string(&item).expect("serialize");
        let back: BoardItem = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(item, back);
    }
}
