//! Collection commands and changes.

use crate::model::{AssetId, SlotId};

/// A command that mutates a collection document.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CollectionCommand {
    /// Change the asset associated with a slot.
    SetSlotAsset { slot_id: SlotId, asset: AssetId },
}

/// A change record produced by applying a collection command.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CollectionChange {
    /// A slot's asset pointer changed.
    SlotAssetChanged {
        slot_id: SlotId,
        before: AssetId,
        after: AssetId,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_serde_roundtrip() {
        let cmd = CollectionCommand::SetSlotAsset {
            slot_id: SlotId::new("slot-1"),
            asset: AssetId::new("asset-1"),
        };
        let json = serde_json::to_string(&cmd).expect("serialize");
        let back: CollectionCommand = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cmd, back);
    }

    #[test]
    fn change_serde_roundtrip() {
        let change = CollectionChange::SlotAssetChanged {
            slot_id: SlotId::new("slot-1"),
            before: AssetId::new("asset-0"),
            after: AssetId::new("asset-1"),
        };
        let json = serde_json::to_string(&change).expect("serialize");
        let back: CollectionChange = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(change, back);
    }
}
