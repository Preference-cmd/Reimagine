//! Collection slot representation.

use crate::model::{AssetId, SlotId};

/// A single slot inside a collection pointing to an asset.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Slot {
    pub id: SlotId,
    pub asset: AssetId,
}

impl Slot {
    /// Create a new slot.
    pub fn new(id: impl Into<SlotId>, asset: impl Into<AssetId>) -> Self {
        Self {
            id: id.into(),
            asset: asset.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_serde_roundtrip() {
        let slot = Slot::new("slot-1", "asset-1");
        let json = serde_json::to_string(&slot).expect("serialize");
        let back: Slot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(slot, back);
    }
}
