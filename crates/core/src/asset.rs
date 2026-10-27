use serde::{Deserialize, Serialize};

use crate::model::{AssetId, StepId};

/// A stored project asset.
///
/// Assets are immutable content-addressed blobs whose origin is always
/// a recorded step. The asset id is distinct from the content hash: identical
/// content produced by different steps yields distinct asset records that
/// share one stored file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRecord {
    pub id: AssetId,
    /// Lowercase hexadecimal SHA-256 hash of the asset bytes.
    pub hash: String,
    pub media_type: String,
    pub size: u64,
    pub origin: StepId,
}

impl AssetRecord {
    pub fn new(
        id: AssetId,
        hash: impl Into<String>,
        media_type: impl Into<String>,
        size: u64,
        origin: StepId,
    ) -> Self {
        Self {
            id,
            hash: hash.into(),
            media_type: media_type.into(),
            size,
            origin,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_record_serde_roundtrip() {
        let record = AssetRecord::new(
            AssetId::new("asset-1"),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "text/plain",
            42,
            StepId::new("step-1"),
        );
        let serialized = serde_json::to_string(&record).expect("serialize");
        let deserialized: AssetRecord = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(record, deserialized);
    }
}
