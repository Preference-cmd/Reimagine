//! Artifact DTOs.

use reimagine_core::model::{ArtifactId, ArtifactRef, NodeId};
use reimagine_runtime::RunArtifactRef;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactDto {
    pub id: ArtifactId,
    pub node_id: NodeId,
    pub reference: ArtifactRef,
}

impl From<RunArtifactRef> for ArtifactDto {
    fn from(value: RunArtifactRef) -> Self {
        Self {
            id: value.id,
            node_id: value.node_id,
            reference: value.reference,
        }
    }
}
