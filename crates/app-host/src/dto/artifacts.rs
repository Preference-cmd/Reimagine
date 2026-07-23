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

/// Host‑neutral projection of resolved artifact access information.
///
/// Strips raw paths from the serialised form — the `path` field is included
/// so the UI can hand it back to Tauri for desktop open/reveal affordances,
/// but callers should treat it as an opaque string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactMetadataDto {
    pub id: String,
    pub node_id: String,
    pub media_type: String,
    pub filename: String,
    pub path: String,
}

impl From<crate::artifact_access::ArtifactAccess> for ArtifactMetadataDto {
    fn from(value: crate::artifact_access::ArtifactAccess) -> Self {
        let filename = value
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_owned();
        Self {
            id: value.artifact_id.to_string(),
            node_id: value.node_id.to_string(),
            media_type: value.media_type,
            filename,
            path: value.path.to_string_lossy().into_owned(),
        }
    }
}
