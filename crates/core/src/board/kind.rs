//! Board item kind variants.

use crate::model::{ArtifactId, RunId, WorkflowId};

/// The kind of item placed on a board.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BoardItemKind {
    /// Reference to a workflow.
    WorkflowRef {
        /// The workflow id this item references.
        workflow_id: WorkflowId,
    },
    /// Reference to an asset (image, video, audio, etc.).
    AssetRef {
        /// The artifact id this item references.
        artifact_id: ArtifactId,
    },
    /// A textual note on the canvas.
    Note {
        /// The note text content.
        content: String,
    },
    /// Reference to a completed or running execution.
    RunRef {
        /// The run id this item references.
        run_id: RunId,
    },
}

impl BoardItemKind {
    /// Returns a human-readable label for the item kind.
    pub fn label(&self) -> &str {
        match self {
            BoardItemKind::WorkflowRef { .. } => "Workflow",
            BoardItemKind::AssetRef { .. } => "Asset",
            BoardItemKind::Note { .. } => "Note",
            BoardItemKind::RunRef { .. } => "Run",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_ref_roundtrip() {
        let kind = BoardItemKind::WorkflowRef {
            workflow_id: crate::model::WorkflowId::new("wf-1"),
        };
        let json = serde_json::to_string(&kind).expect("serialize");
        let back: BoardItemKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(kind, back);
    }

    #[test]
    fn asset_ref_roundtrip() {
        let kind = BoardItemKind::AssetRef {
            artifact_id: crate::model::ArtifactId::new("art/img-001"),
        };
        let json = serde_json::to_string(&kind).expect("serialize");
        let back: BoardItemKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(kind, back);
    }

    #[test]
    fn note_roundtrip() {
        let kind = BoardItemKind::Note {
            content: "Hello, world!".to_string(),
        };
        let json = serde_json::to_string(&kind).expect("serialize");
        let back: BoardItemKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(kind, back);
    }

    #[test]
    fn run_ref_roundtrip() {
        let kind = BoardItemKind::RunRef {
            run_id: crate::model::RunId::new("run-1"),
        };
        let json = serde_json::to_string(&kind).expect("serialize");
        let back: BoardItemKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(kind, back);
    }

    #[test]
    fn item_kind_label() {
        let wf = BoardItemKind::WorkflowRef {
            workflow_id: crate::model::WorkflowId::new("wf-1"),
        };
        assert_eq!(wf.label(), "Workflow");

        let asset = BoardItemKind::AssetRef {
            artifact_id: crate::model::ArtifactId::new("art/img-001"),
        };
        assert_eq!(asset.label(), "Asset");

        let note = BoardItemKind::Note {
            content: "text".to_string(),
        };
        assert_eq!(note.label(), "Note");

        let run = BoardItemKind::RunRef {
            run_id: crate::model::RunId::new("run-1"),
        };
        assert_eq!(run.label(), "Run");
    }
}
