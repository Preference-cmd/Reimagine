use reimagine_core::board::{
    BoardChange, BoardCommandResult, BoardCommandResultStatus, BoardDocument, BoardItem,
};
use serde::{Deserialize, Serialize};

use super::runs::DiagnosticDto;

/// Stable project board snapshot for IPC/UI consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoardSnapshotDto {
    pub id: String,
    pub project_id: String,
    pub version: u64,
    pub items: Vec<BoardItem>,
}

impl From<BoardDocument> for BoardSnapshotDto {
    fn from(board: BoardDocument) -> Self {
        Self {
            id: board.id().to_string(),
            project_id: board.project_id().to_string(),
            version: board.version().get(),
            items: board.items().values().cloned().collect(),
        }
    }
}

/// Stable result projection for board command operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoardCommandResultDto {
    pub status: String,
    pub board_version: u64,
    pub changes: Vec<BoardChange>,
    pub diagnostics: Vec<DiagnosticDto>,
    pub history_entry_id: Option<String>,
}

impl From<BoardCommandResult> for BoardCommandResultDto {
    fn from(result: BoardCommandResult) -> Self {
        Self {
            status: match result.status() {
                BoardCommandResultStatus::Applied => "applied",
                BoardCommandResultStatus::Rejected => "rejected",
                BoardCommandResultStatus::NoOp => "no_op",
            }
            .to_owned(),
            board_version: result.board_version().get(),
            changes: result.changes().to_vec(),
            diagnostics: result
                .diagnostics()
                .iter()
                .cloned()
                .map(DiagnosticDto::from)
                .collect(),
            history_entry_id: result.history_entry_id().map(ToString::to_string),
        }
    }
}
