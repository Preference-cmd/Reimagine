//! Local protocol types for agent turns.
//!
//! These types were previously in agent-daemon but are now defined locally
//! to remove the daemon dependency.

use serde::{Deserialize, Serialize};

/// Status reported by `turn.run`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRunStatus {
    Accepted,
}

/// `turn.run` response result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnRunResult {
    pub status: TurnRunStatus,
    pub session_id: String,
    pub turn_id: String,
}

/// `turn.run` request params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnRunParams {
    pub session_id: String,
    pub turn_id: String,
    pub model: String,
    pub input: serde_json::Value,
}
