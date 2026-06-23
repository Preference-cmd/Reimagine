//! Health-check DTOs.

use serde::{Deserialize, Serialize};

/// V1 `GET /health` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub workspace: String,
}

impl HealthResponse {
    pub fn ok(workspace_id: &str) -> Self {
        Self {
            status: "ok",
            workspace: workspace_id.to_string(),
        }
    }
}
