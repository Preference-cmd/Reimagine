//! Health-check DTOs.

use reimagine_core::diagnostic::DiagnosticDto;
use reimagine_core::health::Health;
use serde::{Deserialize, Serialize};

use crate::WorkspaceHost;

/// V1 `GET /health` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub workspace: String,
    pub kernel: HealthDto,
    pub aigc: HealthDto,
}

impl HealthResponse {
    pub fn from_workspace(host: &WorkspaceHost) -> Self {
        Self {
            workspace: host.workspace_scope().as_str().to_string(),
            kernel: HealthDto::from(&Health::Available),
            aigc: HealthDto::from(host.aigc_health()),
        }
    }
}

/// Wire representation of subsystem health.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum HealthDto {
    Available,
    Unavailable { diagnostic: DiagnosticDto },
}

impl From<&Health> for HealthDto {
    fn from(health: &Health) -> Self {
        match health {
            Health::Available => Self::Available,
            Health::Unavailable(diagnostic) => Self::Unavailable {
                diagnostic: DiagnosticDto::from(diagnostic),
            },
        }
    }
}
