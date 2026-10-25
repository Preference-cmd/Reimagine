//! Progress events streamed while an operation is executing.

use serde::{Deserialize, Serialize};

/// One progress update of a running operation.
///
/// Progress is advisory: workers are free to send none, and the host
/// forwards each frame to the run's progress sink unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProgressFrame {
    pub sequence: u64,
    pub completed: u64,
    pub total: Option<u64>,
    pub message: Option<String>,
}
