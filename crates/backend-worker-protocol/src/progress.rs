use serde::{Deserialize, Serialize};

use crate::{CorrelationId, RequestId};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProgressFrame {
    pub request_id: RequestId,
    pub correlation_id: CorrelationId,
    pub sequence: u64,
    pub completed: u64,
    pub total: Option<u64>,
    pub message: Option<String>,
}
