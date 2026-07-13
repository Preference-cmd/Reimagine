use serde::{Deserialize, Serialize};

use crate::{CorrelationId, ProtocolVersion, RequestId, WorkerIncarnationId};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProgressFrame {
    pub protocol_version: ProtocolVersion,
    pub incarnation_id: WorkerIncarnationId,
    pub request_id: RequestId,
    pub correlation_id: CorrelationId,
    pub sequence: u64,
    pub completed: u64,
    pub total: Option<u64>,
    pub message: Option<String>,
}
