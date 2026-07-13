use serde::{Deserialize, Serialize};

use crate::{CorrelationId, ProtocolVersion, RequestId, WorkerIncarnationId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CancelFrame {
    pub protocol_version: ProtocolVersion,
    pub incarnation_id: WorkerIncarnationId,
    pub request_id: RequestId,
    pub correlation_id: CorrelationId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CancelAckFrame {
    pub protocol_version: ProtocolVersion,
    pub incarnation_id: WorkerIncarnationId,
    pub request_id: RequestId,
    pub correlation_id: CorrelationId,
    pub accepted: bool,
    pub already_terminal: bool,
}
