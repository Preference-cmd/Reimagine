use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    BackendExecutionError, CorrelationId, ProtocolVersion, RequestId, WorkerIncarnationId,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TerminalOutcome {
    Success { output: Value },
    Cancelled,
    BackendError { error: BackendExecutionError },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerminalFrame {
    pub protocol_version: ProtocolVersion,
    pub incarnation_id: WorkerIncarnationId,
    pub request_id: RequestId,
    pub correlation_id: CorrelationId,
    pub outcome: TerminalOutcome,
}
