use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ProtocolVersion, WorkerIncarnationId};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub String);

impl From<&str> for RequestId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CorrelationId(pub String);

impl From<&str> for CorrelationId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestFrame {
    pub protocol_version: ProtocolVersion,
    pub incarnation_id: WorkerIncarnationId,
    pub request_id: RequestId,
    pub correlation_id: CorrelationId,
    pub operation: String,
    pub payload: Value,
}
