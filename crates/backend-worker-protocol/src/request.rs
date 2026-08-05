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

/// Known request operations carried by a [`RequestFrame`] `operation`.
///
/// New capabilities are added as request operations (an operation
/// string plus a JSON `payload` shape) rather than as new
/// [`WireMessage`](crate::WireMessage) variants: the generic
/// `Request { operation, payload }` shape is shared by every worker
/// transport (stdio, QUIC, gRPC) and requires no protocol-schema or
/// conversion changes.
pub mod request_operation {
    /// Apply resource-management hints (VRAM budget, prefetch intent,
    /// component lifecycle) for the upcoming stage.
    ///
    /// Payload is a JSON-serialized
    /// `reimagine_inference::ResourceHints`. The operation is
    /// advisory: backends that do not support it respond with an
    /// `unknown_operation` backend error and the caller must continue.
    pub const APPLY_RESOURCE_HINTS: &str = "resource.apply_hints";
}
