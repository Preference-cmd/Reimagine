use serde::{Deserialize, Serialize};

use crate::CorrelationId;

/// Metadata describing a tensor available for cross-worker transfer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Data type of the tensor (e.g. "f16", "f32", "i8").
    pub dtype: String,
    /// Shape of the tensor (e.g. `[1, 3, 512, 512]`).
    pub shape: Vec<u64>,
    /// Total size in bytes of the tensor payload.
    pub size_bytes: u64,
    /// Backend-specific format hint (e.g. "burn::nchw", "ndarray").
    pub backend_format: String,
}

/// Request from one worker to pull tensor data from another.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorTransferRequestFrame {
    /// Token identifying the source tensor held by the originating worker.
    pub source_token: String,
    /// ID of the worker that holds the source tensor.
    pub target_worker_id: String,
    /// Metadata describing the tensor being requested.
    pub tensor_metadata: TensorMetadata,
}

/// Status of a tensor transfer operation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TransferStatus {
    /// Transfer accepted; data frames will follow.
    Accepted,
    /// Transfer rejected (e.g. token invalid, insufficient memory).
    Rejected { reason: String },
    /// All data frames have been delivered.
    Complete,
}

/// Acknowledgement frame for a tensor transfer request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorTransferAckFrame {
    /// Correlation ID matching the original request.
    pub correlation_id: CorrelationId,
    /// Outcome of the transfer request.
    pub status: TransferStatus,
    /// Optional token the target worker assigns for data frames to reference.
    pub target_token: Option<String>,
}

/// A chunk of tensor data in a transfer stream.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorDataFrame {
    /// Correlation ID matching the original transfer request.
    pub correlation_id: CorrelationId,
    /// Sequence number of this chunk (0-based).
    pub sequence: u64,
    /// Raw tensor bytes for this chunk.
    pub data: Vec<u8>,
    /// Whether this is the final chunk in the transfer.
    #[serde(rename = "final")]
    pub is_final: bool,
}
