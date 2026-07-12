use serde::{Deserialize, Serialize};

use crate::{CorrelationId, RequestId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CancelFrame {
    pub request_id: RequestId,
    pub correlation_id: CorrelationId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CancelAckFrame {
    pub request_id: RequestId,
    pub correlation_id: CorrelationId,
    pub accepted: bool,
    pub already_terminal: bool,
}
