use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendExecutionError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}
