//! The terminal result of an executed operation.
//!
//! Backend errors are a result kind, not a transport failure: they are
//! the final event of a successful `Execute` stream. gRPC `Status` is
//! reserved for transport-level problems.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::BackendExecutionError;

/// How an executed operation ended. Exactly one terminal outcome is
/// reported per execution, chosen by the host when cancel and completion
/// race.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TerminalOutcome {
    Success { output: Value },
    Cancelled,
    BackendError { error: BackendExecutionError },
}
