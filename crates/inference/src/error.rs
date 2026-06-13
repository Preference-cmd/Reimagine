//! Backend-neutral inference errors.
//!
//! [`InferenceError`] is the canonical error type returned by
//! [`InferenceBackend::execute`](crate::InferenceBackend::execute).
//! It converts into [`reimagine_runtime::NodeExecutorError`] through
//! an explicit [`into_executor_error`](InferenceError::into_executor_error)
//! method so the inference-to-runtime boundary remains visible at
//! every call site.

use reimagine_core::model::SlotId;

/// Errors produced by the inference layer.
#[derive(Debug)]
pub enum InferenceError {
    /// The backend does not implement the requested operation for the
    /// given model series/variant combination. The executor should
    /// surface this as a deterministic, non-retryable node failure.
    BackendNotImplemented {
        operation_id: String,
        backend_kind: String,
    },
    /// The backend returned a response that fails the executor's
    /// output validation. The executor should surface this as a
    /// deterministic node failure.
    InvalidResponse { reason: String },
    /// A required input was missing from the request.
    MissingInput { slot_id: SlotId },
    /// The backend encountered an internal execution failure.
    BackendExecutionFailed { message: String },
    /// The model resolver could not resolve the requested model
    /// reference.
    ModelResolutionFailed { message: String },
}

impl InferenceError {
    /// Map this error into a [`reimagine_runtime::NodeExecutorError`].
    ///
    /// V1 uses an explicit method rather than a broad `From`
    /// implementation so every call site makes the inference-to-runtime
    /// boundary visible.
    pub fn into_executor_error(self) -> reimagine_runtime::NodeExecutorError {
        match self {
            Self::BackendNotImplemented {
                operation_id,
                backend_kind,
            } => reimagine_runtime::NodeExecutorError::Failed {
                message: format!(
                    "backend `{backend_kind}` does not implement operation `{operation_id}`"
                ),
            },
            Self::InvalidResponse { reason } => reimagine_runtime::NodeExecutorError::Failed {
                message: format!("invalid backend response: {reason}"),
            },
            Self::MissingInput { slot_id } => reimagine_runtime::NodeExecutorError::MissingInput {
                slot_id: slot_id.to_string(),
            },
            Self::BackendExecutionFailed { message } => {
                reimagine_runtime::NodeExecutorError::Failed { message }
            }
            Self::ModelResolutionFailed { message } => {
                reimagine_runtime::NodeExecutorError::Failed {
                    message: format!("model resolution failed: {message}"),
                }
            }
        }
    }
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotImplemented {
                operation_id,
                backend_kind,
            } => {
                write!(
                    f,
                    "backend `{backend_kind}` does not implement `{operation_id}`"
                )
            }
            Self::InvalidResponse { reason } => write!(f, "invalid response: {reason}"),
            Self::MissingInput { slot_id } => write!(f, "missing input slot `{slot_id}`"),
            Self::BackendExecutionFailed { message } => write!(f, "backend error: {message}"),
            Self::ModelResolutionFailed { message } => write!(f, "model resolution: {message}"),
        }
    }
}

impl std::error::Error for InferenceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_not_implemented_converts_to_failed() {
        let err = InferenceError::BackendNotImplemented {
            operation_id: "diffusion.sample".to_string(),
            backend_kind: "fake".to_string(),
        };
        let exec_err = err.into_executor_error();
        assert!(matches!(
            exec_err,
            reimagine_runtime::NodeExecutorError::Failed { .. }
        ));
        assert!(exec_err.to_string().contains("diffusion.sample"));
    }

    #[test]
    fn missing_input_converts_to_missing_input() {
        let err = InferenceError::MissingInput {
            slot_id: SlotId::new("text"),
        };
        let exec_err = err.into_executor_error();
        assert!(matches!(
            exec_err,
            reimagine_runtime::NodeExecutorError::MissingInput { .. }
        ));
    }

    #[test]
    fn invalid_response_converts_to_failed() {
        let err = InferenceError::InvalidResponse {
            reason: "duplicate slot".to_string(),
        };
        let exec_err = err.into_executor_error();
        assert!(matches!(
            exec_err,
            reimagine_runtime::NodeExecutorError::Failed { .. }
        ));
        assert!(exec_err.to_string().contains("duplicate slot"));
    }
}
