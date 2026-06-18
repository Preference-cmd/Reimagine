//! Inference-to-runtime error bridge.
//!
//! The canonical [`InferenceError`](reimagine_inference_core::InferenceError)
//! lives in `reimagine-inference-core`, which cannot depend on
//! `reimagine-runtime` (the architecture explicitly forbids that
//! edge). This module owns the explicit, call-site-visible
//! conversion from the inference-core error to the runtime
//! [`NodeExecutorError`].
//!
//! Two equivalent forms are provided:
//!
//! - [`IntoNodeExecutorError`] trait — for callers that prefer the
//!   method form: `err.into_executor_error()`.
//! - [`into_executor_error`] free function — preferred at executor
//!   call sites; it stays explicit even if the trait method is
//!   shadowed by another implementation.

use reimagine_inference_core::InferenceError;
use reimagine_runtime::NodeExecutorError;

/// Trait that maps `inference_core::InferenceError` to
/// `runtime::NodeExecutorError` at the inference-to-runtime boundary.
pub trait IntoNodeExecutorError {
    fn into_executor_error(self) -> NodeExecutorError;
}

impl IntoNodeExecutorError for InferenceError {
    fn into_executor_error(self) -> NodeExecutorError {
        match self {
            InferenceError::BackendNotImplemented {
                operation_id,
                backend_kind,
                message,
            } => {
                let base = format!(
                    "backend `{backend_kind}` does not implement operation `{operation_id}`"
                );
                NodeExecutorError::Failed {
                    message: match message {
                        Some(m) => format!("{base}: {m}"),
                        None => base,
                    },
                }
            }
            InferenceError::InvalidResponse { reason } => NodeExecutorError::Failed {
                message: format!("invalid backend response: {reason}"),
            },
            InferenceError::MissingInput { slot_id } => NodeExecutorError::MissingInput {
                slot_id: slot_id.to_string(),
            },
            InferenceError::BackendExecutionFailed { message } => {
                NodeExecutorError::Failed { message }
            }
            InferenceError::ModelResolutionFailed { message } => NodeExecutorError::Failed {
                message: format!("model resolution failed: {message}"),
            },
            InferenceError::BackendNotRegistered { kind } => NodeExecutorError::Failed {
                message: format!(
                    "backend `{kind}` is not registered in the inference-core registry"
                ),
            },
            InferenceError::BackendCapabilityUnsupported { kind, operation_id } => {
                NodeExecutorError::Failed {
                    message: format!(
                        "backend `{kind}` does not advertise capability for `{operation_id}`"
                    ),
                }
            }
            InferenceError::IncompatibleHandleAffinity { expected, actual } => {
                NodeExecutorError::Failed {
                    message: format!(
                        "incompatible handle affinity: expected `{expected}`, got `{actual}`"
                    ),
                }
            }
            InferenceError::BackendBridgeRequired {
                source,
                target,
                operation_id,
            } => NodeExecutorError::Failed {
                message: format!(
                    "operation `{operation_id}` would require a cross-backend bridge from `{source}` to `{target}`"
                ),
            },
            InferenceError::BackendBridgeUnsupported {
                source,
                target,
                operation_id,
                reason,
            } => NodeExecutorError::Failed {
                message: format!(
                    "bridge policy forbids transfer from `{source}` to `{target}` for operation `{operation_id}`: {reason}"
                ),
            },
        }
    }
}

/// Free-function form preferred by executor call sites.
pub fn into_executor_error(err: InferenceError) -> NodeExecutorError {
    IntoNodeExecutorError::into_executor_error(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::SlotId;

    #[test]
    fn backend_not_implemented_converts_to_failed() {
        let err = InferenceError::BackendNotImplemented {
            operation_id: "diffusion.sample".to_string(),
            backend_kind: "fake".to_string(),
            message: None,
        };
        let exec_err = into_executor_error(err);
        assert!(matches!(exec_err, NodeExecutorError::Failed { .. }));
        assert!(exec_err.to_string().contains("diffusion.sample"));
    }

    #[test]
    fn missing_input_converts_to_missing_input() {
        let err = InferenceError::MissingInput {
            slot_id: SlotId::new("text"),
        };
        let exec_err = into_executor_error(err);
        assert!(matches!(exec_err, NodeExecutorError::MissingInput { .. }));
    }

    #[test]
    fn invalid_response_converts_to_failed() {
        let err = InferenceError::InvalidResponse {
            reason: "duplicate slot".to_string(),
        };
        let exec_err = into_executor_error(err);
        assert!(matches!(exec_err, NodeExecutorError::Failed { .. }));
        assert!(exec_err.to_string().contains("duplicate slot"));
    }

    #[test]
    fn backend_bridge_unsupported_converts_to_failed_with_reason() {
        let err = InferenceError::BackendBridgeUnsupported {
            source: "candle".to_string(),
            target: "remote".to_string(),
            operation_id: "diffusion.sample".to_string(),
            reason: "no bridge registered".to_string(),
        };
        let exec_err = into_executor_error(err);
        let msg = exec_err.to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("remote"), "{msg}");
        assert!(msg.contains("no bridge registered"), "{msg}");
    }

    #[test]
    fn backend_capability_unsupported_converts_to_failed() {
        let err = InferenceError::BackendCapabilityUnsupported {
            kind: "candle".to_string(),
            operation_id: "diffusion.sample".to_string(),
        };
        let exec_err = into_executor_error(err);
        let msg = exec_err.to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("diffusion.sample"), "{msg}");
    }
}
