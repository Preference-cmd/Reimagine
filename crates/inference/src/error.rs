//! Inference-to-executor error bridge.
//!
//! The canonical [`InferenceError`](reimagine_inference_core::InferenceError)
//! lives in `reimagine-inference-core`, which cannot depend on
//! `reimagine-inference` (the architecture forbids that edge). This
//! module owns the explicit, call-site-visible conversion from the
//! inference-core error to the executor
//! [`NodeExecutorError`](crate::executor::NodeExecutorError).
//!
//! Two equivalent forms are provided:
//!
//! - [`IntoNodeExecutorError`] trait — for callers that prefer the
//!   method form: `err.into_executor_error()`.
//! - [`into_executor_error`] free function — preferred at executor
//!   call sites; it stays explicit even if the trait method is
//!   shadowed by another implementation.

use crate::executor::NodeExecutorError;
use reimagine_inference_core::InferenceError;

/// Trait that maps `inference_core::InferenceError` to the executor
/// `NodeExecutorError` at the inference boundary.
pub trait IntoNodeExecutorError {
    fn into_executor_error(self) -> NodeExecutorError;
}

impl IntoNodeExecutorError for InferenceError {
    fn into_executor_error(self) -> NodeExecutorError {
        match self {
            InferenceError::BackendNotImplemented {
                capability,
                backend_kind,
                message,
            } => {
                let base = format!(
                    "backend `{backend_kind}` does not implement capability `{capability}`"
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
            InferenceError::BackendCapabilityUnsupported { kind, capability } => {
                NodeExecutorError::Failed {
                    message: format!(
                        "backend `{kind}` does not advertise capability for `{capability}`"
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
                capability,
            } => NodeExecutorError::Failed {
                message: format!(
                    "capability `{capability}` would require a cross-backend bridge from `{source}` to `{target}`"
                ),
            },
            InferenceError::BackendBridgeUnsupported {
                source,
                target,
                capability,
                reason,
            } => NodeExecutorError::Failed {
                message: format!(
                    "bridge policy forbids transfer from `{source}` to `{target}` for capability `{capability}`: {reason}"
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
    use reimagine_inference_core::InferenceCapability;

    #[test]
    fn backend_not_implemented_converts_to_failed() {
        let err = InferenceError::BackendNotImplemented {
            capability: InferenceCapability::DiffusionSample,
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
            capability: InferenceCapability::DiffusionSample,
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
            capability: InferenceCapability::DiffusionSample,
        };
        let exec_err = into_executor_error(err);
        let msg = exec_err.to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("diffusion.sample"), "{msg}");
    }
}
