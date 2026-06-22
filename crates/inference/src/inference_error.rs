//! Backend-neutral inference errors.
//!
//! [`InferenceError`] is the canonical error type returned by the
//! typed capability methods on
//! [`crate::backend::InferenceBackend`] and
//! [`crate::router::InferenceRuntime`].
//!
//! The mapping from [`InferenceError`] to
//! [`crate::executor::NodeExecutorError`] is intentionally kept in
//! [`crate::error`], next to the node executor contract. This module
//! stays focused on the backend/router error domain.
//!
//! Error variants that previously carried `operation_id: String` now
//! carry the structured [`crate::capability::InferenceCapability`]
//! so the error message and diagnostic label stay in lockstep with
//! the capability report.

use reimagine_core::model::SlotId;

use crate::Backend;
use crate::BackendInstance;
use crate::capability::InferenceCapability;

/// Errors produced by the inference layer.
#[derive(Debug)]
pub enum InferenceError {
    /// The backend does not implement the requested capability for the
    /// given model series/variant combination. The executor should
    /// surface this as a deterministic, non-retryable node failure.
    BackendNotImplemented {
        capability: InferenceCapability,
        backend_kind: String,
        message: Option<String>,
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
    /// The router tried to look up a backend for the request but the
    /// registry had nothing registered. Routers and executors should
    /// surface this as a deterministic configuration failure.
    BackendNotRegistered { kind: String },
    /// The selected backend does not advertise support for the
    /// capability the request asked for. Routers must surface this as
    /// a deterministic node failure rather than silently dispatching to
    /// the backend.
    BackendCapabilityUnsupported {
        kind: String,
        capability: InferenceCapability,
    },
    /// A value carried by the request is owned by a different backend
    /// than the one the request is targeting. The router should refuse
    /// the request unless an explicit bridge transfers it.
    IncompatibleHandleAffinity { expected: String, actual: String },
    /// The request would require a cross-backend transfer through a
    /// bridge. No bridge was attempted, but the request is structurally
    /// not legal without one.
    BackendBridgeRequired {
        source: String,
        target: String,
        capability: InferenceCapability,
    },
    /// The bridge policy explicitly refused a cross-backend transfer
    /// for the request. Carries the structured reason for diagnostic
    /// surfaces.
    BackendBridgeUnsupported {
        source: String,
        target: String,
        capability: InferenceCapability,
        reason: String,
    },
    /// The router had no viable candidate for the request. The
    /// policy produced no candidate, or every candidate was filtered
    /// by registration, allowed/disabled, or capability checks.
    BackendSelectionNoCandidate {
        capability: InferenceCapability,
        requested: Option<BackendInstance>,
        registered: usize,
    },
    /// The explicit override or policy candidate selected a backend
    /// instance that is not registered.
    CandidateBackendNotRegistered {
        instance: BackendInstance,
        capability: InferenceCapability,
    },
    /// The explicit override pinned a backend instance whose
    /// registered backend does not advertise the requested
    /// capability.
    CandidateBackendLacksCapability {
        instance: BackendInstance,
        backend: Backend,
        capability: InferenceCapability,
    },
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotImplemented {
                capability,
                backend_kind,
                message,
            } => {
                write!(
                    f,
                    "backend `{backend_kind}` does not implement capability `{capability}`"
                )?;
                if let Some(m) = message {
                    write!(f, ": {m}")?;
                }
                Ok(())
            }
            Self::InvalidResponse { reason } => write!(f, "invalid response: {reason}"),
            Self::MissingInput { slot_id } => write!(f, "missing input slot `{slot_id}`"),
            Self::BackendExecutionFailed { message } => write!(f, "backend error: {message}"),
            Self::ModelResolutionFailed { message } => write!(f, "model resolution: {message}"),
            Self::BackendNotRegistered { kind } => {
                write!(f, "no backend registered for kind `{kind}`")
            }
            Self::BackendCapabilityUnsupported { kind, capability } => write!(
                f,
                "backend `{kind}` does not advertise capability for `{capability}`"
            ),
            Self::IncompatibleHandleAffinity { expected, actual } => write!(
                f,
                "incompatible handle affinity: expected `{expected}`, got `{actual}`"
            ),
            Self::BackendBridgeRequired {
                source,
                target,
                capability,
            } => write!(
                f,
                "capability `{capability}` would require a cross-backend bridge from `{source}` to `{target}`"
            ),
            Self::BackendBridgeUnsupported {
                source,
                target,
                capability,
                reason,
            } => write!(
                f,
                "bridge policy forbids transfer from `{source}` to `{target}` for capability `{capability}`: {reason}"
            ),
            Self::BackendSelectionNoCandidate {
                capability,
                requested,
                registered,
            } => {
                write!(
                    f,
                    "router produced no candidate for capability `{capability}` ({registered} registered)"
                )?;
                if let Some(instance) = requested {
                    write!(f, "; requested `{instance}`")?;
                }
                Ok(())
            }
            Self::CandidateBackendNotRegistered {
                instance,
                capability,
            } => write!(
                f,
                "candidate backend instance `{instance}` for capability `{capability}` is not registered"
            ),
            Self::CandidateBackendLacksCapability {
                instance,
                backend,
                capability,
            } => write!(
                f,
                "candidate backend instance `{instance}` (backend `{backend}`) does not advertise capability `{capability}`"
            ),
        }
    }
}

impl std::error::Error for InferenceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_not_implemented_display() {
        let err = InferenceError::BackendNotImplemented {
            capability: InferenceCapability::DiffusionSample,
            backend_kind: "fake".to_string(),
            message: Some("no kernel".to_string()),
        };
        let msg = err.to_string();
        assert!(msg.contains("diffusion.sample"), "{msg}");
        assert!(msg.contains("fake"), "{msg}");
        assert!(msg.contains("no kernel"), "{msg}");
    }

    #[test]
    fn missing_input_display() {
        let err = InferenceError::MissingInput {
            slot_id: SlotId::new("text"),
        };
        let msg = err.to_string();
        assert!(msg.contains("text"), "{msg}");
    }

    #[test]
    fn invalid_response_display() {
        let err = InferenceError::InvalidResponse {
            reason: "duplicate slot".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("duplicate slot"), "{msg}");
    }

    #[test]
    fn backend_not_registered_display() {
        let err = InferenceError::BackendNotRegistered {
            kind: "candle".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("candle"), "{msg}");
    }

    #[test]
    fn backend_capability_unsupported_display() {
        let err = InferenceError::BackendCapabilityUnsupported {
            kind: "candle".to_string(),
            capability: InferenceCapability::ImageSave,
        };
        let msg = err.to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("image.save"), "{msg}");
    }

    #[test]
    fn incompatible_handle_affinity_display() {
        let err = InferenceError::IncompatibleHandleAffinity {
            expected: "candle".to_string(),
            actual: "remote".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("remote"), "{msg}");
    }

    #[test]
    fn backend_bridge_required_display() {
        let err = InferenceError::BackendBridgeRequired {
            source: "candle".to_string(),
            target: "remote".to_string(),
            capability: InferenceCapability::DiffusionSample,
        };
        let msg = err.to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("remote"), "{msg}");
        assert!(msg.contains("diffusion.sample"), "{msg}");
    }

    #[test]
    fn backend_bridge_unsupported_display() {
        let err = InferenceError::BackendBridgeUnsupported {
            source: "candle".to_string(),
            target: "remote".to_string(),
            capability: InferenceCapability::DiffusionSample,
            reason: "no bridge registered".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("remote"), "{msg}");
        assert!(msg.contains("diffusion.sample"), "{msg}");
        assert!(msg.contains("no bridge registered"), "{msg}");
    }

    #[test]
    fn backend_selection_no_candidate_display() {
        let err = InferenceError::BackendSelectionNoCandidate {
            capability: InferenceCapability::TextEncode,
            requested: Some(crate::BackendInstance::new("candle:metal")),
            registered: 0,
        };
        let msg = err.to_string();
        assert!(msg.contains("text.encode"), "{msg}");
        assert!(msg.contains("candle:metal"), "{msg}");
        assert!(msg.contains("0 registered"), "{msg}");
    }

    #[test]
    fn candidate_backend_not_registered_display() {
        let err = InferenceError::CandidateBackendNotRegistered {
            instance: crate::BackendInstance::new("missing"),
            capability: InferenceCapability::ImageSave,
        };
        let msg = err.to_string();
        assert!(msg.contains("missing"), "{msg}");
        assert!(msg.contains("image.save"), "{msg}");
    }

    #[test]
    fn candidate_backend_lacks_capability_display() {
        let err = InferenceError::CandidateBackendLacksCapability {
            instance: crate::BackendInstance::new("candle:cpu"),
            backend: crate::Backend::new("candle"),
            capability: InferenceCapability::TextEncode,
        };
        let msg = err.to_string();
        assert!(msg.contains("candle:cpu"), "{msg}");
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("text.encode"), "{msg}");
    }
}
