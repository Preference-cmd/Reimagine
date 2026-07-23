//! Inference diagnostic helpers.
//!
//! Each helper returns a structured [`Diagnostic`] for the matching
//! [`InferenceError`](crate::inference_error::InferenceError) variant so the
//! router, app-host, and host adapters can surface the same shape
//! without a manual `format!` per call site.
//!
//! Diagnostic ids carry the capability's dot-separated label (e.g.
//! `inference-core-bridge-required-diffusion.sample-candle-remote`)
//! so existing diagnostic consumers see the same ids they did before
//! the typed-DTO refactor.
//!
//! The `inference-core` source/id/code prefix is intentionally retained
//! as a diagnostic compatibility label even though the physical
//! `reimagine-inference-core` crate has been folded into
//! `reimagine-inference`.

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;

use crate::capability::InferenceCapability;

fn source() -> DiagnosticSourceName {
    DiagnosticSourceName::new("inference-core")
}

fn domain(domain: &str, path: impl Into<String>) -> DiagnosticTarget {
    DiagnosticTarget::new(DiagnosticTargetDomain::new(domain)).with_path(path.into())
}

/// [`InferenceError::BackendBridgeRequired`](crate::inference_error::InferenceError::BackendBridgeRequired) diagnostic.
pub fn backend_bridge_required(
    source_backend: &str,
    target_backend: &str,
    value_kind: &str,
    capability: InferenceCapability,
) -> Diagnostic {
    let label = capability.as_str();
    let id = format!("inference-core-bridge-required-{label}-{source_backend}-{target_backend}");
    let target = domain(
        "inference.bridge",
        format!("{source_backend}->{target_backend}"),
    );
    Diagnostic::new(
        DiagnosticId::new(id),
        DiagnosticCode::new("INFERENCE_CORE/BRIDGE_REQUIRED"),
        DiagnosticSeverity::Error,
        source(),
        format!(
            "capability `{label}` requires cross-backend transfer from `{source_backend}` to `{target_backend}` for value kind `{value_kind}`"
        ),
        target,
    )
}

/// [`InferenceError::BackendBridgeUnsupported`](crate::inference_error::InferenceError::BackendBridgeUnsupported) diagnostic.
pub fn backend_bridge_unsupported(
    source_backend: &str,
    target_backend: &str,
    value_kind: &str,
    capability: InferenceCapability,
    reason: &str,
) -> Diagnostic {
    let label = capability.as_str();
    let id = format!("inference-core-bridge-unsupported-{label}-{source_backend}-{target_backend}");
    let target = domain(
        "inference.bridge",
        format!("{source_backend}->{target_backend}"),
    );
    Diagnostic::new(
        DiagnosticId::new(id),
        DiagnosticCode::new("INFERENCE_CORE/BRIDGE_UNSUPPORTED"),
        DiagnosticSeverity::Error,
        source(),
        format!(
            "capability `{label}` bridge from `{source_backend}` to `{target_backend}` for value kind `{value_kind}` is unsupported: {reason}"
        ),
        target,
    )
}

/// [`InferenceError::BackendNotRegistered`](crate::inference_error::InferenceError::BackendNotRegistered) diagnostic.
pub fn backend_not_registered(kind: &str) -> Diagnostic {
    let id = format!("inference-core-backend-not-registered-{kind}");
    let target = domain("inference.runtime", kind);
    Diagnostic::new(
        DiagnosticId::new(id),
        DiagnosticCode::new("INFERENCE_CORE/BACKEND_NOT_REGISTERED"),
        DiagnosticSeverity::Error,
        source(),
        format!("backend `{kind}` is not registered in the inference registry"),
        target,
    )
}

/// [`InferenceError::BackendCapabilityUnsupported`](crate::inference_error::InferenceError::BackendCapabilityUnsupported) diagnostic.
pub fn backend_capability_unsupported(kind: &str, capability: InferenceCapability) -> Diagnostic {
    let label = capability.as_str();
    let id = format!("inference-core-capability-unsupported-{kind}-{label}");
    let target = domain("inference.runtime", format!("{kind}/{label}"));
    Diagnostic::new(
        DiagnosticId::new(id),
        DiagnosticCode::new("INFERENCE_CORE/BACKEND_CAPABILITY_UNSUPPORTED"),
        DiagnosticSeverity::Error,
        source(),
        format!("backend `{kind}` does not advertise capability for `{label}`"),
        target,
    )
}

/// [`InferenceError::IncompatibleHandleAffinity`](crate::inference_error::InferenceError::IncompatibleHandleAffinity) diagnostic.
pub fn incompatible_handle_affinity(expected: &str, actual: &str) -> Diagnostic {
    let id = format!("inference-core-incompatible-handle-affinity-{expected}-{actual}");
    let target = domain("inference.runtime", format!("{expected}->{actual}"));
    Diagnostic::new(
        DiagnosticId::new(id),
        DiagnosticCode::new("INFERENCE_CORE/INCOMPATIBLE_HANDLE_AFFINITY"),
        DiagnosticSeverity::Error,
        source(),
        format!("incompatible handle affinity: expected `{expected}`, got `{actual}`"),
        target,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_required_diagnostic_carries_all_fields() {
        let d = backend_bridge_required(
            "candle",
            "remote",
            "latent",
            InferenceCapability::DiffusionSample,
        );
        assert_eq!(d.code().as_str(), "INFERENCE_CORE/BRIDGE_REQUIRED");
        let msg = d.message().to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("remote"), "{msg}");
        assert!(msg.contains("latent"), "{msg}");
        assert!(msg.contains("diffusion.sample"), "{msg}");
    }

    #[test]
    fn bridge_unsupported_diagnostic_carries_reason() {
        let d = backend_bridge_unsupported(
            "candle",
            "remote",
            "latent",
            InferenceCapability::DiffusionSample,
            "no bridge registered",
        );
        assert_eq!(d.code().as_str(), "INFERENCE_CORE/BRIDGE_UNSUPPORTED");
        assert!(d.message().to_string().contains("no bridge registered"));
    }

    #[test]
    fn backend_not_registered_diagnostic_uses_kind() {
        let d = backend_not_registered("candle");
        assert_eq!(d.code().as_str(), "INFERENCE_CORE/BACKEND_NOT_REGISTERED");
        assert!(d.message().to_string().contains("candle"));
    }

    #[test]
    fn capability_unsupported_diagnostic_uses_kind_and_capability() {
        let d = backend_capability_unsupported("candle", InferenceCapability::DiffusionSample);
        assert_eq!(
            d.code().as_str(),
            "INFERENCE_CORE/BACKEND_CAPABILITY_UNSUPPORTED"
        );
        let msg = d.message().to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("diffusion.sample"), "{msg}");
    }

    #[test]
    fn incompatible_handle_affinity_diagnostic_uses_both_sides() {
        let d = incompatible_handle_affinity("candle", "remote");
        assert_eq!(
            d.code().as_str(),
            "INFERENCE_CORE/INCOMPATIBLE_HANDLE_AFFINITY"
        );
        let msg = d.message().to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("remote"), "{msg}");
    }
}
