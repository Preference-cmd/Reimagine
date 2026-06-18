//! Bridge policy and bridge trait shapes for cross-backend value transfer.

use crate::request::InferenceOperationId;
use reimagine_core::BackendKind;

/// How a [`BackendBridgePolicy`] classifies a potential value
/// transfer between two backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeSupport {
    /// The value can be used directly in the target backend.
    Direct,
    /// A registered bridge can transfer the value at the documented
    /// cost.
    Bridgeable { cost: String },
    /// No available bridge can perform the transfer.
    Unsupported { reason: String },
}

/// Concrete plan returned by a [`BackendBridgePolicy`].
///
/// V1 routers either pass the value through unchanged (`Direct`)
/// or refuse the transfer (`Unsupported`). `Bridgeable` is reserved
/// for future bridge implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgePlan {
    Direct,
    Bridgeable { bridge_kind: String, cost: String },
    Unsupported { reason: String },
}

/// Bridge policy decides whether a value owned by `source_backend`
/// may be used inside a request targeting `target_backend`.
pub trait BackendBridgePolicy: Send + Sync + 'static {
    fn plan_transfer(
        &self,
        source_backend: &BackendKind,
        target_backend: &BackendKind,
        operation_id: &InferenceOperationId,
    ) -> BridgePlan;
}

/// A concrete bridge implementation that can transform a value
/// across backends.
///
/// V1 ships no bridge implementations; the trait exists so future
/// `BackendBridgePolicy` implementations can locate a bridge by name.
pub trait BackendBridge: Send + Sync + 'static {
    fn name(&self) -> &str;

    fn can_transfer(
        &self,
        source: &BackendKind,
        target: &BackendKind,
        operation_id: &InferenceOperationId,
    ) -> BridgeSupport;
}

/// Default V1 bridge policy: refuse all cross-backend transfers.
///
/// The architecture's rule is "fail explicitly rather than silently
/// reinterpret backend payload keys", and the runtime must not
/// perform implicit cross-backend tensor conversion. The reject-all
/// default enforces both. Future per-backend bridge registrations
/// can introduce `Bridgeable` plans without changing the V1 router
/// contract.
#[derive(Debug, Default, Clone, Copy)]
pub struct RejectAllBridgePolicy;

impl BackendBridgePolicy for RejectAllBridgePolicy {
    fn plan_transfer(
        &self,
        source_backend: &BackendKind,
        target_backend: &BackendKind,
        operation_id: &InferenceOperationId,
    ) -> BridgePlan {
        if source_backend == target_backend {
            BridgePlan::Direct
        } else {
            BridgePlan::Unsupported {
                reason: format!(
                    "bridge policy rejects cross-backend transfer from `{source_backend}` to `{target_backend}` for operation `{operation_id}`"
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{OP_DIFFUSION_SAMPLE, OP_LATENT_CREATE_EMPTY};

    #[test]
    fn same_backend_returns_direct() {
        let p = RejectAllBridgePolicy;
        let plan = p.plan_transfer(
            &BackendKind::new("candle"),
            &BackendKind::new("candle"),
            &OP_LATENT_CREATE_EMPTY.into(),
        );
        assert!(matches!(plan, BridgePlan::Direct));
    }

    #[test]
    fn cross_backend_returns_unsupported_with_reason() {
        let p = RejectAllBridgePolicy;
        let plan = p.plan_transfer(
            &BackendKind::new("candle"),
            &BackendKind::new("remote"),
            &OP_DIFFUSION_SAMPLE.into(),
        );
        match plan {
            BridgePlan::Unsupported { reason } => {
                assert!(reason.contains("candle"), "{reason}");
                assert!(reason.contains("remote"), "{reason}");
                assert!(reason.contains("diffusion.sample"), "{reason}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
