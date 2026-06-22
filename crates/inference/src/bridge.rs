//! Bridge policy and bridge trait shapes for cross-backend value transfer.

use crate::Backend;

use crate::capability::InferenceCapability;

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
/// V1 routers either pass the value through unchanged (`Direct`) or
/// refuse the transfer (`Unsupported`). `Bridgeable` is reserved for
/// future bridge implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgePlan {
    Direct,
    Bridgeable { bridge_kind: String, cost: String },
    Unsupported { reason: String },
}

/// Bridge policy decides whether a value owned by `source_backend`
/// may be used inside a request targeting `target_backend` for the
/// given capability.
///
/// `capability` is a diagnostic / context label, not a dispatch key.
/// It is the caller's responsibility to pass the correct capability
/// for the call being planned.
pub trait BackendBridgePolicy: Send + Sync + 'static {
    fn plan_transfer(
        &self,
        source_backend: &Backend,
        target_backend: &Backend,
        capability: InferenceCapability,
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
        source: &Backend,
        target: &Backend,
        capability: InferenceCapability,
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
        source_backend: &Backend,
        target_backend: &Backend,
        capability: InferenceCapability,
    ) -> BridgePlan {
        if source_backend == target_backend {
            BridgePlan::Direct
        } else {
            BridgePlan::Unsupported {
                reason: format!(
                    "bridge policy rejects cross-backend transfer from `{source_backend}` to `{target_backend}` for capability `{capability}`"
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_backend_returns_direct() {
        let p = RejectAllBridgePolicy;
        let plan = p.plan_transfer(
            &Backend::new("candle"),
            &Backend::new("candle"),
            InferenceCapability::CreateEmptyLatent,
        );
        assert!(matches!(plan, BridgePlan::Direct));
    }

    #[test]
    fn cross_backend_returns_unsupported_with_reason() {
        let p = RejectAllBridgePolicy;
        let plan = p.plan_transfer(
            &Backend::new("candle"),
            &Backend::new("remote"),
            InferenceCapability::DiffusionSample,
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
