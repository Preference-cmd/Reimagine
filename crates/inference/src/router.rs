//! Executor-facing inference router.
//!
//! [`InferenceRouter`] is the concrete struct that built-in executors
//! call to dispatch typed capability requests. It resolves the target
//! backend via a [`BackendSelectionPolicy`](crate::BackendSelectionPolicy),
//! consults the [`BackendBridgePolicy`](crate::BackendBridgePolicy)
//! for cross-backend handle conflicts, and delegates to the selected
//! [`InferenceBackend`](crate::InferenceBackend).

use std::sync::Arc;

use crate::Backend;
use crate::backend_registry::{InferenceBackendRegistry, RegisteredInstance};
use crate::backend_selection::{
    BackendInstance, BackendSelectionPolicy, BackendSelectionRequest, StaticBackendSelectionPolicy,
};
use crate::bridge::{BackendBridgePolicy, BridgePlan};
use crate::capability::InferenceCapability;
use crate::inference_error::InferenceError;
use crate::invocation::InferenceInvocation;
use crate::request::diffusion::DiffusionSampleRequest;
use crate::request::image::{ImagePreviewRequest, ImageSaveRequest};
use crate::request::image_import::ImageImportRequest;
use crate::request::latent::{CreateEmptyLatentRequest, LatentDecodeRequest};
use crate::request::latent_encode::LatentEncodeRequest;
use crate::request::model::LoadBundleRequest;
use crate::request::text::TextEncodeRequest;
use crate::response::diffusion::DiffusionSampleResponse;
use crate::response::image::{ImagePreviewResponse, ImageSaveResponse};
use crate::response::image_import::ImageImportResponse;
use crate::response::latent::{CreateEmptyLatentResponse, LatentDecodeResponse};
use crate::response::latent_encode::LatentEncodeResponse;
use crate::response::model::LoadBundleResponse;
use crate::response::text::TextEncodeResponse;
use crate::routing_request::RoutableInferenceRequest;

/// RouterRef is the type held by executors for lock-free atomic access
/// to the current [`InferenceRouter`]. When a worker hot-swaps, the
/// router is replaced atomically via [`arc_swap::ArcSwap::store`].
pub type RouterRef = Arc<arc_swap::ArcSwap<InferenceRouter>>;

/// Executor-facing inference router.
///
/// Applies a [`BackendSelectionPolicy`] to derive a candidate
/// [`BackendInstance`], consults the [`BackendBridgePolicy`] for
/// cross-backend handle conflicts, and dispatches the typed capability
/// call to the selected backend.
///
/// Selection precedence (deterministic, see `docs/architecture/modules/inference.md`):
///
/// 1. Existing backend-bound handle affinities. Conflicting
///    affinities ask the bridge policy; no silent fallback.
/// 2. Explicit override from the request overlay (when no
///    incompatible handles exist).
/// 3. Priority order from the policy.
/// 4. Diagnostic failure.
#[derive(Clone)]
pub struct InferenceRouter {
    registry: Arc<InferenceBackendRegistry>,
    selection_policy: Arc<dyn BackendSelectionPolicy>,
    bridge_policy: Arc<dyn BackendBridgePolicy>,
}

impl InferenceRouter {
    /// Construct a router with the [`StaticBackendSelectionPolicy`].
    pub fn new(
        registry: Arc<InferenceBackendRegistry>,
        bridge_policy: Arc<dyn BackendBridgePolicy>,
    ) -> Self {
        Self::with_policy(
            registry,
            Arc::new(StaticBackendSelectionPolicy::default()),
            bridge_policy,
        )
    }

    /// Construct a router with an explicit selection policy.
    pub fn with_policy(
        registry: Arc<InferenceBackendRegistry>,
        selection_policy: Arc<dyn BackendSelectionPolicy>,
        bridge_policy: Arc<dyn BackendBridgePolicy>,
    ) -> Self {
        Self {
            registry,
            selection_policy,
            bridge_policy,
        }
    }

    pub fn registry(&self) -> &Arc<InferenceBackendRegistry> {
        &self.registry
    }

    pub fn selection_policy(&self) -> &Arc<dyn BackendSelectionPolicy> {
        &self.selection_policy
    }

    pub fn bridge_policy(&self) -> &Arc<dyn BackendBridgePolicy> {
        &self.bridge_policy
    }

    /// Plan-level affinity validation.
    ///
    /// Returns `Some(target_backend)` when the affinities name a
    /// single backend (or are empty, in which case the router falls
    /// back to policy/override). Conflicting affinities ask the
    /// bridge policy to plan a transfer; if no plan is available,
    /// the router returns a bridge diagnostic and the call fails.
    fn plan_affinity(
        &self,
        affinities: &[BackendInstance],
        capability: InferenceCapability,
    ) -> Result<Option<BackendInstance>, InferenceError> {
        if affinities.is_empty() {
            return Ok(None);
        }
        let first = affinities[0].clone();
        let first_descriptor = self
            .registry
            .get(&first)
            .map(|registered| registered.descriptor);
        for other in &affinities[1..] {
            if other == &first {
                continue;
            }
            let other_descriptor = self
                .registry
                .get(other)
                .map(|registered| registered.descriptor);
            let Some(first_descriptor) = first_descriptor.as_ref() else {
                return Err(InferenceError::CandidateBackendNotRegistered {
                    instance: first,
                    capability,
                });
            };
            let Some(other_descriptor) = other_descriptor.as_ref() else {
                return Err(InferenceError::CandidateBackendNotRegistered {
                    instance: other.clone(),
                    capability,
                });
            };
            match self.bridge_policy.plan_transfer(
                &first_descriptor.backend,
                &other_descriptor.backend,
                capability,
            ) {
                BridgePlan::Direct
                | BridgePlan::Bridgeable { .. }
                | BridgePlan::CrossWorker { .. } => {}
                BridgePlan::Unsupported { reason } => {
                    return Err(InferenceError::BackendBridgeUnsupported {
                        source: other.to_string(),
                        target: first.to_string(),
                        capability,
                        reason,
                    });
                }
            }
        }
        Ok(Some(first))
    }

    /// Pick the first viable candidate and verify the backend
    /// advertises the requested capability.
    fn resolve_backend(
        &self,
        selection_request: &BackendSelectionRequest,
    ) -> Result<RegisteredInstance, InferenceError> {
        let capability = selection_request.capability;
        // 1. Affinity override.
        if let Some(plan_target) = self.plan_affinity(&selection_request.affinities, capability)? {
            if let Some(registered) = self.registry.get(&plan_target) {
                let caps = registered.backend.capabilities();
                if caps.supports_capability(capability) {
                    return Ok(registered);
                }
                return Err(InferenceError::CandidateBackendLacksCapability {
                    instance: plan_target,
                    backend: caps.backend_kind().clone(),
                    capability,
                });
            }
            return Err(InferenceError::CandidateBackendNotRegistered {
                instance: plan_target,
                capability,
            });
        }

        // 2. Explicit override (no conflicting affinities).
        if let Some(instance) = selection_request.explicit_override.clone() {
            if !self
                .selection_policy
                .allows_explicit_override(&instance, selection_request)
            {
                return Err(InferenceError::BackendSelectionNoCandidate {
                    capability,
                    requested: Some(instance),
                    registered: self.registry.len(),
                });
            }
            if let Some(registered) = self.registry.get(&instance) {
                let caps = registered.backend.capabilities();
                if caps.supports_capability(capability) {
                    return Ok(registered);
                }
                return Err(InferenceError::CandidateBackendLacksCapability {
                    instance,
                    backend: caps.backend_kind().clone(),
                    capability,
                });
            }
            return Err(InferenceError::CandidateBackendNotRegistered {
                instance,
                capability,
            });
        }

        // 3. Policy candidates.
        let candidates = self.selection_policy.candidates(selection_request);
        if candidates.is_empty() {
            return Err(InferenceError::BackendSelectionNoCandidate {
                capability,
                requested: None,
                registered: self.registry.len(),
            });
        }

        let mut first_missing: Option<(BackendInstance, Backend)> = None;
        for instance in candidates {
            let Some(registered) = self.registry.get(&instance) else {
                return Err(InferenceError::CandidateBackendNotRegistered {
                    instance,
                    capability,
                });
            };
            let caps = registered.backend.capabilities();
            if caps.supports_capability(capability) {
                return Ok(registered);
            }
            if first_missing.is_none() {
                first_missing = Some((instance, caps.backend_kind().clone()));
            }
        }

        if let Some((instance, backend)) = first_missing {
            return Err(InferenceError::CandidateBackendLacksCapability {
                instance,
                backend,
                capability,
            });
        }

        Err(InferenceError::BackendSelectionNoCandidate {
            capability,
            requested: None,
            registered: self.registry.len(),
        })
    }

    fn resolve_for_request<R: RoutableInferenceRequest>(
        &self,
        request: &R,
    ) -> Result<RegisteredInstance, InferenceError> {
        let selection = request.selection_request(self.registry.descriptors());
        self.resolve_backend(&selection)
    }

    // ─── Typed capability dispatch methods ─────────────────────

    pub async fn load_bundle_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .load_bundle_with_invocation(invocation, request)
            .await
    }

    pub async fn text_encode_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .text_encode_with_invocation(invocation, request)
            .await
    }

    pub async fn create_empty_latent_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .create_empty_latent_with_invocation(invocation, request)
            .await
    }

    pub async fn diffusion_sample_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .diffusion_sample_with_invocation(invocation, request)
            .await
    }

    pub async fn latent_decode_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .latent_decode_with_invocation(invocation, request)
            .await
    }

    pub async fn latent_encode_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .latent_encode_with_invocation(invocation, request)
            .await
    }

    pub async fn image_import_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .image_import_with_invocation(invocation, request)
            .await
    }

    pub async fn image_save_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .image_save_with_invocation(invocation, request)
            .await
    }

    pub async fn image_preview_with_invocation(
        &self,
        invocation: &InferenceInvocation,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        let backend = self.resolve_for_request(&request)?;
        backend
            .backend
            .image_preview_with_invocation(invocation, request)
            .await
    }
}

impl std::fmt::Debug for InferenceRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceRouter")
            .field("registry", &self.registry)
            .field("selection_policy", &"<dyn BackendSelectionPolicy>")
            .field("bridge_policy", &"<dyn BackendBridgePolicy>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LatentSpaceMetadata;
    use crate::RuntimeLatent;
    use crate::backend::InferenceBackend;
    use crate::backend_registry::InferenceBackendRegistry;
    use crate::backend_selection::BackendInstanceDescriptor;
    use crate::bridge::RejectAllBridgePolicy;
    use crate::capability::{
        InferenceBackendCapabilities, InferenceCapability, InferenceCapabilitySupport,
    };
    use crate::request::latent::CreateEmptyLatentRequest;
    use crate::request::latent_encode::LatentEncodeRequest;
    use crate::response::latent::CreateEmptyLatentResponse;
    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

    struct EchoBackend {
        kind: Backend,
        supports_text: bool,
    }

    impl EchoBackend {
        fn new(label: &str) -> Self {
            Self {
                kind: Backend::new(label),
                supports_text: false,
            }
        }
    }

    /// Test bridge policy that prices every cross-backend transfer as
    /// a cross-worker move (T16).
    #[derive(Debug, Clone, Copy)]
    struct CrossWorkerBridgePolicy;

    impl BackendBridgePolicy for CrossWorkerBridgePolicy {
        fn plan_transfer(
            &self,
            source_backend: &Backend,
            target_backend: &Backend,
            _capability: InferenceCapability,
        ) -> BridgePlan {
            if source_backend == target_backend {
                BridgePlan::Direct
            } else {
                BridgePlan::CrossWorker {
                    source_worker: format!("worker-{source_backend}"),
                    target_worker: format!("worker-{target_backend}"),
                    estimated_cost: "~2.0ms".to_owned(),
                }
            }
        }
    }

    #[async_trait::async_trait]
    impl InferenceBackend for EchoBackend {
        fn backend_kind(&self) -> &Backend {
            &self.kind
        }
        fn capabilities(&self) -> InferenceBackendCapabilities {
            let mut caps = InferenceBackendCapabilities::new(self.kind.clone()).with_support(
                InferenceCapabilitySupport::new(InferenceCapability::CreateEmptyLatent),
            );
            if self.supports_text {
                caps = caps.with_support(InferenceCapabilitySupport::new(
                    InferenceCapability::TextEncode,
                ));
            }
            caps
        }
        async fn load_bundle(
            &self,
            _request: LoadBundleRequest,
        ) -> Result<LoadBundleResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::LoadBundle,
                backend_kind: self.kind.to_string(),
                message: None,
            })
        }
        async fn text_encode(
            &self,
            _request: TextEncodeRequest,
        ) -> Result<TextEncodeResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::TextEncode,
                backend_kind: self.kind.to_string(),
                message: None,
            })
        }
        async fn create_empty_latent(
            &self,
            _request: CreateEmptyLatentRequest,
        ) -> Result<CreateEmptyLatentResponse, InferenceError> {
            Ok(CreateEmptyLatentResponse::new(RuntimeLatent::new(
                crate::BackendTensorHandle::new(
                    self.kind.clone(),
                    crate::BackendPayloadKey::new("empty"),
                    reimagine_core::model::TensorDType::F32,
                    reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                    "cpu",
                ),
                64,
                64,
                1,
                4,
                LatentSpaceMetadata::sdxl_base(),
                crate::LatentContent::EmptyGeometry,
            )))
        }
        async fn diffusion_sample(
            &self,
            _request: DiffusionSampleRequest,
        ) -> Result<DiffusionSampleResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::DiffusionSample,
                backend_kind: self.kind.to_string(),
                message: None,
            })
        }
        async fn latent_decode(
            &self,
            _request: LatentDecodeRequest,
        ) -> Result<LatentDecodeResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::LatentDecode,
                backend_kind: self.kind.to_string(),
                message: None,
            })
        }
        async fn latent_encode(
            &self,
            _request: LatentEncodeRequest,
        ) -> Result<LatentEncodeResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::LatentEncode,
                backend_kind: self.kind.to_string(),
                message: None,
            })
        }
        async fn image_import(
            &self,
            _request: ImageImportRequest,
        ) -> Result<ImageImportResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::ImageImport,
                backend_kind: self.kind.to_string(),
                message: None,
            })
        }
        async fn image_save(
            &self,
            _request: ImageSaveRequest,
        ) -> Result<ImageSaveResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::ImageSave,
                backend_kind: self.kind.to_string(),
                message: None,
            })
        }
        async fn image_preview(
            &self,
            _request: ImagePreviewRequest,
        ) -> Result<ImagePreviewResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::ImagePreview,
                backend_kind: self.kind.to_string(),
                message: None,
            })
        }
    }

    fn empty_latent_request() -> CreateEmptyLatentRequest {
        CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            RunId::new("r"),
            WorkflowId::new("w"),
            WorkflowVersion::new(1),
            NodeId::new("n"),
        )
    }

    fn make_registry_with_single_candle() -> Arc<InferenceBackendRegistry> {
        let mut reg = InferenceBackendRegistry::new();
        let descriptor = BackendInstanceDescriptor::new(
            BackendInstance::new("candle:cpu"),
            Backend::new("candle"),
        );
        reg.register(descriptor, Arc::new(EchoBackend::new("candle")));
        Arc::new(reg)
    }

    fn make_registry_with_two_candle_instances() -> Arc<InferenceBackendRegistry> {
        let mut reg = InferenceBackendRegistry::new();
        reg.register(
            BackendInstanceDescriptor::new(
                BackendInstance::new("candle:cpu"),
                Backend::new("candle"),
            ),
            Arc::new(EchoBackend::new("candle")),
        );
        reg.register(
            BackendInstanceDescriptor::new(
                BackendInstance::new("candle:metal"),
                Backend::new("candle"),
            ),
            Arc::new(EchoBackend::new("candle")),
        );
        Arc::new(reg)
    }

    fn noop_invocation() -> InferenceInvocation {
        InferenceInvocation::new(
            RunId::new("test-run"),
            NodeId::new("test-node"),
            None,
            Arc::new(crate::testing::NoopNodeCancellation::new()),
            Arc::new(crate::NoopInferenceProgressSink),
        )
    }

    #[tokio::test]
    async fn routes_through_first_registered_when_no_affinity() {
        let registry = make_registry_with_single_candle();
        let router = InferenceRouter::new(registry, Arc::new(RejectAllBridgePolicy));
        let invocation = noop_invocation();
        let response = router
            .create_empty_latent_with_invocation(&invocation, empty_latent_request())
            .await
            .unwrap();
        assert_eq!(response.latent().width(), 64);
    }

    #[tokio::test]
    async fn returns_no_candidate_when_registry_empty_and_no_policy() {
        let router = InferenceRouter::new(
            Arc::new(InferenceBackendRegistry::new()),
            Arc::new(RejectAllBridgePolicy),
        );
        let invocation = noop_invocation();
        let err = router
            .create_empty_latent_with_invocation(&invocation, empty_latent_request())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::BackendSelectionNoCandidate { .. }
        ));
    }

    #[tokio::test]
    async fn returns_capability_unsupported_when_no_candidate_supports_capability() {
        let registry = make_registry_with_single_candle();
        let policy = StaticBackendSelectionPolicy::new(vec![BackendInstance::new("candle:cpu")]);
        let router = InferenceRouter::with_policy(
            registry,
            Arc::new(policy),
            Arc::new(RejectAllBridgePolicy),
        );
        let clip = crate::RuntimeClipHandle::with_instance(
            reimagine_core::model::ModelId::new("clip"),
            Backend::new("candle"),
            BackendInstance::new("candle:cpu"),
            crate::BackendPayloadKey::new("k"),
        );
        let text = std::sync::Arc::new(crate::ExecutionValue::Param(
            reimagine_core::model::ParamValue::String("hello".to_string()),
        ));
        let req = TextEncodeRequest::new(
            clip,
            text,
            RunId::new("r"),
            WorkflowId::new("w"),
            WorkflowVersion::new(1),
            NodeId::new("n"),
        );
        let invocation = noop_invocation();
        let err = router
            .text_encode_with_invocation(&invocation, req)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::CandidateBackendLacksCapability { .. }
        ));
    }

    #[tokio::test]
    async fn explicit_override_pins_known_instance() {
        let registry = make_registry_with_single_candle();
        let router = InferenceRouter::new(registry, Arc::new(RejectAllBridgePolicy));
        let invocation = noop_invocation();
        let mut req = empty_latent_request();
        req.set_backend_selection_overlay(crate::BackendSelectionOverlay::with_explicit_override(
            BackendInstance::new("candle:cpu"),
        ));
        let response = router
            .create_empty_latent_with_invocation(&invocation, req)
            .await
            .unwrap();
        assert_eq!(response.latent().width(), 64);
    }

    #[tokio::test]
    async fn explicit_override_for_missing_instance_returns_not_registered() {
        let registry = make_registry_with_single_candle();
        let router = InferenceRouter::new(registry, Arc::new(RejectAllBridgePolicy));
        let invocation = noop_invocation();
        let mut req = empty_latent_request();
        req.set_backend_selection_overlay(crate::BackendSelectionOverlay::with_explicit_override(
            BackendInstance::new("missing"),
        ));
        let err = router
            .create_empty_latent_with_invocation(&invocation, req)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::CandidateBackendNotRegistered { .. }
        ));
    }

    #[tokio::test]
    async fn explicit_override_for_disabled_instance_is_rejected_by_policy() {
        let registry = make_registry_with_single_candle();
        let policy = StaticBackendSelectionPolicy::with_overrides(
            crate::BackendOverrides::new(),
            vec![BackendInstance::new("candle:cpu")],
            None,
            vec![BackendInstance::new("candle:cpu")],
        );
        let router = InferenceRouter::with_policy(
            registry,
            Arc::new(policy),
            Arc::new(RejectAllBridgePolicy),
        );
        let invocation = noop_invocation();
        let mut req = empty_latent_request();
        req.set_backend_selection_overlay(crate::BackendSelectionOverlay::with_explicit_override(
            BackendInstance::new("candle:cpu"),
        ));
        let err = router
            .create_empty_latent_with_invocation(&invocation, req)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::BackendSelectionNoCandidate { .. }
        ));
    }

    #[tokio::test]
    async fn handle_affinity_pins_concrete_backend_instance() {
        let registry = make_registry_with_two_candle_instances();
        let router = InferenceRouter::new(registry, Arc::new(RejectAllBridgePolicy));
        let clip = crate::RuntimeClipHandle::with_instance(
            reimagine_core::model::ModelId::new("clip"),
            Backend::new("candle"),
            BackendInstance::new("candle:metal"),
            crate::BackendPayloadKey::new("k"),
        );
        let text = std::sync::Arc::new(crate::ExecutionValue::Param(
            reimagine_core::model::ParamValue::String("hello".to_string()),
        ));
        let req = TextEncodeRequest::new(
            clip,
            text,
            RunId::new("r"),
            WorkflowId::new("w"),
            WorkflowVersion::new(1),
            NodeId::new("n"),
        );
        let invocation = noop_invocation();
        let err = router
            .text_encode_with_invocation(&invocation, req)
            .await
            .unwrap_err();
        match err {
            InferenceError::CandidateBackendLacksCapability { instance, .. } => {
                assert_eq!(instance, BackendInstance::new("candle:metal"));
            }
            other => panic!("expected CandidateBackendLacksCapability, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn affinity_for_unregistered_backend_returns_not_registered() {
        let mut reg = InferenceBackendRegistry::new();
        reg.register(
            BackendInstanceDescriptor::new(BackendInstance::new("a:cpu"), Backend::new("a")),
            Arc::new(EchoBackend::new("a")),
        );
        reg.register(
            BackendInstanceDescriptor::new(BackendInstance::new("b:cpu"), Backend::new("b")),
            Arc::new(EchoBackend::new("b")),
        );
        let router = InferenceRouter::new(Arc::new(reg), Arc::new(RejectAllBridgePolicy));

        let clip = crate::RuntimeClipHandle::with_instance(
            reimagine_core::model::ModelId::new("clip"),
            Backend::new("other"),
            BackendInstance::new("other:cpu"),
            crate::BackendPayloadKey::new("k"),
        );
        let text = std::sync::Arc::new(crate::ExecutionValue::Param(
            reimagine_core::model::ParamValue::String("hello".to_string()),
        ));
        let req = TextEncodeRequest::new(
            clip,
            text,
            RunId::new("r"),
            WorkflowId::new("w"),
            WorkflowVersion::new(1),
            NodeId::new("n"),
        );
        let invocation = noop_invocation();
        let err = router
            .text_encode_with_invocation(&invocation, req)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::CandidateBackendNotRegistered { .. }
        ));
    }

    #[tokio::test]
    async fn empty_registry_produces_no_candidate_with_registered_zero() {
        let router = InferenceRouter::new(
            Arc::new(InferenceBackendRegistry::new()),
            Arc::new(RejectAllBridgePolicy),
        );
        let invocation = noop_invocation();
        let err = router
            .create_empty_latent_with_invocation(&invocation, empty_latent_request())
            .await
            .unwrap_err();
        match err {
            InferenceError::BackendSelectionNoCandidate { registered, .. } => {
                assert_eq!(registered, 0);
            }
            other => panic!("expected BackendSelectionNoCandidate, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invocation_context_is_separate_from_request_and_routes_with_it() {
        let registry = make_registry_with_single_candle();
        let router = InferenceRouter::new(registry, Arc::new(RejectAllBridgePolicy));
        let invocation = crate::InferenceInvocation::new(
            RunId::new("invocation-run"),
            NodeId::new("invocation-node"),
            None,
            Arc::new(crate::testing::NoopNodeCancellation::new()),
            Arc::new(crate::NoopInferenceProgressSink),
        );

        let response = router
            .create_empty_latent_with_invocation(&invocation, empty_latent_request())
            .await
            .expect("router should accept invocation separately from request data");

        assert_eq!(invocation.run_id().as_str(), "invocation-run");
        assert_eq!(invocation.node_id().as_str(), "invocation-node");
        assert_eq!(response.latent().width(), 64);
    }

    fn registry_with_two_backend_kinds() -> Arc<InferenceBackendRegistry> {
        let mut reg = InferenceBackendRegistry::new();
        reg.register(
            BackendInstanceDescriptor::new(BackendInstance::new("a"), Backend::new("a")),
            Arc::new(EchoBackend::new("a")),
        );
        reg.register(
            BackendInstanceDescriptor::new(BackendInstance::new("b"), Backend::new("b")),
            Arc::new(EchoBackend::new("b")),
        );
        Arc::new(reg)
    }

    fn latent_encode_request_on_two_backends() -> LatentEncodeRequest {
        let vae = crate::RuntimeVaeHandle::new(
            reimagine_core::model::ModelId::new("sdxl"),
            Backend::new("a"),
            "vae-key",
        );
        let image = crate::RuntimeImage::new(
            crate::BackendTensorHandle::new(
                Backend::new("b"),
                crate::BackendPayloadKey::new("image-key"),
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![1, 3, 64, 64]),
                "cpu",
            ),
            64,
            64,
            1,
            "rgb",
        );
        LatentEncodeRequest::new(
            vae,
            image,
            RunId::new("r"),
            WorkflowId::new("w"),
            WorkflowVersion::new(1),
            NodeId::new("n"),
        )
    }

    #[tokio::test]
    async fn reject_all_bridge_policy_blocks_conflicting_cross_backend_affinities() {
        let router = InferenceRouter::new(
            registry_with_two_backend_kinds(),
            Arc::new(RejectAllBridgePolicy),
        );
        let invocation = noop_invocation();
        let err = router
            .latent_encode_with_invocation(&invocation, latent_encode_request_on_two_backends())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::BackendBridgeUnsupported { .. }
        ));
    }

    #[tokio::test]
    async fn cross_worker_bridge_plan_does_not_block_affinity_resolution() {
        let router = InferenceRouter::new(
            registry_with_two_backend_kinds(),
            Arc::new(CrossWorkerBridgePolicy),
        );
        let invocation = noop_invocation();
        let err = router
            .latent_encode_with_invocation(&invocation, latent_encode_request_on_two_backends())
            .await
            .unwrap_err();
        // The CrossWorker plan passed the bridge gate; the call then
        // fails only on capability support (EchoBackend does not
        // advertise latent.encode), never on the bridge policy.
        assert!(matches!(
            err,
            InferenceError::CandidateBackendLacksCapability { .. }
        ));
    }
}
