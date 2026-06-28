//! Executor-facing router trait and registry-backed default implementation.

use std::sync::Arc;

use crate::Backend;
use crate::backend_registry::{InferenceBackendRegistry, RegisteredInstance};
use crate::backend_selection::{
    BackendInstance, BackendSelectionPolicy, BackendSelectionRequest, StaticBackendSelectionPolicy,
};
use crate::bridge::{BackendBridgePolicy, BridgePlan};
use crate::capability::InferenceCapability;
use crate::inference_error::InferenceError;
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

/// Executor-facing router. Built-in executors call this trait rather
/// than a concrete backend directly.
#[async_trait::async_trait]
pub trait InferenceRuntime: Send + Sync + 'static {
    async fn load_bundle(
        &self,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError>;

    async fn text_encode(
        &self,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError>;

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError>;

    async fn diffusion_sample(
        &self,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError>;

    async fn latent_decode(
        &self,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError>;

    async fn latent_encode(
        &self,
        request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError>;

    async fn image_import(
        &self,
        request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError>;

    async fn image_save(
        &self,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError>;

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError>;
}

/// Default router: applies a [`BackendSelectionPolicy`] to derive a
/// candidate [`BackendInstance`], consults the
/// [`BackendBridgePolicy`] for cross-backend handle conflicts, and
/// dispatches the typed capability call to the selected backend.
///
/// Selection precedence (deterministic, see `docs/architecture/modules/inference.md`):
///
/// 1. Existing backend-bound handle affinities. Conflicting
///    affinities ask the bridge policy; no silent fallback.
/// 2. Explicit override from the request overlay (when no
///    incompatible handles exist).
/// 3. Priority order from the policy.
/// 4. Diagnostic failure.
pub struct DefaultInferenceRuntime {
    registry: Arc<InferenceBackendRegistry>,
    selection_policy: Arc<dyn BackendSelectionPolicy>,
    bridge_policy: Arc<dyn BackendBridgePolicy>,
}

impl DefaultInferenceRuntime {
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
                BridgePlan::Direct | BridgePlan::Bridgeable { .. } => {}
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

    fn build_selection_request(
        &self,
        capability: InferenceCapability,
        node_id: Option<reimagine_core::model::NodeId>,
        backend_affinities: &[BackendInstance],
        explicit_override: Option<BackendInstance>,
    ) -> BackendSelectionRequest {
        BackendSelectionRequest {
            capability,
            node_id,
            affinities: backend_affinities.to_vec(),
            registered: self.registry.descriptors(),
            explicit_override,
        }
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
            // Record the first capability-missing candidate so we
            // can return a precise diagnostic if no viable
            // candidate is found.
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
}

#[async_trait::async_trait]
impl InferenceRuntime for DefaultInferenceRuntime {
    async fn load_bundle(
        &self,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::LoadBundle,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.load_bundle(request).await
    }

    async fn text_encode(
        &self,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::TextEncode,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.text_encode(request).await
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::CreateEmptyLatent,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.create_empty_latent(request).await
    }

    async fn diffusion_sample(
        &self,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::DiffusionSample,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.diffusion_sample(request).await
    }

    async fn latent_decode(
        &self,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::LatentDecode,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.latent_decode(request).await
    }

    async fn latent_encode(
        &self,
        request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::LatentEncode,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.latent_encode(request).await
    }

    async fn image_import(
        &self,
        request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::ImageImport,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.image_import(request).await
    }

    async fn image_save(
        &self,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::ImageSave,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.image_save(request).await
    }

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        let selection = self.build_selection_request(
            InferenceCapability::ImagePreview,
            Some(request.node_id().clone()),
            &request.backend_affinities(),
            request
                .backend_selection_overlay()
                .explicit_override
                .clone(),
        );
        let backend = self.resolve_backend(&selection)?;
        backend.backend.image_preview(request).await
    }
}

impl std::fmt::Debug for DefaultInferenceRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultInferenceRuntime")
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

    #[tokio::test]
    async fn routes_through_first_registered_when_no_affinity() {
        let registry = make_registry_with_single_candle();
        let runtime = DefaultInferenceRuntime::new(registry, Arc::new(RejectAllBridgePolicy));
        let response = runtime
            .create_empty_latent(empty_latent_request())
            .await
            .unwrap();
        assert_eq!(response.latent().width(), 64);
    }

    #[tokio::test]
    async fn returns_no_candidate_when_registry_empty_and_no_policy() {
        let runtime = DefaultInferenceRuntime::new(
            Arc::new(InferenceBackendRegistry::new()),
            Arc::new(RejectAllBridgePolicy),
        );
        let err = runtime
            .create_empty_latent(empty_latent_request())
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::BackendSelectionNoCandidate { .. }
        ));
    }

    #[tokio::test]
    async fn returns_capability_unsupported_when_no_candidate_supports_capability() {
        // Echo backend does not advertise TextEncode. The clip
        // handle's affinity is `candle`, and candle:cpu is the only
        // registered instance. The router must report that the
        // affinity-targeted instance lacks the requested
        // capability.
        let registry = make_registry_with_single_candle();
        let policy = StaticBackendSelectionPolicy::new(vec![BackendInstance::new("candle:cpu")]);
        let runtime = DefaultInferenceRuntime::with_policy(
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
        let err = runtime.text_encode(req).await.unwrap_err();
        assert!(matches!(
            err,
            InferenceError::CandidateBackendLacksCapability { .. }
        ));
    }

    #[tokio::test]
    async fn explicit_override_pins_known_instance() {
        let registry = make_registry_with_single_candle();
        let runtime = DefaultInferenceRuntime::new(registry, Arc::new(RejectAllBridgePolicy));
        // Use the request overlay to pin candle:cpu.
        let mut req = empty_latent_request();
        req.set_backend_selection_overlay(crate::BackendSelectionOverlay::with_explicit_override(
            BackendInstance::new("candle:cpu"),
        ));
        let response = runtime.create_empty_latent(req).await.unwrap();
        assert_eq!(response.latent().width(), 64);
    }

    #[tokio::test]
    async fn explicit_override_for_missing_instance_returns_not_registered() {
        let registry = make_registry_with_single_candle();
        let runtime = DefaultInferenceRuntime::new(registry, Arc::new(RejectAllBridgePolicy));
        let mut req = empty_latent_request();
        req.set_backend_selection_overlay(crate::BackendSelectionOverlay::with_explicit_override(
            BackendInstance::new("missing"),
        ));
        let err = runtime.create_empty_latent(req).await.unwrap_err();
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
        let runtime = DefaultInferenceRuntime::with_policy(
            registry,
            Arc::new(policy),
            Arc::new(RejectAllBridgePolicy),
        );
        let mut req = empty_latent_request();
        req.set_backend_selection_overlay(crate::BackendSelectionOverlay::with_explicit_override(
            BackendInstance::new("candle:cpu"),
        ));
        let err = runtime.create_empty_latent(req).await.unwrap_err();
        assert!(matches!(
            err,
            InferenceError::BackendSelectionNoCandidate { .. }
        ));
    }

    #[tokio::test]
    async fn handle_affinity_pins_concrete_backend_instance() {
        let registry = make_registry_with_two_candle_instances();
        let runtime = DefaultInferenceRuntime::new(registry, Arc::new(RejectAllBridgePolicy));
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
        let err = runtime.text_encode(req).await.unwrap_err();
        match err {
            InferenceError::CandidateBackendLacksCapability { instance, .. } => {
                assert_eq!(instance, BackendInstance::new("candle:metal"));
            }
            other => panic!("expected CandidateBackendLacksCapability, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn affinity_for_unregistered_backend_returns_not_registered() {
        // The affinity names a backend that no instance advertises.
        // The router's affinity branch short-circuits to
        // "candidate not registered" without consulting the bridge
        // policy. Bridge policy coverage is exercised by the
        // `bridge_policy_rejects_cross_backend_transfer` test in
        // `bridge.rs`.
        let mut reg = InferenceBackendRegistry::new();
        reg.register(
            BackendInstanceDescriptor::new(BackendInstance::new("a:cpu"), Backend::new("a")),
            Arc::new(EchoBackend::new("a")),
        );
        reg.register(
            BackendInstanceDescriptor::new(BackendInstance::new("b:cpu"), Backend::new("b")),
            Arc::new(EchoBackend::new("b")),
        );
        let runtime = DefaultInferenceRuntime::new(Arc::new(reg), Arc::new(RejectAllBridgePolicy));

        // The clip handle's affinity is `other` — a backend that
        // no registered instance advertises.
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
        let err = runtime.text_encode(req).await.unwrap_err();
        assert!(matches!(
            err,
            InferenceError::CandidateBackendNotRegistered { .. }
        ));
    }

    #[tokio::test]
    async fn empty_registry_produces_no_candidate_with_registered_zero() {
        // When the registry is empty and no override is supplied,
        // the router must report "no candidate" with
        // `registered == 0` so downstream diagnostic consumers can
        // distinguish "nothing registered" from "policy filtered
        // every candidate".
        let runtime = DefaultInferenceRuntime::new(
            Arc::new(InferenceBackendRegistry::new()),
            Arc::new(RejectAllBridgePolicy),
        );
        let err = runtime
            .create_empty_latent(empty_latent_request())
            .await
            .unwrap_err();
        match err {
            InferenceError::BackendSelectionNoCandidate { registered, .. } => {
                assert_eq!(registered, 0);
            }
            other => panic!("expected BackendSelectionNoCandidate, got {other:?}"),
        }
    }
}
