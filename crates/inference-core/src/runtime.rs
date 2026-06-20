//! Executor-facing router trait and registry-backed default implementation.

use std::sync::Arc;

use crate::BackendKind;
use crate::backend::InferenceBackend;
use crate::bridge::{BackendBridgePolicy, BridgePlan};
use crate::capability::{InferenceBackendCapabilities, InferenceCapability};
use crate::error::InferenceError;
use crate::registry::InferenceBackendRegistry;
use crate::request::diffusion::DiffusionSampleRequest;
use crate::request::image::{ImagePreviewRequest, ImageSaveRequest};
use crate::request::latent::{CreateEmptyLatentRequest, LatentDecodeRequest};
use crate::request::model::LoadBundleRequest;
use crate::request::text::TextEncodeRequest;
use crate::response::diffusion::DiffusionSampleResponse;
use crate::response::image::{ImagePreviewResponse, ImageSaveResponse};
use crate::response::latent::{CreateEmptyLatentResponse, LatentDecodeResponse};
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

    async fn image_save(
        &self,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError>;

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError>;
}

/// Default router: resolves the target backend from
/// [`InferenceBackendRegistry`], consults the
/// [`BackendBridgePolicy`], and dispatches the typed capability call
/// to the selected backend.
///
/// V1 picks the first registered backend when no handle affinity
/// pins a backend. Per-request backend selection and explicit
/// per-capability router methods remain future work tracked in
/// `inference/02` and `inference-core/02` follow-ups.
pub struct DefaultInferenceRuntime {
    registry: Arc<InferenceBackendRegistry>,
    bridge_policy: Arc<dyn BackendBridgePolicy>,
}

impl DefaultInferenceRuntime {
    pub fn new(
        registry: Arc<InferenceBackendRegistry>,
        bridge_policy: Arc<dyn BackendBridgePolicy>,
    ) -> Self {
        Self {
            registry,
            bridge_policy,
        }
    }

    pub fn registry(&self) -> &Arc<InferenceBackendRegistry> {
        &self.registry
    }

    pub fn bridge_policy(&self) -> &Arc<dyn BackendBridgePolicy> {
        &self.bridge_policy
    }

    fn select_backend(&self, affinities: &[BackendKind]) -> Result<BackendKind, InferenceError> {
        match affinities {
            [] => self
                .registry
                .first()
                .map(|backend| backend.backend_kind().clone())
                .ok_or_else(|| InferenceError::BackendNotRegistered {
                    kind: "(any)".to_string(),
                }),
            [kind] => Ok(kind.clone()),
            [first, rest @ ..] => {
                for other in rest {
                    if other != first {
                        return Err(InferenceError::BackendBridgeRequired {
                            source: other.to_string(),
                            target: first.to_string(),
                            capability: InferenceCapability::LoadBundle,
                        });
                    }
                }
                Ok(first.clone())
            }
        }
    }

    fn validate_affinity(
        &self,
        affinities: &[BackendKind],
        target_backend: &BackendKind,
        capability: InferenceCapability,
    ) -> Result<(), InferenceError> {
        for source in affinities {
            if source == target_backend {
                continue;
            }

            match self
                .bridge_policy
                .plan_transfer(source, target_backend, capability)
            {
                BridgePlan::Direct | BridgePlan::Bridgeable { .. } => {}
                BridgePlan::Unsupported { reason } => {
                    return Err(InferenceError::BackendBridgeUnsupported {
                        source: source.to_string(),
                        target: target_backend.to_string(),
                        capability,
                        reason,
                    });
                }
            }
        }
        Ok(())
    }

    /// Resolve the backend for a typed capability call and verify that
    /// the backend advertises the capability.
    fn resolve_backend(
        &self,
        affinities: &[BackendKind],
        capability: InferenceCapability,
    ) -> Result<Arc<dyn InferenceBackend>, InferenceError> {
        let target_backend = self.select_backend(affinities)?;
        self.validate_affinity(affinities, &target_backend, capability)?;

        let backend = self.registry.get(&target_backend).ok_or_else(|| {
            InferenceError::BackendNotRegistered {
                kind: target_backend.to_string(),
            }
        })?;

        let caps: InferenceBackendCapabilities = backend.capabilities();
        if !caps.supports_capability(capability) {
            return Err(InferenceError::BackendCapabilityUnsupported {
                kind: target_backend.to_string(),
                capability,
            });
        }

        Ok(backend)
    }
}

#[async_trait::async_trait]
impl InferenceRuntime for DefaultInferenceRuntime {
    async fn load_bundle(
        &self,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        let backend = self.resolve_backend(
            &request.backend_affinities(),
            InferenceCapability::LoadBundle,
        )?;
        backend.load_bundle(request).await
    }

    async fn text_encode(
        &self,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        let backend = self.resolve_backend(
            &request.backend_affinities(),
            InferenceCapability::TextEncode,
        )?;
        backend.text_encode(request).await
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        let backend = self.resolve_backend(
            &request.backend_affinities(),
            InferenceCapability::CreateEmptyLatent,
        )?;
        backend.create_empty_latent(request).await
    }

    async fn diffusion_sample(
        &self,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        let backend = self.resolve_backend(
            &request.backend_affinities(),
            InferenceCapability::DiffusionSample,
        )?;
        backend.diffusion_sample(request).await
    }

    async fn latent_decode(
        &self,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        let backend = self.resolve_backend(
            &request.backend_affinities(),
            InferenceCapability::LatentDecode,
        )?;
        backend.latent_decode(request).await
    }

    async fn image_save(
        &self,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        let backend = self.resolve_backend(
            &request.backend_affinities(),
            InferenceCapability::ImageSave,
        )?;
        backend.image_save(request).await
    }

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        let backend = self.resolve_backend(
            &request.backend_affinities(),
            InferenceCapability::ImagePreview,
        )?;
        backend.image_preview(request).await
    }
}

impl std::fmt::Debug for DefaultInferenceRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultInferenceRuntime")
            .field("registry", &self.registry)
            .field("bridge_policy", &"<dyn BackendBridgePolicy>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeLatent;
    use crate::backend::InferenceBackend;
    use crate::capability::{InferenceBackendCapabilities, InferenceCapabilitySupport};
    use crate::error::InferenceError;
    use crate::registry::InferenceBackendRegistry;
    use crate::request::latent::CreateEmptyLatentRequest;
    use crate::response::latent::CreateEmptyLatentResponse;
    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

    struct EchoBackend;

    #[async_trait::async_trait]
    impl InferenceBackend for EchoBackend {
        fn backend_kind(&self) -> &BackendKind {
            static KIND: std::sync::OnceLock<BackendKind> = std::sync::OnceLock::new();
            KIND.get_or_init(|| BackendKind::new("echo"))
        }
        fn capabilities(&self) -> InferenceBackendCapabilities {
            InferenceBackendCapabilities::new(BackendKind::new("echo")).with_support(
                InferenceCapabilitySupport::new(InferenceCapability::CreateEmptyLatent),
            )
        }
        async fn load_bundle(
            &self,
            _request: LoadBundleRequest,
        ) -> Result<LoadBundleResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::LoadBundle,
                backend_kind: "echo".to_string(),
                message: None,
            })
        }
        async fn text_encode(
            &self,
            _request: TextEncodeRequest,
        ) -> Result<TextEncodeResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::TextEncode,
                backend_kind: "echo".to_string(),
                message: None,
            })
        }
        async fn create_empty_latent(
            &self,
            _request: CreateEmptyLatentRequest,
        ) -> Result<CreateEmptyLatentResponse, InferenceError> {
            Ok(CreateEmptyLatentResponse::new(RuntimeLatent::new(
                crate::BackendTensorHandle::new(
                    BackendKind::new("echo"),
                    crate::BackendPayloadKey::new("empty"),
                    reimagine_core::model::TensorDType::F32,
                    reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                    "cpu",
                ),
                64,
                64,
                1,
                4,
            )))
        }
        async fn diffusion_sample(
            &self,
            _request: DiffusionSampleRequest,
        ) -> Result<DiffusionSampleResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::DiffusionSample,
                backend_kind: "echo".to_string(),
                message: None,
            })
        }
        async fn latent_decode(
            &self,
            _request: LatentDecodeRequest,
        ) -> Result<LatentDecodeResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::LatentDecode,
                backend_kind: "echo".to_string(),
                message: None,
            })
        }
        async fn image_save(
            &self,
            _request: ImageSaveRequest,
        ) -> Result<ImageSaveResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::ImageSave,
                backend_kind: "echo".to_string(),
                message: None,
            })
        }
        async fn image_preview(
            &self,
            _request: ImagePreviewRequest,
        ) -> Result<ImagePreviewResponse, InferenceError> {
            Err(InferenceError::BackendNotImplemented {
                capability: InferenceCapability::ImagePreview,
                backend_kind: "echo".to_string(),
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

    #[tokio::test]
    async fn routes_through_registry_first_backend() {
        let mut reg = InferenceBackendRegistry::new();
        reg.register(Arc::new(EchoBackend));
        let runtime = DefaultInferenceRuntime::new(
            Arc::new(reg),
            Arc::new(crate::bridge::RejectAllBridgePolicy),
        );
        let response = runtime
            .create_empty_latent(empty_latent_request())
            .await
            .unwrap();
        assert_eq!(response.latent().width(), 64);
    }

    #[tokio::test]
    async fn returns_backend_not_registered_when_registry_empty() {
        let runtime = DefaultInferenceRuntime::new(
            Arc::new(InferenceBackendRegistry::new()),
            Arc::new(crate::bridge::RejectAllBridgePolicy),
        );
        let err = runtime
            .create_empty_latent(empty_latent_request())
            .await
            .unwrap_err();
        assert!(matches!(err, InferenceError::BackendNotRegistered { .. }));
    }

    #[tokio::test]
    async fn returns_capability_unsupported_for_unsupported_capability() {
        let mut reg = InferenceBackendRegistry::new();
        reg.register(Arc::new(EchoBackend));
        let runtime = DefaultInferenceRuntime::new(
            Arc::new(reg),
            Arc::new(crate::bridge::RejectAllBridgePolicy),
        );
        // Echo backend does not advertise TextEncode. Call text_encode to
        // exercise the unsupported path.
        let clip = crate::RuntimeClipHandle::new(
            reimagine_core::model::ModelId::new("clip"),
            BackendKind::new("echo"),
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
            InferenceError::BackendCapabilityUnsupported { .. }
        ));
    }
}
