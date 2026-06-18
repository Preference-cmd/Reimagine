//! Executor-facing router trait and registry-backed default implementation.

use std::sync::Arc;

use crate::bridge::{BackendBridgePolicy, BridgePlan};
use crate::error::InferenceError;
use crate::registry::InferenceBackendRegistry;
use crate::request::{
    InferenceRequest, OP_DIFFUSION_SAMPLE, OP_IMAGE_PREVIEW, OP_IMAGE_SAVE, OP_LATENT_CREATE_EMPTY,
    OP_LATENT_DECODE, OP_MODEL_LOAD_BUNDLE, OP_TEXT_ENCODE,
};
use crate::response::InferenceResponse;
use reimagine_core::BackendKind;

/// Executor-facing router. Built-in executors call this trait rather
/// than a concrete backend directly.
#[async_trait::async_trait]
pub trait InferenceRuntime: Send + Sync + 'static {
    async fn execute(&self, request: InferenceRequest)
    -> Result<InferenceResponse, InferenceError>;

    async fn load_bundle(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        debug_assert_eq!(request.operation_id().as_str(), OP_MODEL_LOAD_BUNDLE);
        self.execute(request).await
    }

    async fn text_encode(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        debug_assert_eq!(request.operation_id().as_str(), OP_TEXT_ENCODE);
        self.execute(request).await
    }

    async fn create_empty_latent(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        debug_assert_eq!(request.operation_id().as_str(), OP_LATENT_CREATE_EMPTY);
        self.execute(request).await
    }

    async fn diffusion_sample(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        debug_assert_eq!(request.operation_id().as_str(), OP_DIFFUSION_SAMPLE);
        self.execute(request).await
    }

    async fn latent_decode(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        debug_assert_eq!(request.operation_id().as_str(), OP_LATENT_DECODE);
        self.execute(request).await
    }

    async fn image_save(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        debug_assert_eq!(request.operation_id().as_str(), OP_IMAGE_SAVE);
        self.execute(request).await
    }

    async fn image_preview(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        debug_assert_eq!(request.operation_id().as_str(), OP_IMAGE_PREVIEW);
        self.execute(request).await
    }
}

/// Default router: resolves the target backend from
/// [`InferenceBackendRegistry`], consults the
/// [`BackendBridgePolicy`], and dispatches the request.
///
/// V1 picks the first registered backend. Per-request backend
/// selection and explicit per-capability router methods are
/// follow-up work tracked in `inference/02` and `inference-core/02`.
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
}

#[async_trait::async_trait]
impl InferenceRuntime for DefaultInferenceRuntime {
    async fn execute(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let op_id = request.operation_id().clone();
        let target_backend = self.select_backend(&request)?;
        self.validate_affinity(&request, &target_backend, &op_id)?;
        let backend = self.registry.get(&target_backend).ok_or_else(|| {
            InferenceError::BackendNotRegistered {
                kind: target_backend.to_string(),
            }
        })?;

        let caps = backend.capabilities();
        if !caps.supports_operation(&op_id) {
            return Err(InferenceError::BackendCapabilityUnsupported {
                kind: target_backend.to_string(),
                operation_id: op_id.to_string(),
            });
        }

        dispatch_typed(backend.as_ref(), request).await
    }
}

impl DefaultInferenceRuntime {
    fn select_backend(&self, request: &InferenceRequest) -> Result<BackendKind, InferenceError> {
        let affinities = request.backend_affinities();
        match affinities.as_slice() {
            [kind] => Ok(kind.clone()),
            [] => self
                .registry
                .first()
                .map(|backend| backend.backend_kind().clone())
                .ok_or_else(|| InferenceError::BackendNotRegistered {
                    kind: "(any)".to_string(),
                }),
            [first, rest @ ..] => {
                for other in rest {
                    if other != first {
                        return Err(InferenceError::BackendBridgeRequired {
                            source: other.to_string(),
                            target: first.to_string(),
                            operation_id: request.operation_id().to_string(),
                        });
                    }
                }
                Ok(first.clone())
            }
        }
    }

    fn validate_affinity(
        &self,
        request: &InferenceRequest,
        target_backend: &BackendKind,
        op_id: &crate::request::InferenceOperationId,
    ) -> Result<(), InferenceError> {
        for source in request.backend_affinities() {
            if &source == target_backend {
                continue;
            }

            match self
                .bridge_policy
                .plan_transfer(&source, target_backend, op_id)
            {
                BridgePlan::Direct | BridgePlan::Bridgeable { .. } => {}
                BridgePlan::Unsupported { reason } => {
                    return Err(InferenceError::BackendBridgeUnsupported {
                        source: source.to_string(),
                        target: target_backend.to_string(),
                        operation_id: op_id.to_string(),
                        reason,
                    });
                }
            }
        }
        Ok(())
    }
}

async fn dispatch_typed(
    backend: &dyn crate::backend::InferenceBackend,
    request: InferenceRequest,
) -> Result<InferenceResponse, InferenceError> {
    match request.operation_id().as_str() {
        OP_MODEL_LOAD_BUNDLE => backend.load_bundle(request).await,
        OP_TEXT_ENCODE => backend.text_encode(request).await,
        OP_LATENT_CREATE_EMPTY => backend.create_empty_latent(request).await,
        OP_DIFFUSION_SAMPLE => backend.diffusion_sample(request).await,
        OP_LATENT_DECODE => backend.latent_decode(request).await,
        OP_IMAGE_SAVE => backend.image_save(request).await,
        OP_IMAGE_PREVIEW => backend.image_preview(request).await,
        _ => backend.execute(request).await,
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
    use crate::backend::InferenceBackend;
    use crate::capability::{InferenceBackendCapabilities, InferenceOperationSupport};
    use crate::request::{InferenceRequest, OP_DIFFUSION_SAMPLE, OP_LATENT_CREATE_EMPTY};
    use crate::response::InferenceResponse;
    use reimagine_core::BackendKind;
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
                InferenceOperationSupport::new(OP_LATENT_CREATE_EMPTY.into()),
            )
        }
        async fn execute(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
            Ok(InferenceResponse::new(vec![]))
        }
    }

    fn empty_request(op: &str) -> InferenceRequest {
        InferenceRequest::new(
            op.into(),
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
            .execute(empty_request(OP_LATENT_CREATE_EMPTY))
            .await
            .unwrap();
        assert!(response.outputs().is_empty());
    }

    #[tokio::test]
    async fn returns_backend_not_registered_when_registry_empty() {
        let runtime = DefaultInferenceRuntime::new(
            Arc::new(InferenceBackendRegistry::new()),
            Arc::new(crate::bridge::RejectAllBridgePolicy),
        );
        let err = runtime
            .execute(empty_request(OP_LATENT_CREATE_EMPTY))
            .await
            .unwrap_err();
        assert!(matches!(err, InferenceError::BackendNotRegistered { .. }));
    }

    #[tokio::test]
    async fn returns_capability_unsupported_for_unknown_op() {
        let mut reg = InferenceBackendRegistry::new();
        reg.register(Arc::new(EchoBackend));
        let runtime = DefaultInferenceRuntime::new(
            Arc::new(reg),
            Arc::new(crate::bridge::RejectAllBridgePolicy),
        );
        let err = runtime
            .execute(empty_request(OP_DIFFUSION_SAMPLE))
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            InferenceError::BackendCapabilityUnsupported { .. }
        ));
    }
}
