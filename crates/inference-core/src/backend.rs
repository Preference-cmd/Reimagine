//! The backend-neutral inference execution trait.
//!
//! [`InferenceBackend`] is the central async trait that concrete
//! backends implement. The executor-facing router dispatches
//! requests to the selected backend via this trait.

use crate::capability::InferenceBackendCapabilities;
use crate::error::InferenceError;
use crate::request::{
    InferenceRequest, OP_DIFFUSION_SAMPLE, OP_IMAGE_PREVIEW, OP_IMAGE_SAVE, OP_LATENT_CREATE_EMPTY,
    OP_LATENT_DECODE, OP_MODEL_LOAD_BUNDLE, OP_TEXT_ENCODE,
};
use crate::response::InferenceResponse;
use reimagine_core::BackendKind;

/// Backend-neutral inference execution trait.
///
/// V1 uses `async_trait` for a readable async trait-object surface,
/// matching the pattern already used by `runtime::NodeExecutor` and
/// `agent::AgentProvider`.
#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync + 'static {
    /// The stable kind identifier for this backend (e.g. `"candle"`,
    /// `"fake"`, `"remote"`).
    fn backend_kind(&self) -> &BackendKind;

    /// The capabilities this backend advertises.
    fn capabilities(&self) -> InferenceBackendCapabilities;

    /// Execute a single inference operation.
    ///
    /// Compatibility envelope for existing backend implementations. Typed
    /// capability methods below are the primary contract. Backends may keep
    /// this dispatcher while migrating, but router/executor code should prefer
    /// the typed methods.
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
