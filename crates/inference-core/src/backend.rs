//! The backend-neutral inference execution trait.
//!
//! [`InferenceBackend`] is the central async trait that concrete
//! backends implement. The executor-facing router dispatches typed
//! capability calls to the selected backend via this trait.
//!
//! The primary surface is the typed capability methods. Each method
//! takes a capability-specific request DTO and returns a
//! capability-specific response DTO. The trait does not expose a
//! `Vec<SlotId, ExecutionValue>` envelope and does not require
//! callers to construct an `InferenceOperationId`.

use crate::capability::InferenceBackendCapabilities;
use crate::error::InferenceError;
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
