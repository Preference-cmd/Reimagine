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

use crate::Backend;
use crate::capability::InferenceBackendCapabilities;
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

/// Backend-neutral inference execution trait.
///
/// V1 uses `async_trait` for a readable async trait-object surface,
/// matching the pattern already used by `runtime::NodeExecutor` and
/// `agent::AgentProvider`.
#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync + 'static {
    /// The stable kind identifier for this backend (e.g. `"candle"`,
    /// `"fake"`, `"remote"`).
    fn backend_kind(&self) -> &Backend;

    /// The capabilities this backend advertises.
    fn capabilities(&self) -> InferenceBackendCapabilities;

    fn admit_invocation(&self, _invocation: &InferenceInvocation) -> Result<(), InferenceError> {
        Ok(())
    }

    fn finish_invocation(&self, _invocation: &InferenceInvocation) {}

    async fn load_bundle(
        &self,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError>;

    async fn load_bundle_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.load_bundle(request).await;
        self.finish_invocation(_invocation);
        result
    }

    async fn text_encode(
        &self,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError>;

    async fn text_encode_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.text_encode(request).await;
        self.finish_invocation(_invocation);
        result
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError>;

    async fn create_empty_latent_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.create_empty_latent(request).await;
        self.finish_invocation(_invocation);
        result
    }

    async fn diffusion_sample(
        &self,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError>;

    async fn diffusion_sample_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.diffusion_sample(request).await;
        self.finish_invocation(_invocation);
        result
    }

    async fn latent_decode(
        &self,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError>;

    async fn latent_decode_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.latent_decode(request).await;
        self.finish_invocation(_invocation);
        result
    }

    async fn latent_encode(
        &self,
        request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError>;

    async fn latent_encode_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.latent_encode(request).await;
        self.finish_invocation(_invocation);
        result
    }

    async fn image_import(
        &self,
        request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError>;

    async fn image_import_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.image_import(request).await;
        self.finish_invocation(_invocation);
        result
    }

    async fn image_save(
        &self,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError>;

    async fn image_save_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.image_save(request).await;
        self.finish_invocation(_invocation);
        result
    }

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError>;

    async fn image_preview_with_invocation(
        &self,
        _invocation: &InferenceInvocation,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        self.admit_invocation(_invocation)?;
        let result = self.image_preview(request).await;
        self.finish_invocation(_invocation);
        result
    }
}
