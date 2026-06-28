//! `image.import` and `latent.encode` operations.
//!
//! Both capabilities are reserved in this slice. V1 Candle advertises
//! them in capability reports only when real weight-driven
//! implementations land; the current implementation returns
//! [`InferenceError::BackendNotImplemented`] with a precise capability
//! label so the executor surfaces a deterministic, non-retryable
//! failure rather than silently degrading to a placeholder tensor.

use reimagine_inference::{
    ImageImportRequest, ImageImportResponse, InferenceError, LatentEncodeRequest,
    LatentEncodeResponse,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;

/// `image.import` operation.
///
/// V1 reserves `builtin.load_image` and the typed
/// [`ImageImportRequest`] for future Candle implementations. The
/// current Candle backend does not yet decode user-supplied images
/// into its payload store; the executor must surface this as a
/// precise `BackendNotImplemented` failure rather than silently
/// producing a placeholder tensor.
pub fn execute_image_import(
    _backend: &CandleBackend,
    _request: ImageImportRequest,
) -> Result<ImageImportResponse, CandleBackendError> {
    Err(CandleBackendError::BackendNotImplemented(
        crate::error::BackendNotImplementedError::new(
            "candle",
            reimagine_inference::InferenceCapability::ImageImport,
            "image.import is reserved in this slice; V1 Candle does not yet decode user-supplied images into its payload store",
        ),
    ))
}

/// `latent.encode` operation.
///
/// V1 reserves `builtin.vae_encode` and the typed
/// [`LatentEncodeRequest`] for future Candle implementations. Real
/// VAE encode on weights is deferred; the current Candle backend
/// surfaces a precise `BackendNotImplemented` failure.
pub fn execute_latent_encode(
    _backend: &CandleBackend,
    _request: LatentEncodeRequest,
) -> Result<LatentEncodeResponse, CandleBackendError> {
    Err(CandleBackendError::BackendNotImplemented(
        crate::error::BackendNotImplementedError::new(
            "candle",
            reimagine_inference::InferenceCapability::LatentEncode,
            "latent.encode is reserved in this slice; V1 Candle does not yet run a real VAE encoder",
        ),
    ))
}

// Re-import for callers that only see the inference error type.
#[allow(dead_code)]
fn _error_alias(err: CandleBackendError) -> InferenceError {
    match err {
        CandleBackendError::BackendNotImplemented(inner) => InferenceError::BackendNotImplemented {
            capability: inner.capability(),
            backend_kind: inner.backend_kind().to_string(),
            message: Some(inner.message().to_string()),
        },
        other => panic!("unexpected error variant: {other:?}"),
    }
}
