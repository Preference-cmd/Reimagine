//! `latent.encode` operation.
//!
//! The capability is reserved in this slice. V1 Candle advertises
//! it in capability reports only when real weight-driven
//! implementations land; the current implementation returns
//! [`InferenceError::BackendNotImplemented`] with a precise capability
//! label so the executor surfaces a deterministic, non-retryable
//! failure rather than silently degrading to a placeholder tensor.

use reimagine_inference::{LatentEncodeRequest, LatentEncodeResponse};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;

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
