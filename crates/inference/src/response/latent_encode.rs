//! `latent.encode` response DTO.
//!
//! Carries a [`RuntimeLatent`] whose [`LatentContent`] is
//! `EncodedImage`. V1 production code paths return real
//! `EncodedImage` latents; backends that defer real VAE encoding
//! return a precise `BackendNotImplemented` rather than a silent
//! empty latent.

use crate::RuntimeLatent;

/// `latent.encode` response.
#[derive(Debug, Clone)]
pub struct LatentEncodeResponse {
    latent: RuntimeLatent,
}

impl LatentEncodeResponse {
    pub fn new(latent: RuntimeLatent) -> Self {
        Self { latent }
    }

    pub fn latent(&self) -> &RuntimeLatent {
        &self.latent
    }

    /// Consume the response and return its inner latent.
    pub fn into_latent(self) -> RuntimeLatent {
        self.latent
    }
}
