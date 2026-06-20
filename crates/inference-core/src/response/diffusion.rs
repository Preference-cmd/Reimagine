//! `diffusion.sample` response DTO.

use crate::RuntimeLatent;

/// `diffusion.sample` response.
#[derive(Debug, Clone)]
pub struct DiffusionSampleResponse {
    latent: RuntimeLatent,
}

impl DiffusionSampleResponse {
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
