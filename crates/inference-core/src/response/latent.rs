//! `latent.create_empty` and `latent.decode` response DTOs.

use crate::RuntimeImage;
use crate::RuntimeLatent;

/// `latent.create_empty` response.
#[derive(Debug, Clone)]
pub struct CreateEmptyLatentResponse {
    latent: RuntimeLatent,
}

impl CreateEmptyLatentResponse {
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

/// `latent.decode` response.
#[derive(Debug, Clone)]
pub struct LatentDecodeResponse {
    image: RuntimeImage,
}

impl LatentDecodeResponse {
    pub fn new(image: RuntimeImage) -> Self {
        Self { image }
    }

    pub fn image(&self) -> &RuntimeImage {
        &self.image
    }

    /// Consume the response and return its inner image.
    pub fn into_image(self) -> RuntimeImage {
        self.image
    }
}
