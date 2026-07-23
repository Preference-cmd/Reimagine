//! `image.import` response DTO.

use crate::RuntimeImage;

/// `image.import` response.
#[derive(Debug, Clone)]
pub struct ImageImportResponse {
    image: RuntimeImage,
}

impl ImageImportResponse {
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
