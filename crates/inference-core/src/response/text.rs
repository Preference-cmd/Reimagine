//! `text.encode` response DTO.

use reimagine_core::ExecutionConditioning;

/// `text.encode` response.
#[derive(Debug, Clone)]
pub struct TextEncodeResponse {
    conditioning: ExecutionConditioning,
}

impl TextEncodeResponse {
    pub fn new(conditioning: ExecutionConditioning) -> Self {
        Self { conditioning }
    }

    pub fn conditioning(&self) -> &ExecutionConditioning {
        &self.conditioning
    }

    /// Consume the response and return its inner conditioning.
    pub fn into_conditioning(self) -> ExecutionConditioning {
        self.conditioning
    }
}
