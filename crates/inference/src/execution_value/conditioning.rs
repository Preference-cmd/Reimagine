//! `ExecutionConditioning` is the public conditioning value carried by
//! [`ExecutionValue::Conditioning`](super::value::ExecutionValue::Conditioning).
//!
//! It bundles:
//!
//! - the text-embedding tensor handle (always present)
//! - an optional pooled-embedding tensor handle (used by SDXL UNet)
//! - a [`ConditioningMetadata`] carrying public execution context
//!
//! Conditioning metadata is part of the conditioning value in V1 and
//! is not split into a separate public abstraction.

use super::handles::BackendTensorHandle;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ConditioningMetadata {
    width: u32,
    height: u32,
    crop_x: u32,
    crop_y: u32,
    target_width: u32,
    target_height: u32,
}

impl ConditioningMetadata {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            crop_x: 0,
            crop_y: 0,
            target_width: width,
            target_height: height,
        }
    }

    pub fn with_crop(mut self, crop_x: u32, crop_y: u32) -> Self {
        self.crop_x = crop_x;
        self.crop_y = crop_y;
        self
    }

    pub fn with_target_size(mut self, target_width: u32, target_height: u32) -> Self {
        self.target_width = target_width;
        self.target_height = target_height;
        self
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn crop_x(&self) -> u32 {
        self.crop_x
    }

    pub fn crop_y(&self) -> u32 {
        self.crop_y
    }

    pub fn target_width(&self) -> u32 {
        self.target_width
    }

    pub fn target_height(&self) -> u32 {
        self.target_height
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExecutionConditioning {
    text_embedding: BackendTensorHandle,
    pooled_embedding: Option<BackendTensorHandle>,
    metadata: ConditioningMetadata,
}

impl ExecutionConditioning {
    pub fn new(text_embedding: BackendTensorHandle, metadata: ConditioningMetadata) -> Self {
        Self {
            text_embedding,
            pooled_embedding: None,
            metadata,
        }
    }

    pub fn with_pooled_embedding(mut self, pooled_embedding: BackendTensorHandle) -> Self {
        self.pooled_embedding = Some(pooled_embedding);
        self
    }

    pub fn text_embedding(&self) -> &BackendTensorHandle {
        &self.text_embedding
    }

    pub fn pooled_embedding(&self) -> Option<&BackendTensorHandle> {
        self.pooled_embedding.as_ref()
    }

    pub fn metadata(&self) -> &ConditioningMetadata {
        &self.metadata
    }
}
