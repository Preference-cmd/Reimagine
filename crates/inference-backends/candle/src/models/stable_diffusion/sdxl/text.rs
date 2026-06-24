//! SDXL text encoder implementation.
//!
//! SDXL uses a dual CLIP text encoder architecture:
//! - CLIP-L (ViT-L/14): produces 768-dimensional text embeddings
//! - CLIP-G (ViT-bigG/14): produces 1280-dimensional text embeddings
//!   and a 1280-dimensional pooled embedding
//!
//! The final text embedding is the concatenation of CLIP-L and CLIP-G
//! outputs along the feature dimension: [batch, seq_len, 2048].
//!
//! V1 uses placeholder conditioning tensors with correct shapes. The
//! actual CLIP inference will be implemented when candle CLIP weights
//! are integrated. Tokenization is performed via the bundle's
//! `SdxlTokenizer` and converted to tensors on the bundle's device.

use candle_core::{DType, Tensor};

use super::LoadedSdxlBundle;
use crate::error::CandleBackendError;
use crate::models::stable_diffusion::sdxl::tokenizer::MAX_SEQUENCE_LENGTH;

/// SDXL CLIP-L embedding dimension.
const CLIP_L_DIM: usize = 768;

/// SDXL CLIP-G embedding dimension.
const CLIP_G_DIM: usize = 1280;

/// Combined text embedding dimension (CLIP-L + CLIP-G).
const COMBINED_DIM: usize = CLIP_L_DIM + CLIP_G_DIM; // 2048

/// SDXL text encoder that produces conditioning tensors.
///
/// Stateless namespace — the tokenizer lives on [`LoadedSdxlBundle`]
/// and is constructed once during bundle loading.
pub struct SdxlTextEncoder;

impl SdxlTextEncoder {
    /// Encode text into conditioning tensors.
    ///
    /// Tokenizes `text` using the bundle's tokenizer, converts the
    /// token ids and attention mask to tensors, and produces
    /// placeholder conditioning tensors with correct SDXL shapes.
    ///
    /// Returns `(text_embedding, pooled_embedding)` tensors. The text
    /// embedding has shape `[1, MAX_SEQUENCE_LENGTH, COMBINED_DIM]`
    /// and the pooled embedding has shape `[1, CLIP_G_DIM]`.
    pub fn encode(
        bundle: &LoadedSdxlBundle,
        text: &str,
    ) -> Result<(Tensor, Tensor), TextEncoderError> {
        let prompt = bundle.tokenizer.tokenize(text)?;

        let token_ids_tensor = Tensor::from_vec(
            prompt.token_ids,
            (1, MAX_SEQUENCE_LENGTH),
            bundle.device.as_ref(),
        )
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;

        let _attention_mask_tensor = Tensor::from_vec(
            prompt.attention_mask,
            (1, MAX_SEQUENCE_LENGTH),
            bundle.device.as_ref(),
        )
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;

        // Placeholder conditioning tensors — real CLIP forward pass is Issue 03.
        // Shape: [1, 77, 2048] + [1, 1280]
        let text_embedding = Tensor::zeros(
            (1, MAX_SEQUENCE_LENGTH, COMBINED_DIM),
            DType::F32,
            bundle.device.as_ref(),
        )
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;

        let pooled_embedding = Tensor::zeros((1, CLIP_G_DIM), DType::F32, bundle.device.as_ref())
            .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;

        // TODO: remove allow when token_ids_tensor is wired to CLIP forward pass
        let _ = token_ids_tensor;

        Ok((text_embedding, pooled_embedding))
    }
}

#[derive(Debug, Clone)]
pub enum TextEncoderError {
    TensorCreation(String),
    TokenizerError(String),
}

impl std::fmt::Display for TextEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorCreation(msg) => write!(f, "text encoder tensor creation failed: {msg}"),
            Self::TokenizerError(msg) => write!(f, "text encoder tokenizer error: {msg}"),
        }
    }
}

impl std::error::Error for TextEncoderError {}

impl From<super::tokenizer::TokenizerError> for TextEncoderError {
    fn from(err: super::tokenizer::TokenizerError) -> Self {
        Self::TokenizerError(err.to_string())
    }
}

impl From<TextEncoderError> for CandleBackendError {
    fn from(err: TextEncoderError) -> Self {
        CandleBackendError::InvalidRequest(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use reimagine_core::model::{ModelId, ModelRole};
    use reimagine_inference::{
        ModelFormat, ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let process = std::process::id();
        std::env::temp_dir().join(format!(
            "reimagine-text-encoder-{process}-{nonce}-{counter}"
        ))
    }

    fn test_bundle() -> Arc<LoadedSdxlBundle> {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        fs::write(&path, b"placeholder").unwrap();
        let source = ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            path,
            ModelFormat::SafeTensors,
        );
        let source_set = ResolvedInferenceModelSourceSet::new(source);
        LoadedSdxlBundle::from_resolved_with_source_set(
            ModelId::new("test-sdxl"),
            source_set,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .expect("test bundle")
    }

    #[test]
    fn text_encoder_produces_correct_shapes() {
        let bundle = test_bundle();
        let (text_emb, pooled_emb) = SdxlTextEncoder::encode(&bundle, "hello world").unwrap();
        assert_eq!(
            text_emb.shape().dims(),
            &[1, MAX_SEQUENCE_LENGTH, COMBINED_DIM]
        );
        assert_eq!(pooled_emb.shape().dims(), &[1, CLIP_G_DIM]);
    }

    #[test]
    fn text_encoder_produces_f32_tensors() {
        let bundle = test_bundle();
        let (text_emb, pooled_emb) = SdxlTextEncoder::encode(&bundle, "test").unwrap();
        assert_eq!(text_emb.dtype(), DType::F32);
        assert_eq!(pooled_emb.dtype(), DType::F32);
    }

    #[test]
    fn text_encoder_handles_empty_string() {
        let bundle = test_bundle();
        let (text_emb, pooled_emb) = SdxlTextEncoder::encode(&bundle, "").unwrap();
        assert_eq!(
            text_emb.shape().dims(),
            &[1, MAX_SEQUENCE_LENGTH, COMBINED_DIM]
        );
        assert_eq!(pooled_emb.shape().dims(), &[1, CLIP_G_DIM]);
    }
}
