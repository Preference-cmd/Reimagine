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
//! V1 uses placeholder tensors with correct shapes. The actual CLIP
//! inference will be implemented when candle CLIP weights are integrated.

use candle_core::{DType, Device, Tensor};
use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_core::{BackendKind, BackendPayloadKey, BackendTensorHandle};

use crate::error::CandleBackendError;
use crate::models::stable_diffusion::sdxl::tokenizer::{MAX_SEQUENCE_LENGTH, SdxlTokenizer};
use crate::store::{CandleConditioning, CandleStore};

/// SDXL CLIP-L embedding dimension.
const CLIP_L_DIM: usize = 768;

/// SDXL CLIP-G embedding dimension.
const CLIP_G_DIM: usize = 1280;

/// Combined text embedding dimension (CLIP-L + CLIP-G).
const COMBINED_DIM: usize = CLIP_L_DIM + CLIP_G_DIM; // 2048

/// SDXL text encoder that produces conditioning tensors.
///
/// The encoder owns a tokenizer and dispatches text encoding for the
/// SDXL model family. It produces both text embeddings and pooled
/// embeddings as required by SDXL's UNet architecture.
pub struct SdxlTextEncoder {
    tokenizer: SdxlTokenizer,
}

impl SdxlTextEncoder {
    pub fn new() -> Self {
        Self {
            tokenizer: SdxlTokenizer::new(),
        }
    }

    /// Encode text into conditioning tensors.
    ///
    /// Returns `(text_embedding, pooled_embedding)` tensors. The text
    /// embedding has shape `[1, MAX_SEQUENCE_LENGTH, COMBINED_DIM]`
    /// and the pooled embedding has shape `[1, CLIP_G_DIM]`.
    ///
    /// V1 produces zero-valued tensors with correct shapes. The actual
    /// CLIP inference will be implemented when candle CLIP weights are
    /// integrated.
    pub fn encode(
        &self,
        text: &str,
        device: &Device,
    ) -> Result<(Tensor, Tensor), TextEncoderError> {
        let _tokens = self.tokenizer.encode(text, device)?;
        let _attention_mask = self.tokenizer.attention_mask(text, device)?;

        // Produce text embedding with correct SDXL shape
        // Shape: [1, 77, 2048] (batch, seq_len, combined_dim)
        let text_embedding =
            Tensor::zeros((1, MAX_SEQUENCE_LENGTH, COMBINED_DIM), DType::F32, device)
                .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;

        // Produce pooled embedding with correct SDXL shape
        // Shape: [1, 1280] (batch, clip_g_dim)
        let pooled_embedding = Tensor::zeros((1, CLIP_G_DIM), DType::F32, device)
            .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;

        Ok((text_embedding, pooled_embedding))
    }

    /// Encode text and store conditioning payload in the backend store.
    ///
    /// Returns the payload key for the stored conditioning.
    pub fn encode_and_store(
        &self,
        text: &str,
        device: &Device,
        store: &CandleStore,
        run_id: &reimagine_core::model::RunId,
        node_id: &reimagine_core::model::NodeId,
        _backend_kind: &str,
        _device_label: &str,
    ) -> Result<(BackendPayloadKey, Tensor, Tensor), TextEncoderError> {
        let (text_emb, pooled_emb) = self.encode(text, device)?;

        let payload_key = BackendPayloadKey::new(format!(
            "conditioning:{}:{}",
            run_id.as_str(),
            node_id.as_str()
        ));

        let conditioning = CandleConditioning::new(text_emb.clone(), Some(pooled_emb.clone()));
        store.insert_conditioning(run_id.clone(), payload_key.clone(), conditioning);

        Ok((payload_key, text_emb, pooled_emb))
    }
}

impl Default for SdxlTextEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a `ExecutionValue::Conditioning` from stored tensors.
///
/// This helper constructs the lightweight `BackendTensorHandle` values
/// that cross the backend boundary. The actual tensors remain in the
/// store; only the handles are returned to runtime.
pub fn build_conditioning_runtime_value(
    payload_key: BackendPayloadKey,
    text_embedding_shape: Vec<usize>,
    pooled_embedding_shape: Vec<usize>,
    backend_kind: &str,
    device_label: &str,
) -> reimagine_core::ExecutionValue {
    let text_handle = BackendTensorHandle::new(
        BackendKind::from(backend_kind),
        payload_key.clone(),
        TensorDType::F32,
        TensorShape::new(text_embedding_shape),
        device_label,
    );

    let pooled_handle = BackendTensorHandle::new(
        BackendKind::from(backend_kind),
        payload_key,
        TensorDType::F32,
        TensorShape::new(pooled_embedding_shape),
        device_label,
    );

    let metadata = reimagine_core::ConditioningMetadata::new(0, 0);

    reimagine_core::ExecutionValue::Conditioning(
        reimagine_core::ExecutionConditioning::new(text_handle, metadata)
            .with_pooled_embedding(pooled_handle),
    )
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

    #[test]
    fn text_encoder_produces_correct_shapes() {
        let encoder = SdxlTextEncoder::new();
        let device = Device::Cpu;
        let (text_emb, pooled_emb) = encoder.encode("hello world", &device).unwrap();
        assert_eq!(
            text_emb.shape().dims(),
            &[1, MAX_SEQUENCE_LENGTH, COMBINED_DIM]
        );
        assert_eq!(pooled_emb.shape().dims(), &[1, CLIP_G_DIM]);
    }

    #[test]
    fn text_encoder_produces_f32_tensors() {
        let encoder = SdxlTextEncoder::new();
        let device = Device::Cpu;
        let (text_emb, pooled_emb) = encoder.encode("test", &device).unwrap();
        assert_eq!(text_emb.dtype(), DType::F32);
        assert_eq!(pooled_emb.dtype(), DType::F32);
    }

    #[test]
    fn text_encoder_handles_empty_string() {
        let encoder = SdxlTextEncoder::new();
        let device = Device::Cpu;
        let (text_emb, pooled_emb) = encoder.encode("", &device).unwrap();
        assert_eq!(
            text_emb.shape().dims(),
            &[1, MAX_SEQUENCE_LENGTH, COMBINED_DIM]
        );
        assert_eq!(pooled_emb.shape().dims(), &[1, CLIP_G_DIM]);
    }

    #[test]
    fn text_encoder_encode_and_store_returns_payload_key() {
        let encoder = SdxlTextEncoder::new();
        let device = Device::Cpu;
        let store = CandleStore::new();
        let run_id = reimagine_core::model::RunId::new("run-test");
        let node_id = reimagine_core::model::NodeId::new("node-test");

        let (key, _, _) = encoder
            .encode_and_store(
                "test prompt",
                &device,
                &store,
                &run_id,
                &node_id,
                "candle",
                "cpu",
            )
            .unwrap();

        assert!(key.as_str().contains("conditioning"));
        assert!(key.as_str().contains("run-test"));
        assert!(key.as_str().contains("node-test"));
        assert_eq!(store.payload_count(), 1);
    }

    #[test]
    fn build_conditioning_runtime_value_has_correct_kind() {
        let key = BackendPayloadKey::new("test:conditioning");
        let value = build_conditioning_runtime_value(
            key,
            vec![1, MAX_SEQUENCE_LENGTH, COMBINED_DIM],
            vec![1, CLIP_G_DIM],
            "candle",
            "cpu",
        );
        assert!(matches!(
            value,
            reimagine_core::ExecutionValue::Conditioning(_)
        ));
    }
}
