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
//! Tokenization is performed via the bundle's `SdxlTokenizer`; this
//! module owns the backend-private CLIP-L / CLIP-G Candle modules.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{D, DType, Device, Tensor};
use reimagine_inference::ResolvedInferenceModelSourceSet;

use super::LoadedSdxlBundle;
use crate::error::CandleBackendError;
use crate::models::stable_diffusion::sdxl::text_sources::{
    SdxlTextEncoderSources, resolve_text_encoder_sources,
};
use crate::models::stable_diffusion::sdxl::tokenizer::{MAX_SEQUENCE_LENGTH, SdxlTokenizedPrompt};

/// SDXL CLIP-L embedding dimension.
const CLIP_L_DIM: usize = 768;

/// SDXL CLIP-G embedding dimension.
const CLIP_G_DIM: usize = 1280;

/// Combined text embedding dimension (CLIP-L + CLIP-G).
const COMBINED_DIM: usize = CLIP_L_DIM + CLIP_G_DIM; // 2048

/// Backend-local SDXL text encoder graph.
///
/// The graph is owned by [`LoadedSdxlBundle`] and reused across
/// `text.encode` calls. Production loading validates that real
/// CLIP-L / CLIP-G weight prefixes are present before exposing the
/// graph; test-only placeholders use an explicit projection mode.
#[derive(Debug)]
pub struct SdxlTextEncoderGraph {
    device: Arc<Device>,
    mode: SdxlTextEncoderMode,
}

#[derive(Debug)]
enum SdxlTextEncoderMode {
    Real { encoder: Box<SdxlRealTextEncoder> },
    TokenProjection { sources: SdxlTextEncoderSources },
}

impl SdxlTextEncoderGraph {
    pub fn load(
        source_set: &ResolvedInferenceModelSourceSet,
        _primary_path: &Path,
        device: Arc<Device>,
    ) -> Result<Self, TextEncoderError> {
        let sources = resolve_text_encoder_sources(source_set)
            .map_err(|err| TextEncoderError::SourceResolution(err.to_string()))?;
        let (clip_l_source, clip_g_source) = resolve_clip_weight_sources(&sources)?;
        let sources = SdxlRealTextEncoderSources {
            clip_l: clip_l_source,
            clip_g: clip_g_source,
        };
        let encoder = Box::new(SdxlRealTextEncoder::load(&sources, device.as_ref())?);

        Ok(Self {
            device,
            mode: SdxlTextEncoderMode::Real { encoder },
        })
    }

    pub(crate) fn load_test_projection(
        source_set: &ResolvedInferenceModelSourceSet,
        device: Arc<Device>,
    ) -> Result<Self, TextEncoderError> {
        let sources = resolve_text_encoder_sources(source_set)
            .map_err(|err| TextEncoderError::SourceResolution(err.to_string()))?;
        Ok(Self {
            device,
            mode: SdxlTextEncoderMode::TokenProjection { sources },
        })
    }

    pub fn encode(
        &self,
        clip_l: &SdxlTokenizedPrompt,
        clip_g: &SdxlTokenizedPrompt,
    ) -> Result<(Tensor, Tensor), TextEncoderError> {
        match &self.mode {
            SdxlTextEncoderMode::Real { encoder } => self.encode_real(encoder, clip_l, clip_g),
            SdxlTextEncoderMode::TokenProjection { sources } => {
                let _source_fingerprint = sources.fingerprint();
                self.encode_token_projection(clip_l, clip_g)
            }
        }
    }

    fn encode_real(
        &self,
        encoder: &SdxlRealTextEncoder,
        clip_l: &SdxlTokenizedPrompt,
        clip_g: &SdxlTokenizedPrompt,
    ) -> Result<(Tensor, Tensor), TextEncoderError> {
        let clip_l_tokens = Tensor::from_vec(
            clip_l.token_ids.clone(),
            (1, MAX_SEQUENCE_LENGTH),
            self.device.as_ref(),
        )
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;
        let clip_g_tokens = Tensor::from_vec(
            clip_g.token_ids.clone(),
            (1, MAX_SEQUENCE_LENGTH),
            self.device.as_ref(),
        )
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;
        let clip_l_output = encoder.clip_l.forward(&clip_l_tokens)?;
        let clip_g_output = encoder.clip_g.forward(&clip_g_tokens)?;
        let text_embedding = Tensor::cat(
            &[clip_l_output.hidden, clip_g_output.hidden.clone()],
            D::Minus1,
        )
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?
        .to_dtype(DType::F32)
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;
        let pooled_embedding = pooled_from_final_hidden(&clip_g_output.final_hidden, clip_g)?
            .to_dtype(DType::F32)
            .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;

        Ok((text_embedding, pooled_embedding))
    }

    fn encode_token_projection(
        &self,
        clip_l: &SdxlTokenizedPrompt,
        clip_g: &SdxlTokenizedPrompt,
    ) -> Result<(Tensor, Tensor), TextEncoderError> {
        let clip_l_embedding = project_tokens(clip_l, CLIP_L_DIM, 0.013);
        let clip_g_embedding = project_tokens(clip_g, CLIP_G_DIM, 0.017);
        let mut text_embedding = Vec::with_capacity(MAX_SEQUENCE_LENGTH * COMBINED_DIM);
        for position in 0..MAX_SEQUENCE_LENGTH {
            let l_start = position * CLIP_L_DIM;
            let g_start = position * CLIP_G_DIM;
            text_embedding.extend_from_slice(&clip_l_embedding[l_start..l_start + CLIP_L_DIM]);
            text_embedding.extend_from_slice(&clip_g_embedding[g_start..g_start + CLIP_G_DIM]);
        }

        let pooled_embedding = pooled_from_token_embeddings(&clip_g_embedding, clip_g);

        let text_embedding = Tensor::from_vec(
            text_embedding,
            (1, MAX_SEQUENCE_LENGTH, COMBINED_DIM),
            self.device.as_ref(),
        )
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;
        let pooled_embedding =
            Tensor::from_vec(pooled_embedding, (1, CLIP_G_DIM), self.device.as_ref())
                .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;

        Ok((text_embedding, pooled_embedding))
    }
}

/// Compatibility namespace for the graph facade.
pub struct SdxlTextEncoder;

impl SdxlTextEncoder {
    /// Encode text into conditioning tensors.
    ///
    /// Tokenizes `text` using the bundle's two SDXL tokenizers and
    /// delegates to the bundle-owned text encoder graph.
    ///
    /// Returns `(text_embedding, pooled_embedding)` tensors. The text
    /// embedding has shape `[1, MAX_SEQUENCE_LENGTH, COMBINED_DIM]`
    /// and the pooled embedding has shape `[1, CLIP_G_DIM]`.
    pub fn encode(
        bundle: &LoadedSdxlBundle,
        text: &str,
    ) -> Result<(Tensor, Tensor), TextEncoderError> {
        let prompt = bundle.tokenizer.tokenize_pair(text)?;
        bundle.text_encoder.encode(&prompt.clip_l, &prompt.clip_g)
    }
}

#[derive(Debug, Clone)]
pub enum TextEncoderError {
    TensorCreation(String),
    TokenizerError(String),
    SourceResolution(String),
    WeightLoad(String),
    Forward(String),
}

impl std::fmt::Display for TextEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorCreation(msg) => write!(f, "text encoder tensor creation failed: {msg}"),
            Self::TokenizerError(msg) => write!(f, "text encoder tokenizer error: {msg}"),
            Self::SourceResolution(msg) => {
                write!(f, "text encoder source resolution failed: {msg}")
            }
            Self::WeightLoad(msg) => write!(f, "SDXL text encoder weights unavailable: {msg}"),
            Self::Forward(msg) => write!(f, "SDXL text encoder forward failed: {msg}"),
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClipWeightSource {
    path: PathBuf,
    prefix: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SdxlRealTextEncoderSources {
    clip_l: ClipWeightSource,
    clip_g: ClipWeightSource,
}

#[derive(Debug)]
struct SdxlRealTextEncoder {
    clip_l: ClipTextTransformer,
    clip_g: ClipTextTransformer,
}

impl SdxlRealTextEncoder {
    fn load(
        sources: &SdxlRealTextEncoderSources,
        device: &Device,
    ) -> Result<Self, TextEncoderError> {
        Ok(Self {
            clip_l: ClipTextTransformer::load(&sources.clip_l, ClipTextConfig::sdxl_l(), device)?,
            clip_g: ClipTextTransformer::load(&sources.clip_g, ClipTextConfig::sdxl_g(), device)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ClipTextConfig {
    vocab_size: usize,
    embed_dim: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    activation: ClipActivation,
    hidden_output_layer: usize,
}

impl ClipTextConfig {
    fn sdxl_l() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: CLIP_L_DIM,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            activation: ClipActivation::QuickGelu,
            hidden_output_layer: 10,
        }
    }

    fn sdxl_g() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: CLIP_G_DIM,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            activation: ClipActivation::Gelu,
            hidden_output_layer: 30,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ClipActivation {
    QuickGelu,
    Gelu,
}

#[derive(Debug)]
struct ClipTextTransformer {
    token_embedding: Tensor,
    position_embedding: Tensor,
    layers: Vec<ClipEncoderLayer>,
    final_layer_norm: ClipLayerNorm,
    config: ClipTextConfig,
}

#[derive(Debug)]
struct ClipTextOutput {
    hidden: Tensor,
    final_hidden: Tensor,
}

impl ClipTextTransformer {
    fn load(
        source: &ClipWeightSource,
        config: ClipTextConfig,
        device: &Device,
    ) -> Result<Self, TextEncoderError> {
        let weights = ClipWeightLoader::load(source, device)?;
        let token_embedding = weights.tensor("embeddings.token_embedding.weight")?;
        let position_embedding = weights.tensor("embeddings.position_embedding.weight")?;
        assert_shape(
            &token_embedding,
            &[config.vocab_size, config.embed_dim],
            "token embedding",
        )?;
        assert_shape(
            &position_embedding,
            &[MAX_SEQUENCE_LENGTH, config.embed_dim],
            "position embedding",
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(ClipEncoderLayer::load(&weights, layer_index, config)?);
        }
        let final_layer_norm = ClipLayerNorm::load(&weights, "final_layer_norm", config.embed_dim)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
            config,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<ClipTextOutput, TextEncoderError> {
        let (batch, seq_len) = input_ids
            .dims2()
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let embed_dim = self.config.embed_dim;
        let flat_input_ids = input_ids
            .flatten_all()
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let token_embedding_flat = self
            .token_embedding
            .index_select(&flat_input_ids, 0)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let token_embedding = token_embedding_flat
            .reshape((batch, seq_len, embed_dim))
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())
            .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;
        let flat_position_ids = position_ids
            .flatten_all()
            .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))?;
        let position_embedding_flat = self
            .position_embedding
            .index_select(&flat_position_ids, 0)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let position_embedding = position_embedding_flat
            .reshape((1, seq_len, embed_dim))
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?
            .broadcast_as((batch, seq_len, embed_dim))
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let mut hidden = token_embedding
            .broadcast_add(&position_embedding)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let mask = causal_attention_mask(batch, seq_len, input_ids.device())?;
        let mut selected_hidden = None;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &mask)?;
            if layer_index == self.config.hidden_output_layer {
                selected_hidden = Some(hidden.clone());
            }
        }
        let selected_hidden = selected_hidden.unwrap_or_else(|| hidden.clone());
        let final_hidden = self.final_layer_norm.forward(&hidden)?;
        Ok(ClipTextOutput {
            hidden: selected_hidden,
            final_hidden,
        })
    }
}

#[derive(Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: ClipLayerNorm,
    fc1: ClipLinear,
    fc2: ClipLinear,
    layer_norm2: ClipLayerNorm,
    activation: ClipActivation,
}

impl ClipEncoderLayer {
    fn load(
        weights: &ClipWeightLoader,
        layer_index: usize,
        config: ClipTextConfig,
    ) -> Result<Self, TextEncoderError> {
        let prefix = format!("encoder.layers.{layer_index}");
        Ok(Self {
            self_attn: ClipAttention::load(weights, &format!("{prefix}.self_attn"), config)?,
            layer_norm1: ClipLayerNorm::load(
                weights,
                &format!("{prefix}.layer_norm1"),
                config.embed_dim,
            )?,
            fc1: ClipLinear::load(
                weights,
                &format!("{prefix}.mlp.fc1"),
                config.embed_dim,
                config.intermediate_size,
            )?,
            fc2: ClipLinear::load(
                weights,
                &format!("{prefix}.mlp.fc2"),
                config.intermediate_size,
                config.embed_dim,
            )?,
            layer_norm2: ClipLayerNorm::load(
                weights,
                &format!("{prefix}.layer_norm2"),
                config.embed_dim,
            )?,
            activation: config.activation,
        })
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor, TextEncoderError> {
        let residual = hidden;
        let normed = self.layer_norm1.forward(hidden)?;
        let attended = self.self_attn.forward(&normed, mask)?;
        let hidden = attended
            .broadcast_add(residual)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let residual = &hidden;
        let normed = self.layer_norm2.forward(&hidden)?;
        let mlp = self.fc1.forward(&normed)?;
        let mlp = apply_activation(&mlp, self.activation)?;
        let mlp = self.fc2.forward(&mlp)?;
        mlp.broadcast_add(residual)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))
    }
}

#[derive(Debug)]
struct ClipAttention {
    q_proj: ClipLinear,
    k_proj: ClipLinear,
    v_proj: ClipLinear,
    out_proj: ClipLinear,
    num_attention_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl ClipAttention {
    fn load(
        weights: &ClipWeightLoader,
        prefix: &str,
        config: ClipTextConfig,
    ) -> Result<Self, TextEncoderError> {
        let head_dim = config.embed_dim / config.num_attention_heads;
        Ok(Self {
            q_proj: ClipLinear::load(
                weights,
                &format!("{prefix}.q_proj"),
                config.embed_dim,
                config.embed_dim,
            )?,
            k_proj: ClipLinear::load(
                weights,
                &format!("{prefix}.k_proj"),
                config.embed_dim,
                config.embed_dim,
            )?,
            v_proj: ClipLinear::load(
                weights,
                &format!("{prefix}.v_proj"),
                config.embed_dim,
                config.embed_dim,
            )?,
            out_proj: ClipLinear::load(
                weights,
                &format!("{prefix}.out_proj"),
                config.embed_dim,
                config.embed_dim,
            )?,
            num_attention_heads: config.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor, TextEncoderError> {
        let in_dtype = hidden.dtype();
        let (batch, seq_len, embed_dim) = hidden
            .dims3()
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let query = self.shape_heads(&self.q_proj.forward(hidden)?, batch, seq_len)?;
        let key = self.shape_heads(&self.k_proj.forward(hidden)?, batch, seq_len)?;
        let value = self.shape_heads(&self.v_proj.forward(hidden)?, batch, seq_len)?;
        let query = (&query * self.scale)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?
            .to_dtype(DType::F32)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let key = key
            .to_dtype(DType::F32)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let value = value
            .to_dtype(DType::F32)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let attention = query
            .matmul(
                &key.transpose(1, 2)
                    .map_err(|e| TextEncoderError::Forward(e.to_string()))?,
            )
            .and_then(|scores| scores.reshape((batch, self.num_attention_heads, seq_len, seq_len)))
            .and_then(|scores| scores.broadcast_add(mask))
            .and_then(|scores| scores.reshape((batch * self.num_attention_heads, seq_len, seq_len)))
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let attention = softmax(&attention, D::Minus1)?;
        let output = attention
            .matmul(&value)
            .and_then(|output| {
                output
                    .reshape((batch, self.num_attention_heads, seq_len, self.head_dim))
                    .and_then(|output| output.transpose(1, 2))
                    .and_then(|output| output.reshape((batch, seq_len, embed_dim)))
            })
            .and_then(|output| output.to_dtype(in_dtype))
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        self.out_proj.forward(&output)
    }

    fn shape_heads(
        &self,
        tensor: &Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Result<Tensor, TextEncoderError> {
        tensor
            .reshape((batch, seq_len, self.num_attention_heads, self.head_dim))
            .and_then(|tensor| tensor.transpose(1, 2))
            .and_then(|tensor| {
                tensor.reshape((batch * self.num_attention_heads, seq_len, self.head_dim))
            })
            .map_err(|e| TextEncoderError::Forward(e.to_string()))
    }
}

#[derive(Debug)]
struct ClipLinear {
    weight: Tensor,
    bias: Tensor,
}

impl ClipLinear {
    fn load(
        weights: &ClipWeightLoader,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, TextEncoderError> {
        let weight_raw = weights.tensor(&format!("{prefix}.weight"))?;
        let bias = weights.tensor(&format!("{prefix}.bias"))?;
        // Candle's `matmul` requires both operands to share the same
        // number of dimensions. Loading CLIP weight tensors in
        // [in_dim, out_dim] layout (transposed at load time) lets the
        // forward path matmul a 3D hidden tensor against a 2D weight
        // directly without broadcasting.
        let weight = weight_raw
            .transpose(0, 1)
            .map_err(|e| TextEncoderError::WeightLoad(format!("{prefix}.weight transpose: {e}")))?;
        assert_shape(&weight, &[in_dim, out_dim], prefix)?;
        assert_shape(&bias, &[out_dim], prefix)?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, TextEncoderError> {
        input
            .broadcast_matmul(&self.weight)
            .and_then(|output| output.broadcast_add(&self.bias))
            .map_err(|e| TextEncoderError::Forward(e.to_string()))
    }
}

#[derive(Debug)]
struct ClipLayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl ClipLayerNorm {
    fn load(
        weights: &ClipWeightLoader,
        prefix: &str,
        dim: usize,
    ) -> Result<Self, TextEncoderError> {
        let weight = weights.tensor(&format!("{prefix}.weight"))?;
        let bias = weights.tensor(&format!("{prefix}.bias"))?;
        assert_shape(&weight, &[dim], prefix)?;
        assert_shape(&bias, &[dim], prefix)?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, TextEncoderError> {
        let mean = input
            .mean_keepdim(D::Minus1)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let centered = input
            .broadcast_sub(&mean)
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let variance = centered
            .sqr()
            .and_then(|x| x.mean_keepdim(D::Minus1))
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        let denom = (variance + 1e-5f64)
            .and_then(|x| x.sqrt())
            .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
        centered
            .broadcast_div(&denom)
            .and_then(|x| x.broadcast_mul(&self.weight))
            .and_then(|x| x.broadcast_add(&self.bias))
            .map_err(|e| TextEncoderError::Forward(e.to_string()))
    }
}

fn softmax(xs: &Tensor, dim: D) -> Result<Tensor, TextEncoderError> {
    let max = xs
        .max_keepdim(dim)
        .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
    let shifted = xs
        .broadcast_sub(&max)
        .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
    let exp = shifted
        .exp()
        .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
    let sum = exp
        .sum_keepdim(dim)
        .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
    exp.broadcast_div(&sum)
        .map_err(|e| TextEncoderError::Forward(e.to_string()))
}

fn causal_attention_mask(
    batch: usize,
    seq_len: usize,
    device: &Device,
) -> Result<Tensor, TextEncoderError> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for row in 0..seq_len {
        for col in 0..seq_len {
            values.push(if col > row { -f32::INFINITY } else { 0.0 });
        }
    }
    Tensor::from_vec(values, (seq_len, seq_len), device)
        .and_then(|mask| mask.reshape((1, 1, seq_len, seq_len)))
        .and_then(|mask| mask.broadcast_as((batch, 1, seq_len, seq_len)))
        .map_err(|e| TextEncoderError::TensorCreation(e.to_string()))
}

fn apply_activation(xs: &Tensor, activation: ClipActivation) -> Result<Tensor, TextEncoderError> {
    match activation {
        ClipActivation::QuickGelu => {
            let sigmoid_denominator = xs
                .affine(-1.702, 0.0)
                .and_then(|x| x.exp())
                .and_then(|x| x + 1.0f64)
                .map_err(|e| TextEncoderError::Forward(e.to_string()))?;
            xs.broadcast_div(&sigmoid_denominator)
                .map_err(|e| TextEncoderError::Forward(e.to_string()))
        }
        ClipActivation::Gelu => xs
            .gelu()
            .map_err(|e| TextEncoderError::Forward(e.to_string())),
    }
}

fn assert_shape(tensor: &Tensor, expected: &[usize], label: &str) -> Result<(), TextEncoderError> {
    if tensor.shape().dims() == expected {
        Ok(())
    } else {
        Err(TextEncoderError::WeightLoad(format!(
            "{label} shape mismatch: expected {:?}, got {:?}",
            expected,
            tensor.shape().dims()
        )))
    }
}

struct ClipWeightLoader<'a> {
    source: &'a ClipWeightSource,
    buffer: Vec<u8>,
    device: &'a Device,
}

impl<'a> ClipWeightLoader<'a> {
    fn load(source: &'a ClipWeightSource, device: &'a Device) -> Result<Self, TextEncoderError> {
        let buffer = std::fs::read(&source.path).map_err(|e| {
            TextEncoderError::WeightLoad(format!(
                "failed to read text encoder weights from {}: {e}",
                source.path.display()
            ))
        })?;
        Ok(Self {
            source,
            buffer,
            device,
        })
    }

    fn tensor(&self, suffix: &str) -> Result<Tensor, TextEncoderError> {
        let name = format!("{}.{}", self.source.prefix, suffix);
        let safetensors =
            candle_core::safetensors::SliceSafetensors::new(&self.buffer).map_err(|e| {
                TextEncoderError::WeightLoad(format!(
                    "failed to parse text encoder safetensors header from {}: {e}",
                    self.source.path.display()
                ))
            })?;
        safetensors.load(&name, self.device).map_err(|e| {
            TextEncoderError::WeightLoad(format!(
                "failed to load tensor `{name}` from {}: {e}",
                self.source.path.display()
            ))
        })
    }
}

fn resolve_clip_weight_sources(
    sources: &SdxlTextEncoderSources,
) -> Result<(ClipWeightSource, ClipWeightSource), TextEncoderError> {
    match sources {
        SdxlTextEncoderSources::Split { clip_l, clip_g } => Ok((
            detect_clip_weight_source(clip_l, &CLIP_L_PREFIXES, "clip_l")?,
            detect_clip_weight_source(clip_g, &CLIP_G_PREFIXES, "clip_g")?,
        )),
        SdxlTextEncoderSources::Combined { path } | SdxlTextEncoderSources::Checkpoint { path } => {
            Ok((
                detect_clip_weight_source(path, &CLIP_L_PREFIXES, "clip_l")?,
                detect_clip_weight_source(path, &CLIP_G_PREFIXES, "clip_g")?,
            ))
        }
    }
}

const CLIP_L_PREFIXES: [&str; 5] = [
    "conditioner.embedders.0.transformer.text_model",
    "cond_stage_model.transformer.text_model",
    "text_encoder.text_model",
    "transformer.text_model",
    "clip_l.text_model",
];

const CLIP_G_PREFIXES: [&str; 6] = [
    "conditioner.embedders.1.model.transformer.text_model",
    "conditioner.embedders.1.transformer.text_model",
    "text_encoder_2.text_model",
    "transformer.text_model",
    "clip_g.text_model",
    "text_model",
];

fn detect_clip_weight_source(
    path: &Path,
    prefixes: &[&'static str],
    component: &'static str,
) -> Result<ClipWeightSource, TextEncoderError> {
    let bytes = std::fs::read(path).map_err(|e| {
        TextEncoderError::WeightLoad(format!(
            "failed to read {component} text encoder weights from {}: {e}",
            path.display()
        ))
    })?;
    let safetensors = candle_core::safetensors::SliceSafetensors::new(&bytes).map_err(|e| {
        TextEncoderError::WeightLoad(format!(
            "failed to parse {component} text encoder safetensors header from {}: {e}",
            path.display()
        ))
    })?;
    let names: Vec<String> = safetensors
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    detect_prefix_from_names(&names, prefixes, component).map(|prefix| ClipWeightSource {
        path: path.to_path_buf(),
        prefix,
    })
}

fn detect_prefix_from_names(
    names: &[String],
    prefixes: &[&'static str],
    component: &'static str,
) -> Result<&'static str, TextEncoderError> {
    let matches: Vec<&'static str> = prefixes
        .iter()
        .copied()
        .filter(|prefix| has_required_clip_tensors(names, prefix))
        .collect();
    match matches.as_slice() {
        [prefix] => Ok(*prefix),
        [] => Err(TextEncoderError::WeightLoad(format!(
            "missing {component} text encoder weights; no supported key prefix found"
        ))),
        _ => Err(TextEncoderError::WeightLoad(format!(
            "ambiguous {component} text encoder weights; matched prefixes: {}",
            matches.join(", ")
        ))),
    }
}

fn has_required_clip_tensors(names: &[String], prefix: &str) -> bool {
    let required_suffixes = [
        "embeddings.token_embedding.weight",
        "embeddings.position_embedding.weight",
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.self_attn.k_proj.weight",
        "encoder.layers.0.self_attn.v_proj.weight",
        "encoder.layers.0.self_attn.out_proj.weight",
        "encoder.layers.0.layer_norm1.weight",
        "final_layer_norm.weight",
    ];
    required_suffixes.iter().all(|suffix| {
        names
            .iter()
            .any(|name| name == &format!("{prefix}.{suffix}"))
    })
}

fn project_tokens(prompt: &SdxlTokenizedPrompt, width: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(MAX_SEQUENCE_LENGTH * width);
    for (position, (&token_id, &mask)) in prompt
        .token_ids
        .iter()
        .zip(prompt.attention_mask.iter())
        .enumerate()
    {
        let token = token_id as f32;
        let pos = (position + 1) as f32;
        for channel in 0..width {
            let chan = (channel + 1) as f32;
            let phase = token * scale + pos * 0.071 + chan * 0.0031;
            let value = if mask > 0.0 {
                phase.sin() * 0.5 + phase.cos() * 0.25
            } else {
                0.0
            };
            out.push(value);
        }
    }
    out
}

fn pooled_from_token_embeddings(
    token_embeddings: &[f32],
    prompt: &SdxlTokenizedPrompt,
) -> Vec<f32> {
    let attended = prompt
        .attention_mask
        .iter()
        .filter(|&&v| v > 0.0)
        .count()
        .max(1) as f32;
    let mut pooled = vec![0.0; CLIP_G_DIM];
    for position in 0..MAX_SEQUENCE_LENGTH {
        if prompt.attention_mask[position] <= 0.0 {
            continue;
        }
        let start = position * CLIP_G_DIM;
        for channel in 0..CLIP_G_DIM {
            pooled[channel] += token_embeddings[start + channel] / attended;
        }
    }
    pooled
}

fn pooled_from_final_hidden(
    final_hidden: &Tensor,
    prompt: &SdxlTokenizedPrompt,
) -> Result<Tensor, TextEncoderError> {
    let eos_position = prompt
        .token_ids
        .iter()
        .enumerate()
        .filter(|(_, token)| **token != super::tokenizer::TOKEN_PAD)
        .max_by_key(|(_, token)| *token)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    final_hidden
        .get(0)
        .and_then(|hidden| hidden.get(eos_position))
        .and_then(|hidden| hidden.unsqueeze(0))
        .map_err(|e| TextEncoderError::Forward(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
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
        LoadedSdxlBundle::from_resolved_with_test_text_projection(
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

    #[test]
    fn text_encoder_outputs_depend_on_prompt() {
        let bundle = test_bundle();
        let (first_text, first_pooled) = SdxlTextEncoder::encode(&bundle, "sunrise lake").unwrap();
        let (second_text, second_pooled) =
            SdxlTextEncoder::encode(&bundle, "midnight city").unwrap();
        let first_values = first_text.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let second_values = second_text.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let first_pooled_values = first_pooled
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert!(
            first_values.iter().any(|value| *value != 0.0),
            "text encoder must not return all-zero placeholder text embeddings"
        );
        assert!(
            first_pooled_values.iter().any(|value| *value != 0.0),
            "text encoder must not return all-zero placeholder pooled embeddings"
        );
        assert_ne!(first_values, second_values);
        assert_ne!(
            first_pooled
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            second_pooled
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }
}
