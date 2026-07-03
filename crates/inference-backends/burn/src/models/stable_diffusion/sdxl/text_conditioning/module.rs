//! Burn-native SDXL CLIP module structs.
//!
//! Plain data structs — no #[derive(Module)] since Burn 0.21's derive
//! requires `AutodiffBackend` for Vec<SubModule>. The tensors are
//! stored as `Vec<f32>` and converted to Burn tensors during forward
//! or can be wrapped at the call site. V1 focuses on the data
//! architecture; full Module trait integration is deferred to when
//! burn-core's derive supports plain `Backend` with Vec fields.

use burn_ndarray::NdArray;
use burn_core::tensor::Tensor;

/// Weight data loaded from safetensors — pre-allocated f32 buffers
/// that can be converted to Burn tensors on demand.
#[derive(Debug, Clone)]
pub struct ClipWeightData {
    /// Buffer of f32 values.
    pub data: Vec<f32>,
}

/// Single transformer block weights.
#[derive(Debug, Clone)]
pub struct ClipTransformerWeights {
    pub ln_1_weight: ClipWeightData,
    pub ln_1_bias: ClipWeightData,
    pub ln_2_weight: ClipWeightData,
    pub ln_2_bias: ClipWeightData,
    pub attn_in_proj_weight: ClipWeightData,
    pub attn_in_proj_bias: ClipWeightData,
    pub attn_out_proj_weight: ClipWeightData,
    pub attn_out_proj_bias: ClipWeightData,
    pub mlp_fc1_weight: ClipWeightData,
    pub mlp_fc1_bias: ClipWeightData,
    pub mlp_fc2_weight: ClipWeightData,
    pub mlp_fc2_bias: ClipWeightData,
}

/// Complete CLIP text encoder weights — the safetensors content
/// indexed by the ClipTextEncoderProfile key-space.
#[derive(Debug, Clone)]
pub struct ClipTextEncoderWeights {
    pub token_embedding: ClipWeightData,
    pub position_embedding: ClipWeightData,
    pub final_layer_norm_weight: ClipWeightData,
    pub final_layer_norm_bias: ClipWeightData,
    pub text_projection_weight: ClipWeightData,
    pub text_projection_bias: ClipWeightData,
    pub blocks: Vec<ClipTransformerWeights>,
}
