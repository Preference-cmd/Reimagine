//! Diffusion UNet weight structs — plain data loaded from safetensors.
//!
//! Follows the same pattern as `text_conditioning/module.rs`: no
//! `#[derive(Module)]`, just `Vec<f32>` buffers indexed by key.

/// Weight data buffer — pre-allocated f32 values from safetensors.
#[derive(Debug, Clone)]
pub struct DiffusionWeightData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Complete set of SDXL UNet weights loaded from the diffusion
/// component safetensors file.
///
/// V1 captures the key tensor families needed for the euler/normal
/// sampling loop. The struct is deliberately flat; a full UNet module
/// graph is deferred to when Burn's `#[derive(Module)]` supports
/// `B: Backend` with Vec fields.
#[derive(Debug, Clone)]
pub struct DiffusionUNetWeights {
    pub conv_in_weight: DiffusionWeightData,
    pub conv_in_bias: DiffusionWeightData,
    pub time_embed_0_weight: DiffusionWeightData,
    pub time_embed_0_bias: DiffusionWeightData,
    pub time_embed_2_weight: DiffusionWeightData,
    pub time_embed_2_bias: DiffusionWeightData,
    // Input blocks (down-sampling)
    pub input_blocks: Vec<DiffusionBlockWeights>,
    // Middle block
    pub middle_block: Option<DiffusionBlockWeights>,
    // Output blocks (up-sampling)
    pub output_blocks: Vec<DiffusionBlockWeights>,
    pub out_0_weight: DiffusionWeightData,
    pub out_0_bias: DiffusionWeightData,
}

/// Weights for one diffusion block (input, middle, or output).
#[derive(Debug, Clone)]
pub struct DiffusionBlockWeights {
    pub conv_weight: DiffusionWeightData,
    pub conv_bias: DiffusionWeightData,
    // Optional attention/transformer weights
    pub attn_q_weight: Option<DiffusionWeightData>,
    pub attn_k_weight: Option<DiffusionWeightData>,
    pub attn_v_weight: Option<DiffusionWeightData>,
    pub attn_out_weight: Option<DiffusionWeightData>,
}
