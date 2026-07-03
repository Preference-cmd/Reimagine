//! Burn VAE module weight structs — plain data from safetensors.
//!
//! Follows the same pattern as `text_conditioning/module.rs` and
//! `diffusion/module.rs`.

/// Weight data buffer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VaeWeightData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// VAE decoder weights loaded from the VAE component safetensors.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SdxlVaeDecoderWeights {
    pub conv_in_weight: VaeWeightData,
    pub conv_in_bias: VaeWeightData,
    pub decoder_mid_block: Vec<VaeWeightData>,
    pub decoder_up_blocks: Vec<VaeBlockWeights>,
    pub conv_out_weight: VaeWeightData,
    pub conv_out_bias: VaeWeightData,
}

/// Weights for one VAE decoder up-block.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VaeBlockWeights {
    pub conv_weight: VaeWeightData,
    pub conv_bias: VaeWeightData,
}
