//! Load SDXL diffusion UNet weights from Burn-native safetensors
//! component files using the bundle's component paths.

use std::fs;

use safetensors::tensor::SafeTensors;

use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::{BurnLoadedModelBundle, BurnSdxlComponentRole};

use super::module::{DiffusionBlockWeights, DiffusionUNetWeights, DiffusionWeightData};

/// Load diffusion UNet weights from the bundle's diffusion component
/// file. The bundle owns the resolved component path; this loader
/// reads the safetensors file and projects the keys into the weight
/// struct.
#[allow(dead_code)]
pub fn load_diffusion_weights(
    bundle: &BurnLoadedModelBundle,
) -> Result<DiffusionUNetWeights, BurnBackendError> {
    let sdxl = match bundle {
        BurnLoadedModelBundle::StableDiffusionSdxl(bundle) => bundle.as_ref(),
    };

    let component = sdxl
        .components()
        .iter()
        .find(|c| c.component_role == BurnSdxlComponentRole::Diffusion)
        .ok_or_else(|| BurnBackendError::MissingComponent("diffusion".to_owned()))?;

    let bytes = fs::read(&component.source_path).map_err(|e| BurnBackendError::ComponentRead {
        path: component.source_path.clone(),
        message: e.to_string(),
    })?;

    let safetensors =
        SafeTensors::deserialize(&bytes).map_err(|e| BurnBackendError::ComponentRead {
            path: component.source_path.clone(),
            message: e.to_string(),
        })?;

    // V1: build a minimal UNet weights struct with representative tensors
    // The full key-space projection is a follow-up deepening.
    let conv_in_weight = load_tensor(&safetensors, "model.diffusion.conv_in.weight")?;
    let conv_in_bias = load_tensor(&safetensors, "model.diffusion.conv_in.bias")?;
    let time_embed_0_weight = load_tensor(&safetensors, "model.diffusion.time_embed.0.weight")?;
    let time_embed_0_bias = load_tensor(&safetensors, "model.diffusion.time_embed.0.bias")?;
    let time_embed_2_weight = load_tensor(&safetensors, "model.diffusion.time_embed.2.weight")?;
    let time_embed_2_bias = load_tensor(&safetensors, "model.diffusion.time_embed.2.bias")?;

    // For V1, input/output blocks are loaded from known keys
    let mut input_blocks = Vec::new();
    for i in 0..12 {
        let prefix = format!("model.diffusion.input_blocks.{i}");
        if let Ok(w) = load_tensor_opt(&safetensors, &format!("{prefix}.0.weight")) {
            let b =
                load_tensor_opt(&safetensors, &format!("{prefix}.0.bias")).unwrap_or_else(|_| {
                    DiffusionWeightData {
                        data: vec![],
                        shape: vec![],
                    }
                });
            input_blocks.push(DiffusionBlockWeights {
                conv_weight: w,
                conv_bias: b,
                attn_q_weight: None,
                attn_k_weight: None,
                attn_v_weight: None,
                attn_out_weight: None,
            });
        }
    }

    let out_0_weight = load_tensor(&safetensors, "model.diffusion.out.0.weight")?;
    let out_0_bias = load_tensor(&safetensors, "model.diffusion.out.0.bias")?;

    Ok(DiffusionUNetWeights {
        conv_in_weight,
        conv_in_bias,
        time_embed_0_weight,
        time_embed_0_bias,
        time_embed_2_weight,
        time_embed_2_bias,
        input_blocks,
        middle_block: None,
        output_blocks: Vec::new(),
        out_0_weight,
        out_0_bias,
    })
}

#[allow(dead_code)]
fn load_tensor(
    safetensors: &SafeTensors,
    key: &str,
) -> Result<DiffusionWeightData, BurnBackendError> {
    let tensor = safetensors
        .tensor(key)
        .map_err(|_| BurnBackendError::ComponentRead {
            path: Default::default(),
            message: format!("missing diffusion tensor key `{key}`"),
        })?;
    let data = tensor.data().to_vec();
    let shape = tensor.shape().to_vec();
    // Convert bytes to f32
    let f32_data: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(DiffusionWeightData {
        data: f32_data,
        shape,
    })
}

#[allow(dead_code)]
fn load_tensor_opt(
    safetensors: &SafeTensors,
    key: &str,
) -> Result<DiffusionWeightData, BurnBackendError> {
    load_tensor(safetensors, key)
}
