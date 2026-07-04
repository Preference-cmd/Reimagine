//! CLIP text encoder weight loading from safetensors component files.
//!
//! Provides [`load_clip_l`] and [`load_clip_g`] that read a
//! [`BurnLoadedSdxlBundle`]'s component safetensors and produce a
//! [`ClipTextEncoderWeights`] struct ready for the forward pass.

use std::fs;
use std::path::Path;

use safetensors::tensor::SafeTensors;

use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::loaded::BurnLoadedSdxlBundle;
use crate::text_encoder::clip::ClipTextEncoderProfile;
use crate::text_encoder::keyspace::TextEncoderKeyspace;

use super::module::{ClipTextEncoderWeights, ClipTransformerWeights, ClipWeightData};

/// Load CLIP-L (primary text encoder) weights from the bundle.
pub fn load_clip_l(
    bundle: &BurnLoadedSdxlBundle,
) -> Result<ClipTextEncoderWeights, BurnBackendError> {
    let profile = ClipTextEncoderProfile::sdxl_clip_l();
    let (primary_path, _secondary_path) = bundle.text_encoder_component_paths()?;
    load_from_path(&primary_path, &profile)
}

/// Load CLIP-G (secondary text encoder / OpenCLIP-G) weights from the
/// bundle.
pub fn load_clip_g(
    bundle: &BurnLoadedSdxlBundle,
) -> Result<ClipTextEncoderWeights, BurnBackendError> {
    let profile = ClipTextEncoderProfile::sdxl_open_clip_g();
    let (_primary_path, secondary_path) = bundle.text_encoder_component_paths()?;
    load_from_path(&secondary_path, &profile)
}

/// Read a safetensors file and extract all weights matching the given
/// profile's key-space.
fn load_from_path(
    path: &Path,
    profile: &ClipTextEncoderProfile,
) -> Result<ClipTextEncoderWeights, BurnBackendError> {
    let bytes = fs::read(path).map_err(|source| BurnBackendError::ComponentRead {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;
    let safetensors =
        SafeTensors::deserialize(&bytes).map_err(|source| BurnBackendError::ComponentRead {
            path: path.to_path_buf(),
            message: source.to_string(),
        })?;

    let keys = TextEncoderKeyspace::new(profile);

    // Non-block embeddings
    let token_embedding = read_weight(&safetensors, &keys.token_embedding(), path)?;
    let position_embedding = read_weight(&safetensors, &keys.position_embedding(), path)?;
    let final_layer_norm_weight = read_weight(&safetensors, &keys.final_layer_norm_weight(), path)?;
    let final_layer_norm_bias = read_weight(&safetensors, &keys.final_layer_norm_bias(), path)?;

    // Optional text projection
    let text_projection_weight = if let Some(key) = keys.text_projection_weight() {
        read_weight(&safetensors, &key, path)?
    } else {
        ClipWeightData { data: Vec::new() }
    };
    let text_projection_bias = if let Some(key) = keys.text_projection_bias() {
        read_weight(&safetensors, &key, path)?
    } else {
        ClipWeightData { data: Vec::new() }
    };

    // Per-block weights
    let mut blocks = Vec::with_capacity(profile.num_layers as usize);
    for layer in 0..profile.num_layers {
        blocks.push(ClipTransformerWeights {
            ln_1_weight: read_weight(&safetensors, &keys.ln_1_weight(layer), path)?,
            ln_1_bias: read_weight(&safetensors, &keys.ln_1_bias(layer), path)?,
            ln_2_weight: read_weight(&safetensors, &keys.ln_2_weight(layer), path)?,
            ln_2_bias: read_weight(&safetensors, &keys.ln_2_bias(layer), path)?,
            attn_in_proj_weight: read_weight(&safetensors, &keys.attn_in_proj_weight(layer), path)?,
            attn_in_proj_bias: read_weight(&safetensors, &keys.attn_in_proj_bias(layer), path)?,
            attn_out_proj_weight: read_weight(
                &safetensors,
                &keys.attn_out_proj_weight(layer),
                path,
            )?,
            attn_out_proj_bias: read_weight(&safetensors, &keys.attn_out_proj_bias(layer), path)?,
            mlp_fc1_weight: read_weight(&safetensors, &keys.mlp_fc1_weight(layer), path)?,
            mlp_fc1_bias: read_weight(&safetensors, &keys.mlp_fc1_bias(layer), path)?,
            mlp_fc2_weight: read_weight(&safetensors, &keys.mlp_fc2_weight(layer), path)?,
            mlp_fc2_bias: read_weight(&safetensors, &keys.mlp_fc2_bias(layer), path)?,
        });
    }

    Ok(ClipTextEncoderWeights {
        token_embedding,
        position_embedding,
        final_layer_norm_weight,
        final_layer_norm_bias,
        text_projection_weight,
        text_projection_bias,
        blocks,
    })
}

/// Read a named tensor from safetensors as f32 data.
fn read_weight(
    safetensors: &SafeTensors<'_>,
    key: &str,
    path: &Path,
) -> Result<ClipWeightData, BurnBackendError> {
    let tensor = safetensors
        .tensor(key)
        .map_err(|_| BurnBackendError::ComponentRead {
            path: path.to_path_buf(),
            message: format!("tensor `{key}` not found in safetensors"),
        })?;

    let data = match tensor.dtype() {
        safetensors::tensor::Dtype::F32 => {
            let raw = tensor.data();
            // safetensors guarantees f32 alignment internally
            let (prefix, f32s, suffix) = unsafe { raw.align_to::<f32>() };
            debug_assert!(prefix.is_empty(), "f32 data not aligned");
            debug_assert!(suffix.is_empty(), "f32 data has trailing bytes");
            f32s.to_vec()
        }
        other => {
            return Err(BurnBackendError::ComponentRead {
                path: path.to_path_buf(),
                message: format!("tensor `{key}` has unsupported dtype {other:?}; expected F32"),
            });
        }
    };

    Ok(ClipWeightData { data })
}
