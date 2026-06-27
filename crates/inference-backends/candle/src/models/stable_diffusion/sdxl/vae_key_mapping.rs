//! Original Stability AI SDXL VAE compvis/LDM → diffusers key mapping.
//!
//! Candle's [`AutoEncoderKL`] (from `candle-transformers::stable_diffusion::vae`)
//! expects **diffusers-style** VAE keys with names like
//! `encoder.down_blocks.0.resnets.0.norm1.weight`. The original Stability AI
//! SDXL checkpoint stores VAE weights under `first_stage_model.*` using the
//! older **compvis/LDM** layout with `encoder.down.0.block.0.norm1.weight`.
//!
//! [`map_original_sdxl_vae_key`] translates between the two. Keys already
//! in diffusers layout (e.g. `encoder.conv_in.weight`) are intentionally
//! rejected so the caller can fall back to a 1:1 prefix strip.
//!
//! ## Key mapping
//!
//! | Compvis (after `first_stage_model.`) | Diffusers |
//! |--------------------------------------|-----------|
//! | `encoder.conv_in.{weight,bias}` | `encoder.conv_in.{weight,bias}` (1:1) |
//! | `encoder.conv_out.{weight,bias}` | `encoder.conv_out.{weight,bias}` (1:1) |
//! | `encoder.norm_out.{weight,bias}` | `encoder.conv_norm_out.{weight,bias}` |
//! | `encoder.down.{N}.block.{M}.*` | `encoder.down_blocks.{N}.resnets.{M}.*` |
//! | `encoder.down.{N}.downsample.*` | `encoder.down_blocks.{N}.downsamplers.0.*` |
//! | `encoder.mid.block_{N}.*` | `encoder.mid_block.resnets.{N-1}.*` |
//! | `encoder.mid.attn_1.norm.*` | `encoder.mid_block.attentions.0.group_norm.*` |
//! | `encoder.mid.attn_1.{q,k,v}.{weight,bias}` | `encoder.mid_block.attentions.0.to_{q,k,v}.{weight,bias}` |
//! | `encoder.mid.attn_1.to_{q,k,v}.{weight,bias}` | `encoder.mid_block.attentions.0.to_{q,k,v}.{weight,bias}` (1:1) |
//! | `encoder.mid.attn_1.{proj_out,to_out}.{weight,bias}` | `encoder.mid_block.attentions.0.to_out.0.{weight,bias}` |
//! | `encoder.mid.attn_1.to_qkv.weight` | split into `to_q/k/v.weight` row ranges |
//! | `decoder.up.{N}.block.{M}.*` | `decoder.up_blocks.{3-N}.resnets.{M}.*` |
//! | `decoder.up.{N}.upsample.*` | `decoder.up_blocks.{3-N}.upsamplers.0.*` |
//! | `quant_conv.{weight,bias}` | `quant_conv.{weight,bias}` (1:1) |
//! | `post_quant_conv.{weight,bias}` | `post_quant_conv.{weight,bias}` (1:1) |
//!
//! ## Why not just 1:1 strip
//!
//! The original import pipeline only stripped the `first_stage_model.`
//! prefix. Candle's `AutoEncoderKL::new` then failed to materialize
//! because `encoder.down.0.block.0.norm1.weight` (compvis) was looked up
//! as `encoder.down_blocks.0.resnets.0.norm1.weight` (diffusers). See
//! issue `real-inference/07a1` for the original reproduction.
//!
//! [`AutoEncoderKL`]: https://docs.rs/candle-transformers/latest/candle_transformers/models/stable_diffusion/vae/struct.AutoEncoderKL.html

use super::checkpoint_import::SdxlConvertedComponent;
use super::checkpoint_mapping::{SdxlMappedTensor, SdxlTensorMappingError};

/// SDXL VAE mid-block feature channels. The SDXL `AutoEncoderKLConfig`
/// uses `block_out_channels = [128, 256, 512, 512]`, so the mid block
/// operates on 512-channel feature maps.
///
/// `LinearAttention.to_qkv` is a 1×1 Conv2d of shape `[3 * 512, 512, 1, 1]`
/// which we slice into three `[512, 512]` weights for `to_q`/`to_k`/`to_v`.
const VAE_MID_ATTN_EMBED_DIM: usize = 512;
const VAE_MID_ATTN_FUSED_QKV_DIM: usize = 3 * VAE_MID_ATTN_EMBED_DIM;

/// Map a single `first_stage_model.<suffix>` VAE tensor to one or more
/// diffusers-style target tensors.
///
/// `full_source` is the original checkpoint tensor name (used for error
/// messages). `suffix` is the part after `first_stage_model.`.
///
/// Returns [`SdxlTensorMappingError::UnknownRequiredFamily`] when the
/// suffix is not a recognized compvis VAE key; callers may then fall
/// back to a 1:1 prefix strip when the suffix is already in diffusers
/// layout.
pub(crate) fn map_original_sdxl_vae_key(
    full_source: &str,
    suffix: &str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    match suffix {
        "quant_conv.weight"
        | "quant_conv.bias"
        | "post_quant_conv.weight"
        | "post_quant_conv.bias" => {
            return Ok(vec![single(full_source, suffix)]);
        }
        _ => {}
    }

    if let Some(rest) = suffix.strip_prefix("encoder.") {
        return map_vae_side(full_source, rest, "encoder");
    }
    if let Some(rest) = suffix.strip_prefix("decoder.") {
        return map_vae_side(full_source, rest, "decoder");
    }

    Err(SdxlTensorMappingError::UnknownRequiredFamily {
        name: full_source.to_owned(),
    })
}

fn map_vae_side(
    full_source: &str,
    rest: &str,
    side: &'static str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    match rest {
        "conv_in.weight" | "conv_in.bias" | "conv_out.weight" | "conv_out.bias" => {
            let target = format!("{side}.{rest}");
            return Ok(vec![single(full_source, &target)]);
        }
        "norm_out.weight" | "norm_out.bias" => {
            let target = format!("{side}.conv_norm_out.{}", last_segment(rest));
            return Ok(vec![single(full_source, &target)]);
        }
        _ => {}
    }

    if let Some(rest) = rest.strip_prefix("down.") {
        return map_vae_down(full_source, rest, side);
    }
    if let Some(rest) = rest.strip_prefix("up.") {
        return map_vae_up(full_source, rest, side);
    }
    if let Some(rest) = rest.strip_prefix("mid.") {
        return map_vae_mid(full_source, rest, side);
    }

    Err(SdxlTensorMappingError::UnknownRequiredFamily {
        name: full_source.to_owned(),
    })
}

fn map_vae_down(
    full_source: &str,
    rest: &str,
    side: &'static str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    let (idx_str, after) = split_index_segment(full_source, rest)?;
    let block: usize = idx_str.parse().map_err(|_| unknown_family(full_source))?;

    if let Some(rest) = after.strip_prefix("block.") {
        let (idx_str, suffix) = split_index_segment(full_source, rest)?;
        let layer: usize = idx_str.parse().map_err(|_| unknown_family(full_source))?;
        let suffix = map_resnet_suffix(suffix);
        let target = format!("{side}.down_blocks.{block}.resnets.{layer}.{suffix}");
        return Ok(vec![single(full_source, &target)]);
    }

    if let Some(rest) = after.strip_prefix("downsample.") {
        let target = format!("{side}.down_blocks.{block}.downsamplers.0.{rest}");
        return Ok(vec![single(full_source, &target)]);
    }

    Err(unknown_family(full_source))
}

fn map_vae_up(
    full_source: &str,
    rest: &str,
    side: &'static str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    let (idx_str, after) = split_index_segment(full_source, rest)?;
    let block: usize = idx_str.parse().map_err(|_| unknown_family(full_source))?;
    let target_block = if side == "decoder" {
        decoder_up_target_block(full_source, block)?
    } else {
        block
    };

    if let Some(rest) = after.strip_prefix("block.") {
        let (idx_str, suffix) = split_index_segment(full_source, rest)?;
        let layer: usize = idx_str.parse().map_err(|_| unknown_family(full_source))?;
        let suffix = map_resnet_suffix(suffix);
        let target = format!("{side}.up_blocks.{target_block}.resnets.{layer}.{suffix}");
        return Ok(vec![single(full_source, &target)]);
    }

    if let Some(rest) = after.strip_prefix("upsample.") {
        let target = format!("{side}.up_blocks.{target_block}.upsamplers.0.{rest}");
        return Ok(vec![single(full_source, &target)]);
    }

    Err(unknown_family(full_source))
}

fn decoder_up_target_block(
    full_source: &str,
    source_block: usize,
) -> Result<usize, SdxlTensorMappingError> {
    const SDXL_VAE_UP_BLOCKS: usize = 4;
    SDXL_VAE_UP_BLOCKS
        .checked_sub(1)
        .and_then(|last| last.checked_sub(source_block))
        .ok_or_else(|| unknown_family(full_source))
}

fn map_vae_mid(
    full_source: &str,
    rest: &str,
    side: &'static str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    if let Some(rest) = rest.strip_prefix("block_") {
        let (idx_str, suffix) = split_index_segment(full_source, rest)?;
        let idx: usize = idx_str.parse().map_err(|_| unknown_family(full_source))?;
        let target_layer = idx
            .checked_sub(1)
            .ok_or_else(|| unknown_family(full_source))?;
        let target = format!("{side}.mid_block.resnets.{target_layer}.{suffix}");
        return Ok(vec![single(full_source, &target)]);
    }

    if let Some(rest) = rest.strip_prefix("attn_1.") {
        return map_vae_attn(full_source, rest, side);
    }

    Err(unknown_family(full_source))
}

fn map_vae_attn(
    full_source: &str,
    rest: &str,
    side: &'static str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    let prefix = format!("{side}.mid_block.attentions.0");

    match rest {
        "norm.weight" | "norm.bias" => Ok(vec![single(
            full_source,
            &format!("{prefix}.group_norm.{}", last_segment(rest)),
        )]),
        "q.weight" => Ok(vec![single(full_source, &format!("{prefix}.to_q.weight"))]),
        "q.bias" => Ok(vec![single(full_source, &format!("{prefix}.to_q.bias"))]),
        "k.weight" => Ok(vec![single(full_source, &format!("{prefix}.to_k.weight"))]),
        "k.bias" => Ok(vec![single(full_source, &format!("{prefix}.to_k.bias"))]),
        "v.weight" => Ok(vec![single(full_source, &format!("{prefix}.to_v.weight"))]),
        "v.bias" => Ok(vec![single(full_source, &format!("{prefix}.to_v.bias"))]),
        "to_q.weight" => Ok(vec![single(full_source, &format!("{prefix}.to_q.weight"))]),
        "to_q.bias" => Ok(vec![single(full_source, &format!("{prefix}.to_q.bias"))]),
        "to_k.weight" => Ok(vec![single(full_source, &format!("{prefix}.to_k.weight"))]),
        "to_k.bias" => Ok(vec![single(full_source, &format!("{prefix}.to_k.bias"))]),
        "to_v.weight" => Ok(vec![single(full_source, &format!("{prefix}.to_v.weight"))]),
        "to_v.bias" => Ok(vec![single(full_source, &format!("{prefix}.to_v.bias"))]),
        "proj_out.weight" => Ok(vec![single(
            full_source,
            &format!("{prefix}.to_out.0.weight"),
        )]),
        "proj_out.bias" => Ok(vec![single(
            full_source,
            &format!("{prefix}.to_out.0.bias"),
        )]),
        "to_out.weight" => Ok(vec![single(
            full_source,
            &format!("{prefix}.to_out.0.weight"),
        )]),
        "to_out.bias" => Ok(vec![single(
            full_source,
            &format!("{prefix}.to_out.0.bias"),
        )]),
        "to_qkv.weight" => {
            let third = VAE_MID_ATTN_EMBED_DIM;
            Ok(vec![
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::Vae,
                    target_name: format!("{prefix}.to_q.weight"),
                    source_row_range: Some((0, third)),
                },
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::Vae,
                    target_name: format!("{prefix}.to_k.weight"),
                    source_row_range: Some((third, 2 * third)),
                },
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::Vae,
                    target_name: format!("{prefix}.to_v.weight"),
                    source_row_range: Some((2 * third, VAE_MID_ATTN_FUSED_QKV_DIM)),
                },
            ])
        }
        _ => Err(unknown_family(full_source)),
    }
}

fn split_index_segment<'a>(
    full_source: &str,
    value: &'a str,
) -> Result<(&'a str, &'a str), SdxlTensorMappingError> {
    value
        .split_once('.')
        .ok_or_else(|| unknown_family(full_source))
}

fn unknown_family(full_source: &str) -> SdxlTensorMappingError {
    SdxlTensorMappingError::UnknownRequiredFamily {
        name: full_source.to_owned(),
    }
}

fn last_segment(rest: &str) -> &str {
    rest.rsplit_once('.').map(|(_, tail)| tail).unwrap_or(rest)
}

fn map_resnet_suffix(suffix: &str) -> String {
    if let Some(rest) = suffix.strip_prefix("nin_shortcut.") {
        format!("conv_shortcut.{rest}")
    } else {
        suffix.to_owned()
    }
}

fn single(_full_source: &str, target_name: &str) -> SdxlMappedTensor {
    SdxlMappedTensor {
        component: SdxlConvertedComponent::Vae,
        target_name: target_name.to_owned(),
        source_row_range: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SdxlMappedTensor, SdxlTensorMappingError, VAE_MID_ATTN_EMBED_DIM,
        VAE_MID_ATTN_FUSED_QKV_DIM, map_original_sdxl_vae_key,
    };
    use crate::models::stable_diffusion::sdxl::checkpoint_import::SdxlConvertedComponent;

    fn first_target(suffix: &str) -> String {
        let full = format!("first_stage_model.{suffix}");
        let mapped = map_original_sdxl_vae_key(&full, suffix).expect("compvis key should map");
        assert_eq!(mapped.len(), 1, "expected single mapping, got {mapped:?}");
        assert_eq!(mapped[0].component, SdxlConvertedComponent::Vae);
        assert_eq!(mapped[0].source_row_range, None);
        mapped[0].target_name.clone()
    }

    fn all_targets(suffix: &str) -> Vec<SdxlMappedTensor> {
        let full = format!("first_stage_model.{suffix}");
        map_original_sdxl_vae_key(&full, suffix).expect("compvis key should map")
    }

    #[test]
    fn conv_in_and_conv_out_are_one_to_one() {
        assert_eq!(
            first_target("encoder.conv_in.weight"),
            "encoder.conv_in.weight"
        );
        assert_eq!(first_target("encoder.conv_in.bias"), "encoder.conv_in.bias");
        assert_eq!(
            first_target("encoder.conv_out.weight"),
            "encoder.conv_out.weight"
        );
        assert_eq!(
            first_target("encoder.conv_out.bias"),
            "encoder.conv_out.bias"
        );
        assert_eq!(
            first_target("decoder.conv_in.weight"),
            "decoder.conv_in.weight"
        );
        assert_eq!(
            first_target("decoder.conv_out.weight"),
            "decoder.conv_out.weight"
        );
    }

    #[test]
    fn norm_out_renames_to_conv_norm_out() {
        assert_eq!(
            first_target("encoder.norm_out.weight"),
            "encoder.conv_norm_out.weight"
        );
        assert_eq!(
            first_target("encoder.norm_out.bias"),
            "encoder.conv_norm_out.bias"
        );
        assert_eq!(
            first_target("decoder.norm_out.weight"),
            "decoder.conv_norm_out.weight"
        );
        assert_eq!(
            first_target("decoder.norm_out.bias"),
            "decoder.conv_norm_out.bias"
        );
    }

    #[test]
    fn quant_and_post_quant_conv_are_one_to_one() {
        assert_eq!(first_target("quant_conv.weight"), "quant_conv.weight");
        assert_eq!(first_target("quant_conv.bias"), "quant_conv.bias");
        assert_eq!(
            first_target("post_quant_conv.weight"),
            "post_quant_conv.weight"
        );
        assert_eq!(first_target("post_quant_conv.bias"), "post_quant_conv.bias");
    }

    #[test]
    fn encoder_down_blocks_map_resnets_and_downsamplers() {
        assert_eq!(
            first_target("encoder.down.0.block.0.norm1.weight"),
            "encoder.down_blocks.0.resnets.0.norm1.weight"
        );
        assert_eq!(
            first_target("encoder.down.0.block.0.conv1.weight"),
            "encoder.down_blocks.0.resnets.0.conv1.weight"
        );
        assert_eq!(
            first_target("encoder.down.0.block.0.norm2.bias"),
            "encoder.down_blocks.0.resnets.0.norm2.bias"
        );
        assert_eq!(
            first_target("encoder.down.0.block.0.conv2.weight"),
            "encoder.down_blocks.0.resnets.0.conv2.weight"
        );
        assert_eq!(
            first_target("encoder.down.1.block.1.nin_shortcut.weight"),
            "encoder.down_blocks.1.resnets.1.conv_shortcut.weight"
        );
        assert_eq!(
            first_target("encoder.down.0.downsample.conv.weight"),
            "encoder.down_blocks.0.downsamplers.0.conv.weight"
        );
        assert_eq!(
            first_target("encoder.down.2.downsample.conv.bias"),
            "encoder.down_blocks.2.downsamplers.0.conv.bias"
        );
    }

    #[test]
    fn decoder_up_blocks_map_resnets_and_upsamplers() {
        assert_eq!(
            first_target("decoder.up.0.block.0.norm1.weight"),
            "decoder.up_blocks.3.resnets.0.norm1.weight"
        );
        assert_eq!(
            first_target("decoder.up.2.block.1.nin_shortcut.weight"),
            "decoder.up_blocks.1.resnets.1.conv_shortcut.weight"
        );
        assert_eq!(
            first_target("decoder.up.1.upsample.conv.weight"),
            "decoder.up_blocks.2.upsamplers.0.conv.weight"
        );
        assert_eq!(
            first_target("decoder.up.3.upsample.conv.bias"),
            "decoder.up_blocks.0.upsamplers.0.conv.bias"
        );
    }

    #[test]
    fn mid_block_resnets_use_zero_indexed_layers() {
        assert_eq!(
            first_target("encoder.mid.block_1.norm1.weight"),
            "encoder.mid_block.resnets.0.norm1.weight"
        );
        assert_eq!(
            first_target("encoder.mid.block_2.conv2.bias"),
            "encoder.mid_block.resnets.1.conv2.bias"
        );
        assert_eq!(
            first_target("decoder.mid.block_1.norm1.weight"),
            "decoder.mid_block.resnets.0.norm1.weight"
        );
    }

    #[test]
    fn mid_attn_norm_renames_to_group_norm() {
        assert_eq!(
            first_target("encoder.mid.attn_1.norm.weight"),
            "encoder.mid_block.attentions.0.group_norm.weight"
        );
        assert_eq!(
            first_target("encoder.mid.attn_1.norm.bias"),
            "encoder.mid_block.attentions.0.group_norm.bias"
        );
        assert_eq!(
            first_target("decoder.mid.attn_1.norm.weight"),
            "decoder.mid_block.attentions.0.group_norm.weight"
        );
    }

    #[test]
    fn mid_attn_qkv_separate_renames_to_to_qkv() {
        assert_eq!(
            first_target("encoder.mid.attn_1.q.weight"),
            "encoder.mid_block.attentions.0.to_q.weight"
        );
        assert_eq!(
            first_target("encoder.mid.attn_1.k.bias"),
            "encoder.mid_block.attentions.0.to_k.bias"
        );
        assert_eq!(
            first_target("encoder.mid.attn_1.v.weight"),
            "encoder.mid_block.attentions.0.to_v.weight"
        );
        assert_eq!(
            first_target("decoder.mid.attn_1.q.bias"),
            "decoder.mid_block.attentions.0.to_q.bias"
        );
    }

    #[test]
    fn mid_attn_to_qkv_keys_are_one_to_one() {
        assert_eq!(
            first_target("encoder.mid.attn_1.to_q.weight"),
            "encoder.mid_block.attentions.0.to_q.weight"
        );
        assert_eq!(
            first_target("encoder.mid.attn_1.to_k.bias"),
            "encoder.mid_block.attentions.0.to_k.bias"
        );
        assert_eq!(
            first_target("encoder.mid.attn_1.to_v.weight"),
            "encoder.mid_block.attentions.0.to_v.weight"
        );
    }

    #[test]
    fn mid_attn_proj_out_and_to_out_rename_to_to_out_zero() {
        assert_eq!(
            first_target("encoder.mid.attn_1.proj_out.weight"),
            "encoder.mid_block.attentions.0.to_out.0.weight"
        );
        assert_eq!(
            first_target("encoder.mid.attn_1.proj_out.bias"),
            "encoder.mid_block.attentions.0.to_out.0.bias"
        );
        assert_eq!(
            first_target("encoder.mid.attn_1.to_out.weight"),
            "encoder.mid_block.attentions.0.to_out.0.weight"
        );
        assert_eq!(
            first_target("encoder.mid.attn_1.to_out.bias"),
            "encoder.mid_block.attentions.0.to_out.0.bias"
        );
    }

    #[test]
    fn mid_attn_fused_to_qkv_splits_into_three_row_ranges() {
        let mapped = all_targets("encoder.mid.attn_1.to_qkv.weight");
        assert_eq!(mapped.len(), 3);

        let expected_targets = [
            "encoder.mid_block.attentions.0.to_q.weight",
            "encoder.mid_block.attentions.0.to_k.weight",
            "encoder.mid_block.attentions.0.to_v.weight",
        ];
        let expected_ranges = [
            (0, VAE_MID_ATTN_EMBED_DIM),
            (VAE_MID_ATTN_EMBED_DIM, 2 * VAE_MID_ATTN_EMBED_DIM),
            (2 * VAE_MID_ATTN_EMBED_DIM, VAE_MID_ATTN_FUSED_QKV_DIM),
        ];
        for (i, ((target, range), (expected_target, expected_range))) in mapped
            .iter()
            .map(|m| (m.target_name.as_str(), m.source_row_range))
            .zip(expected_targets.iter().zip(expected_ranges.iter()))
            .enumerate()
        {
            assert_eq!(target, *expected_target, "target[{i}] mismatch");
            assert_eq!(range, Some(*expected_range), "range[{i}] mismatch");
            assert_eq!(mapped[i].component, SdxlConvertedComponent::Vae);
        }
    }

    #[test]
    fn unknown_compvis_vae_keys_are_rejected() {
        let err = map_original_sdxl_vae_key(
            "first_stage_model.encoder.unknown.path.weight",
            "encoder.unknown.path.weight",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SdxlTensorMappingError::UnknownRequiredFamily { .. }
        ));

        let err = map_original_sdxl_vae_key(
            "first_stage_model.encoder.down.foo.weight",
            "encoder.down.foo.weight",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SdxlTensorMappingError::UnknownRequiredFamily { .. }
        ));

        let err = map_original_sdxl_vae_key(
            "first_stage_model.encoder.mid.block_0.weight",
            "encoder.mid.block_0.weight",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SdxlTensorMappingError::UnknownRequiredFamily { .. }
        ));
    }

    #[test]
    fn keys_without_vae_prefix_are_rejected() {
        // No `encoder.` / `decoder.` / `quant_conv.` / `post_quant_conv.` prefix.
        let err = map_original_sdxl_vae_key(
            "first_stage_model.encoder.down.weight",
            "encoder.down.weight",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SdxlTensorMappingError::UnknownRequiredFamily { .. }
        ));

        // Without the first_stage_model. prefix the suffix wouldn't
        // normally reach this function, but we guard anyway.
        let err = map_original_sdxl_vae_key(
            "model.diffusion_model.input.weight",
            "model.diffusion_model.input.weight",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SdxlTensorMappingError::UnknownRequiredFamily { .. }
        ));
    }

    #[test]
    fn mid_attn_unknown_keys_are_rejected() {
        let err = map_original_sdxl_vae_key(
            "first_stage_model.encoder.mid.attn_1.unknown.weight",
            "encoder.mid.attn_1.unknown.weight",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SdxlTensorMappingError::UnknownRequiredFamily { .. }
        ));
    }
}
