use super::checkpoint_import::SdxlConvertedComponent;
use super::unet_key_mapping::{SdxlOriginalUnetMappingDecision, map_original_sdxl_unet_key};
use super::vae_key_mapping::map_original_sdxl_vae_key;

/// OpenCLIP-G hidden dimension (CLIP-G bigG encoder).
const CLIP_G_EMBED_DIM: usize = 1280;

/// OpenCLIP-G multi-head attention fused q/k/v dimension = 3 * embed_dim.
const CLIP_G_FUSED_QKV_DIM: usize = 3 * CLIP_G_EMBED_DIM;

/// Number of resblocks in OpenCLIP-G.
const CLIP_G_NUM_RESBLOCKS: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdxlMappedTensor {
    pub(crate) component: SdxlConvertedComponent,
    pub(crate) target_name: String,
    /// Row range to slice from the source tensor, if the source is
    /// a fused weight that needs splitting (e.g. OpenCLIP-G
    /// `in_proj_weight` which packs q/k/v). None means use the full
    /// tensor (1:1 copy).
    pub(crate) source_row_range: Option<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlTensorMappingError {
    Ignored { name: String, reason: String },
    UnknownRequiredFamily { name: String },
}

impl std::fmt::Display for SdxlTensorMappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ignored { name, reason } => {
                write!(f, "ignored checkpoint tensor `{name}`: {reason}")
            }
            Self::UnknownRequiredFamily { name } => {
                write!(
                    f,
                    "checkpoint tensor `{name}` is not mapped to any Candle example split component"
                )
            }
        }
    }
}

impl std::error::Error for SdxlTensorMappingError {}

pub(crate) fn map_sdxl_checkpoint_tensor(
    name: &str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    if name.starts_with("model_ema.") {
        return Err(SdxlTensorMappingError::Ignored {
            name: name.to_owned(),
            reason: "EMA weights are not part of Candle example split execution".to_owned(),
        });
    }

    if let Some(target_name) = map_diffusers_unet_name(name) {
        return Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::Unet,
            target_name,
            source_row_range: None,
        }]);
    }

    if name.starts_with("model.diffusion_model.") {
        match map_original_sdxl_unet_key(name) {
            Ok(SdxlOriginalUnetMappingDecision::Map(mapped)) => {
                return Ok(vec![SdxlMappedTensor {
                    component: SdxlConvertedComponent::Unet,
                    target_name: mapped.target_name,
                    source_row_range: None,
                }]);
            }
            Ok(SdxlOriginalUnetMappingDecision::Ignore { reason }) => {
                return Err(SdxlTensorMappingError::Ignored {
                    name: name.to_owned(),
                    reason,
                });
            }
            Err(err) => {
                return Err(SdxlTensorMappingError::UnknownRequiredFamily {
                    name: format!("{name}: {err}"),
                });
            }
        }
    }

    if let Some(suffix) = name.strip_prefix("first_stage_model.") {
        // First try the compvis/LDM → diffusers VAE mapping (issue
        // real-inference/07a1). When the suffix is already in diffusers
        // layout (e.g. `encoder.conv_in.weight`) the mapper rejects it
        // with `UnknownRequiredFamily`, in which case we fall back to a
        // 1:1 prefix strip so the existing diffusers-format fixtures
        // keep working.
        return match map_original_sdxl_vae_key(name, suffix) {
            Ok(mapped) => Ok(mapped),
            Err(SdxlTensorMappingError::UnknownRequiredFamily { .. }) => {
                Ok(vec![SdxlMappedTensor {
                    component: SdxlConvertedComponent::Vae,
                    target_name: suffix.to_owned(),
                    source_row_range: None,
                }])
            }
            Err(error) => Err(error),
        };
    }

    if let Some(target_name) = name.strip_prefix("conditioner.embedders.0.") {
        return Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipL,
            target_name: target_name.to_owned(),
            source_row_range: None,
        }]);
    }

    for prefix in ["conditioner.embedders.1.model.", "conditioner.embedders.1."] {
        if let Some(suffix) = name.strip_prefix(prefix) {
            return map_sdxl_clipg_name(name, suffix);
        }
    }

    Err(SdxlTensorMappingError::UnknownRequiredFamily {
        name: name.to_owned(),
    })
}

/// Maps an OpenCLIP-G or standard CLIP-G key (after the
/// `conditioner.embedders.1.model.` prefix has been stripped) to one
/// or more CLIP-shaped target tensors.
///
/// OpenCLIP-G uses `transformer.resblocks.{N}` and fused `in_proj_weight`
/// that must be split into q/k/v. Standard CLIP uses
/// `transformer.text_model.encoder.layers.{N}` with separated proj weights.
fn map_sdxl_clipg_name(
    full_source: &str,
    suffix: &str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    // Standard CLIP format already in the right shape.
    if let Some(rest) = suffix.strip_prefix("transformer.text_model.") {
        return Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!("transformer.text_model.{rest}"),
            source_row_range: None,
        }]);
    }

    // OpenCLIP-G transformer.resblocks.{N} format.
    if let Some(rest) = suffix.strip_prefix("transformer.resblocks.") {
        return map_openclipg_resblock(full_source, rest);
    }

    // Top-level keys: token_embedding, positional_embedding, ln_final, etc.
    // These don't have a resblocks prefix.
    map_openclipg_top_level(full_source, suffix)
}

/// Maps a top-level OpenCLIP-G key (no resblocks prefix) to CLIP
/// shaped target(s).
fn map_openclipg_top_level(
    _full_source: &str,
    suffix: &str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    let (target_name, is_ignored) = match suffix {
        "token_embedding.weight" => (
            "transformer.text_model.embeddings.token_embedding.weight",
            false,
        ),
        "positional_embedding" => (
            "transformer.text_model.embeddings.position_embedding.weight",
            false,
        ),
        "ln_final.weight" => ("transformer.text_model.final_layer_norm.weight", false),
        "ln_final.bias" => ("transformer.text_model.final_layer_norm.bias", false),
        "text_projection" => ("text_projection", true),
        "logit_scale" => ("logit_scale", true),
        _ => {
            return Err(SdxlTensorMappingError::UnknownRequiredFamily {
                name: _full_source.to_owned(),
            });
        }
    };

    if is_ignored {
        return Err(SdxlTensorMappingError::Ignored {
            name: _full_source.to_owned(),
            reason: format!(
                "OpenCLIP-G `{suffix}` is not part of the Candle example CLIP load path"
            ),
        });
    }

    Ok(vec![SdxlMappedTensor {
        component: SdxlConvertedComponent::ClipG,
        target_name: target_name.to_owned(),
        source_row_range: None,
    }])
}

/// Maps a transformer.resblocks.{N}.{path} key to CLIP-shaped targets.
///
/// For `in_proj_weight` / `in_proj_bias`, the fused tensor is split
/// into three row ranges (q/k/v). All other keys are 1:1 with the
/// target tensor.
fn map_openclipg_resblock(
    full_source: &str,
    rest: &str,
) -> Result<Vec<SdxlMappedTensor>, SdxlTensorMappingError> {
    // rest is like "0.attn.in_proj_weight" or "15.mlp.c_fc.bias"
    let dot = rest
        .find('.')
        .ok_or_else(|| SdxlTensorMappingError::UnknownRequiredFamily {
            name: full_source.to_owned(),
        })?;
    let layer_str = &rest[..dot];
    let layer: usize =
        layer_str
            .parse()
            .map_err(|_| SdxlTensorMappingError::UnknownRequiredFamily {
                name: full_source.to_owned(),
            })?;
    if layer >= CLIP_G_NUM_RESBLOCKS {
        return Err(SdxlTensorMappingError::UnknownRequiredFamily {
            name: full_source.to_owned(),
        });
    }
    let path = &rest[dot + 1..];

    match path {
        "attn.in_proj_weight" => {
            let third = CLIP_G_EMBED_DIM;
            Ok(vec![
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::ClipG,
                    target_name: format!(
                        "transformer.text_model.encoder.layers.{layer}.self_attn.q_proj.weight"
                    ),
                    source_row_range: Some((0, third)),
                },
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::ClipG,
                    target_name: format!(
                        "transformer.text_model.encoder.layers.{layer}.self_attn.k_proj.weight"
                    ),
                    source_row_range: Some((third, 2 * third)),
                },
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::ClipG,
                    target_name: format!(
                        "transformer.text_model.encoder.layers.{layer}.self_attn.v_proj.weight"
                    ),
                    source_row_range: Some((2 * third, CLIP_G_FUSED_QKV_DIM)),
                },
            ])
        }
        "attn.in_proj_bias" => {
            let third = CLIP_G_EMBED_DIM;
            Ok(vec![
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::ClipG,
                    target_name: format!(
                        "transformer.text_model.encoder.layers.{layer}.self_attn.q_proj.bias"
                    ),
                    source_row_range: Some((0, third)),
                },
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::ClipG,
                    target_name: format!(
                        "transformer.text_model.encoder.layers.{layer}.self_attn.k_proj.bias"
                    ),
                    source_row_range: Some((third, 2 * third)),
                },
                SdxlMappedTensor {
                    component: SdxlConvertedComponent::ClipG,
                    target_name: format!(
                        "transformer.text_model.encoder.layers.{layer}.self_attn.v_proj.bias"
                    ),
                    source_row_range: Some((2 * third, CLIP_G_FUSED_QKV_DIM)),
                },
            ])
        }
        "attn.out_proj.weight" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!(
                "transformer.text_model.encoder.layers.{layer}.self_attn.out_proj.weight"
            ),
            source_row_range: None,
        }]),
        "attn.out_proj.bias" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!(
                "transformer.text_model.encoder.layers.{layer}.self_attn.out_proj.bias"
            ),
            source_row_range: None,
        }]),
        "ln_1.weight" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!(
                "transformer.text_model.encoder.layers.{layer}.layer_norm1.weight"
            ),
            source_row_range: None,
        }]),
        "ln_1.bias" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!("transformer.text_model.encoder.layers.{layer}.layer_norm1.bias"),
            source_row_range: None,
        }]),
        "ln_2.weight" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!(
                "transformer.text_model.encoder.layers.{layer}.layer_norm2.weight"
            ),
            source_row_range: None,
        }]),
        "ln_2.bias" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!("transformer.text_model.encoder.layers.{layer}.layer_norm2.bias"),
            source_row_range: None,
        }]),
        "mlp.c_fc.weight" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!("transformer.text_model.encoder.layers.{layer}.mlp.fc1.weight"),
            source_row_range: None,
        }]),
        "mlp.c_fc.bias" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!("transformer.text_model.encoder.layers.{layer}.mlp.fc1.bias"),
            source_row_range: None,
        }]),
        "mlp.c_proj.weight" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!("transformer.text_model.encoder.layers.{layer}.mlp.fc2.weight"),
            source_row_range: None,
        }]),
        "mlp.c_proj.bias" => Ok(vec![SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipG,
            target_name: format!("transformer.text_model.encoder.layers.{layer}.mlp.fc2.bias"),
            source_row_range: None,
        }]),
        _ => Err(SdxlTensorMappingError::UnknownRequiredFamily {
            name: full_source.to_owned(),
        }),
    }
}

fn map_diffusers_unet_name(name: &str) -> Option<String> {
    if name == "conv_in.weight"
        || name == "conv_in.bias"
        || name.starts_with("time_embedding.")
        || name.starts_with("down_blocks.")
        || name.starts_with("up_blocks.")
        || name.starts_with("mid_block.")
        || name.starts_with("conv_norm_out.")
        || name.starts_with("conv_out.")
        || name.starts_with("class_embedding.")
    {
        return Some(name.to_owned());
    }
    name.strip_prefix("unet.").map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::{SdxlTensorMappingError, map_sdxl_checkpoint_tensor};
    use crate::models::stable_diffusion::sdxl::checkpoint_import::SdxlConvertedComponent;

    #[test]
    fn maps_diffusers_unet_keys_without_unet_prefix() {
        let results = map_sdxl_checkpoint_tensor("unet.down_blocks.0.resnets.0.conv1.weight")
            .expect("diffusers key maps");
        let mapped = &results[0];

        assert_eq!(mapped.component, SdxlConvertedComponent::Unet);
        assert_eq!(mapped.target_name, "down_blocks.0.resnets.0.conv1.weight");
    }

    #[test]
    fn maps_original_text_and_vae_to_component_local_keys() {
        let clip_l = &map_sdxl_checkpoint_tensor(
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        )
        .unwrap()[0];
        let clip_g = &map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.text_model.embeddings.token_embedding.weight",
        )
        .unwrap()[0];
        let vae =
            &map_sdxl_checkpoint_tensor("first_stage_model.decoder.conv_in.weight").unwrap()[0];

        assert_eq!(clip_l.component, SdxlConvertedComponent::ClipL);
        assert_eq!(
            clip_l.target_name,
            "transformer.text_model.embeddings.token_embedding.weight"
        );
        assert_eq!(clip_g.component, SdxlConvertedComponent::ClipG);
        assert_eq!(
            clip_g.target_name,
            "transformer.text_model.embeddings.token_embedding.weight"
        );
        assert_eq!(vae.component, SdxlConvertedComponent::Vae);
        assert_eq!(vae.target_name, "decoder.conv_in.weight");
    }

    #[test]
    fn maps_original_unet_keys_to_candle_example_targets() {
        let mapped = &map_sdxl_checkpoint_tensor("model.diffusion_model.input_blocks.0.0.weight")
            .unwrap()[0];
        assert_eq!(mapped.component, SdxlConvertedComponent::Unet);
        assert_eq!(mapped.target_name, "conv_in.weight");
    }

    #[test]
    fn ignores_label_emb_as_added_conditioning_evidence() {
        let err =
            map_sdxl_checkpoint_tensor("model.diffusion_model.label_emb.0.0.weight").unwrap_err();

        assert!(matches!(err, SdxlTensorMappingError::Ignored { .. }));
        let msg = err.to_string();
        assert!(msg.contains("Candle SDXL example path"), "{msg}");
        assert!(msg.contains("added-conditioning"), "{msg}");
    }

    // ---- OpenCLIP-G mapping tests ----

    #[test]
    fn maps_openclipg_in_proj_weight_to_qkv_split() {
        let results = map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
        )
        .expect("OpenCLIP-G in_proj_weight maps");
        assert_eq!(results.len(), 3);

        assert_eq!(results[0].component, SdxlConvertedComponent::ClipG);
        assert_eq!(
            results[0].target_name,
            "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(results[0].source_row_range, Some((0, 1280)));

        assert_eq!(results[1].component, SdxlConvertedComponent::ClipG);
        assert_eq!(
            results[1].target_name,
            "transformer.text_model.encoder.layers.0.self_attn.k_proj.weight"
        );
        assert_eq!(results[1].source_row_range, Some((1280, 2560)));

        assert_eq!(results[2].component, SdxlConvertedComponent::ClipG);
        assert_eq!(
            results[2].target_name,
            "transformer.text_model.encoder.layers.0.self_attn.v_proj.weight"
        );
        assert_eq!(results[2].source_row_range, Some((2560, 3840)));
    }

    #[test]
    fn maps_openclipg_in_proj_bias_to_qkv_split() {
        let results = map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.15.attn.in_proj_bias",
        )
        .expect("OpenCLIP-G in_proj_bias maps");
        assert_eq!(results.len(), 3);

        assert_eq!(
            results[0].target_name,
            "transformer.text_model.encoder.layers.15.self_attn.q_proj.bias"
        );
        assert_eq!(results[0].source_row_range, Some((0, 1280)));
        assert_eq!(
            results[1].target_name,
            "transformer.text_model.encoder.layers.15.self_attn.k_proj.bias"
        );
        assert_eq!(results[1].source_row_range, Some((1280, 2560)));
        assert_eq!(
            results[2].target_name,
            "transformer.text_model.encoder.layers.15.self_attn.v_proj.bias"
        );
        assert_eq!(results[2].source_row_range, Some((2560, 3840)));
    }

    #[test]
    fn maps_openclipg_layer_keys_1to1() {
        // out_proj
        let r = &map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.weight",
        )
        .unwrap()[0];
        assert_eq!(
            r.target_name,
            "transformer.text_model.encoder.layers.0.self_attn.out_proj.weight"
        );
        assert_eq!(r.source_row_range, None);

        let r = &map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.bias",
        )
        .unwrap()[0];
        assert_eq!(
            r.target_name,
            "transformer.text_model.encoder.layers.0.self_attn.out_proj.bias"
        );
        assert_eq!(r.source_row_range, None);

        // layer norms
        let r = &map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.5.ln_1.weight",
        )
        .unwrap()[0];
        assert_eq!(
            r.target_name,
            "transformer.text_model.encoder.layers.5.layer_norm1.weight"
        );
        assert_eq!(r.source_row_range, None);
    }

    #[test]
    fn maps_openclipg_mlp_keys_1to1() {
        let r = &map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.10.mlp.c_fc.weight",
        )
        .unwrap()[0];
        assert_eq!(
            r.target_name,
            "transformer.text_model.encoder.layers.10.mlp.fc1.weight"
        );
        assert_eq!(r.source_row_range, None);

        let r = &map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.10.mlp.c_proj.weight",
        )
        .unwrap()[0];
        assert_eq!(
            r.target_name,
            "transformer.text_model.encoder.layers.10.mlp.fc2.weight"
        );
    }

    #[test]
    fn maps_openclipg_top_level_keys() {
        let r = &map_sdxl_checkpoint_tensor("conditioner.embedders.1.model.token_embedding.weight")
            .unwrap()[0];
        assert_eq!(
            r.target_name,
            "transformer.text_model.embeddings.token_embedding.weight"
        );
        assert_eq!(r.source_row_range, None);

        let r = &map_sdxl_checkpoint_tensor("conditioner.embedders.1.model.positional_embedding")
            .unwrap()[0];
        assert_eq!(
            r.target_name,
            "transformer.text_model.embeddings.position_embedding.weight"
        );

        let r = &map_sdxl_checkpoint_tensor("conditioner.embedders.1.model.ln_final.weight")
            .unwrap()[0];
        assert_eq!(
            r.target_name,
            "transformer.text_model.final_layer_norm.weight"
        );
    }

    #[test]
    fn ignores_openclipg_text_projection_and_logit_scale() {
        let err = map_sdxl_checkpoint_tensor("conditioner.embedders.1.model.text_projection")
            .unwrap_err();
        assert!(matches!(err, SdxlTensorMappingError::Ignored { .. }));
        let msg = err.to_string();
        assert!(msg.contains("text_projection"), "{msg}");

        let err =
            map_sdxl_checkpoint_tensor("conditioner.embedders.1.model.logit_scale").unwrap_err();
        assert!(matches!(err, SdxlTensorMappingError::Ignored { .. }));
    }

    #[test]
    fn openclipg_beyond_31_blocks_is_unknown() {
        let err = map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.99.attn.in_proj_weight",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SdxlTensorMappingError::UnknownRequiredFamily { .. }
        ));
    }

    #[test]
    fn openclipg_unknown_path_is_unknown() {
        let err = map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.resblocks.3.attn.unknown_key",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SdxlTensorMappingError::UnknownRequiredFamily { .. }
        ));
    }
}
