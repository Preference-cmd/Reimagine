use super::unet_target_keys::classify_sdxl_unet_target_key;

const ORIGINAL_UNET_PREFIX: &str = "model.diffusion_model.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdxlOriginalUnetMappedKey {
    pub(crate) source_name: String,
    pub(crate) target_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlOriginalUnetMappingDecision {
    Map(SdxlOriginalUnetMappedKey),
    Ignore { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlOriginalUnetMappingError {
    NotOriginalUnet { name: String },
    UnsupportedFamily { name: String, family: String },
    UnsupportedBlockIndex { name: String, index: usize },
    UnsupportedBlockPath { name: String, reason: String },
    UnsupportedTarget { name: String, target_name: String },
}

impl std::fmt::Display for SdxlOriginalUnetMappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotOriginalUnet { name } => {
                write!(f, "`{name}` is not an original SDXL UNet tensor")
            }
            Self::UnsupportedFamily { name, family } => write!(
                f,
                "original SDXL UNet tensor `{name}` has unsupported family `{family}`"
            ),
            Self::UnsupportedBlockIndex { name, index } => write!(
                f,
                "original SDXL UNet tensor `{name}` uses unsupported block index {index}"
            ),
            Self::UnsupportedBlockPath { name, reason } => write!(
                f,
                "original SDXL UNet tensor `{name}` has unsupported block path: {reason}"
            ),
            Self::UnsupportedTarget { name, target_name } => write!(
                f,
                "original SDXL UNet tensor `{name}` mapped to unsupported Candle SDXL target `{target_name}`"
            ),
        }
    }
}

impl std::error::Error for SdxlOriginalUnetMappingError {}

pub(crate) fn map_original_sdxl_unet_key(
    name: &str,
) -> Result<SdxlOriginalUnetMappingDecision, SdxlOriginalUnetMappingError> {
    let Some(local) = name.strip_prefix(ORIGINAL_UNET_PREFIX) else {
        return Err(SdxlOriginalUnetMappingError::NotOriginalUnet {
            name: name.to_owned(),
        });
    };

    let target_name = if let Some(rest) = local.strip_prefix("input_blocks.") {
        map_input_block(name, rest)?
    } else if let Some(rest) = local.strip_prefix("middle_block.") {
        map_middle_block(name, rest)?
    } else if let Some(rest) = local.strip_prefix("output_blocks.") {
        map_output_block(name, rest)?
    } else if let Some(rest) = local.strip_prefix("time_embed.") {
        map_time_embed(name, rest)?
    } else if let Some(rest) = local.strip_prefix("out.") {
        map_out(name, rest)?
    } else if local.starts_with("label_emb.") {
        return Ok(SdxlOriginalUnetMappingDecision::Ignore {
            reason: "Candle SDXL example path does not expose explicit added-conditioning weights; label_emb.* is recorded as ignored evidence".to_owned(),
        });
    } else {
        let family = local.split('.').next().unwrap_or(local).to_owned();
        return Err(SdxlOriginalUnetMappingError::UnsupportedFamily {
            name: name.to_owned(),
            family,
        });
    };

    if classify_sdxl_unet_target_key(&target_name).is_none() {
        return Err(SdxlOriginalUnetMappingError::UnsupportedTarget {
            name: name.to_owned(),
            target_name,
        });
    }

    Ok(SdxlOriginalUnetMappingDecision::Map(
        SdxlOriginalUnetMappedKey {
            source_name: name.to_owned(),
            target_name,
        },
    ))
}

fn map_time_embed(name: &str, rest: &str) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some((layer, suffix)) = split_index(rest) else {
        return unsupported_path(name, "expected time_embed.<0|2>.<weight|bias>");
    };
    let target_layer = match layer {
        0 => "linear_1",
        2 => "linear_2",
        _ => {
            return Err(SdxlOriginalUnetMappingError::UnsupportedBlockIndex {
                name: name.to_owned(),
                index: layer,
            });
        }
    };
    Ok(format!("time_embedding.{target_layer}.{suffix}"))
}

fn map_out(name: &str, rest: &str) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some((layer, suffix)) = split_index(rest) else {
        return unsupported_path(name, "expected out.<0|2>.<weight|bias>");
    };
    match layer {
        0 => Ok(format!("conv_norm_out.{suffix}")),
        2 => Ok(format!("conv_out.{suffix}")),
        _ => Err(SdxlOriginalUnetMappingError::UnsupportedBlockIndex {
            name: name.to_owned(),
            index: layer,
        }),
    }
}

fn map_input_block(name: &str, rest: &str) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some((block, inner)) = split_index(rest) else {
        return unsupported_path(name, "expected input_blocks.<index>.<path>");
    };
    if block == 0 {
        return map_input_conv(name, inner);
    }
    let Some(plan) = map_input_block_index(block) else {
        return Err(SdxlOriginalUnetMappingError::UnsupportedBlockIndex {
            name: name.to_owned(),
            index: block,
        });
    };
    match plan {
        DownPlan::Resnet { block, layer } => map_input_or_middle_inner(name, inner, block, layer),
        DownPlan::Downsample { block } => {
            map_downsample_block_inner(name, inner, &format!("down_blocks.{block}.downsamplers.0"))
        }
    }
}

fn map_middle_block(name: &str, rest: &str) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some((block, inner)) = split_index(rest) else {
        return unsupported_path(name, "expected middle_block.<0|1|2>.<path>");
    };
    match block {
        0 => map_resnet_inner(name, inner, "mid_block.resnets.0"),
        1 => map_attention_inner(name, inner, "mid_block.attentions.0"),
        2 => map_resnet_inner(name, inner, "mid_block.resnets.1"),
        _ => Err(SdxlOriginalUnetMappingError::UnsupportedBlockIndex {
            name: name.to_owned(),
            index: block,
        }),
    }
}

fn map_output_block(name: &str, rest: &str) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some((block, inner)) = split_index(rest) else {
        return unsupported_path(name, "expected output_blocks.<index>.<path>");
    };
    let Some(plan) = map_output_block_index(block) else {
        return Err(SdxlOriginalUnetMappingError::UnsupportedBlockIndex {
            name: name.to_owned(),
            index: block,
        });
    };

    let Some((layer, suffix)) = split_index(inner) else {
        return unsupported_path(name, "expected output_blocks.<index>.<layer>.<path>");
    };
    if layer == 0 {
        return map_resnet_inner(
            name,
            suffix,
            &format!("up_blocks.{}.resnets.{}", plan.block, plan.resnet),
        );
    }
    if layer == 1 && plan.has_attention {
        return map_attention_inner(
            name,
            suffix,
            &format!("up_blocks.{}.attentions.{}", plan.block, plan.resnet),
        );
    }
    if layer == 2 && plan.has_upsample {
        return map_upsample_inner(
            name,
            suffix,
            &format!("up_blocks.{}.upsamplers.0", plan.block),
        );
    }
    unsupported_path(name, "unsupported output block layer")
}

fn map_input_conv(name: &str, inner: &str) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some((layer, suffix)) = split_index(inner) else {
        return unsupported_path(name, "expected input_blocks.0.0.<weight|bias>");
    };
    if layer != 0 {
        return Err(SdxlOriginalUnetMappingError::UnsupportedBlockIndex {
            name: name.to_owned(),
            index: layer,
        });
    }
    Ok(format!("conv_in.{suffix}"))
}

fn map_input_or_middle_inner(
    name: &str,
    inner: &str,
    target_block: usize,
    target_layer: usize,
) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some((layer, suffix)) = split_index(inner) else {
        return unsupported_path(name, "expected input_blocks.<index>.<layer>.<path>");
    };
    match layer {
        0 => map_resnet_inner(
            name,
            suffix,
            &format!("down_blocks.{target_block}.resnets.{target_layer}"),
        ),
        1 => map_attention_inner(
            name,
            suffix,
            &format!("down_blocks.{target_block}.attentions.{target_layer}"),
        ),
        2 => map_downsample_inner(
            name,
            suffix,
            &format!("down_blocks.{target_block}.downsamplers.0"),
        ),
        _ => Err(SdxlOriginalUnetMappingError::UnsupportedBlockIndex {
            name: name.to_owned(),
            index: layer,
        }),
    }
}

fn map_resnet_inner(
    name: &str,
    inner: &str,
    target_prefix: &str,
) -> Result<String, SdxlOriginalUnetMappingError> {
    let target = if let Some(rest) = inner.strip_prefix("in_layers.0.") {
        format!("{target_prefix}.norm1.{rest}")
    } else if let Some(rest) = inner.strip_prefix("in_layers.2.") {
        format!("{target_prefix}.conv1.{rest}")
    } else if let Some(rest) = inner.strip_prefix("emb_layers.1.") {
        format!("{target_prefix}.time_emb_proj.{rest}")
    } else if let Some(rest) = inner.strip_prefix("out_layers.0.") {
        format!("{target_prefix}.norm2.{rest}")
    } else if let Some(rest) = inner.strip_prefix("out_layers.3.") {
        format!("{target_prefix}.conv2.{rest}")
    } else if let Some(rest) = inner.strip_prefix("skip_connection.") {
        format!("{target_prefix}.conv_shortcut.{rest}")
    } else {
        return unsupported_path(name, "unsupported ResNet subpath");
    };
    Ok(target)
}

fn map_attention_inner(
    name: &str,
    inner: &str,
    target_prefix: &str,
) -> Result<String, SdxlOriginalUnetMappingError> {
    if let Some(rest) = inner.strip_prefix("transformer_blocks.") {
        return Ok(format!("{target_prefix}.transformer_blocks.{rest}"));
    }
    if let Some(rest) = inner.strip_prefix("norm.") {
        return Ok(format!("{target_prefix}.norm.{rest}"));
    }
    if let Some(rest) = inner.strip_prefix("proj_in.") {
        return Ok(format!("{target_prefix}.proj_in.{rest}"));
    }
    if let Some(rest) = inner.strip_prefix("proj_out.") {
        return Ok(format!("{target_prefix}.proj_out.{rest}"));
    }
    unsupported_path(name, "unsupported attention subpath")
}

fn map_downsample_inner(
    name: &str,
    inner: &str,
    target_prefix: &str,
) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some(rest) = inner.strip_prefix("op.") else {
        return unsupported_path(name, "expected downsample op.<weight|bias>");
    };
    Ok(format!("{target_prefix}.conv.{rest}"))
}

fn map_downsample_block_inner(
    name: &str,
    inner: &str,
    target_prefix: &str,
) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some((layer, suffix)) = split_index(inner) else {
        return unsupported_path(name, "expected downsample layer <0>.op.<weight|bias>");
    };
    if layer != 0 {
        return Err(SdxlOriginalUnetMappingError::UnsupportedBlockIndex {
            name: name.to_owned(),
            index: layer,
        });
    }
    map_downsample_inner(name, suffix, target_prefix)
}

fn map_upsample_inner(
    name: &str,
    inner: &str,
    target_prefix: &str,
) -> Result<String, SdxlOriginalUnetMappingError> {
    let Some(rest) = inner.strip_prefix("conv.") else {
        return unsupported_path(name, "expected upsample conv.<weight|bias>");
    };
    Ok(format!("{target_prefix}.conv.{rest}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownPlan {
    Resnet { block: usize, layer: usize },
    Downsample { block: usize },
}

fn map_input_block_index(index: usize) -> Option<DownPlan> {
    match index {
        1 => Some(DownPlan::Resnet { block: 0, layer: 0 }),
        2 => Some(DownPlan::Resnet { block: 0, layer: 1 }),
        3 => Some(DownPlan::Downsample { block: 0 }),
        4 => Some(DownPlan::Resnet { block: 1, layer: 0 }),
        5 => Some(DownPlan::Resnet { block: 1, layer: 1 }),
        6 => Some(DownPlan::Downsample { block: 1 }),
        7 => Some(DownPlan::Resnet { block: 2, layer: 0 }),
        8 => Some(DownPlan::Resnet { block: 2, layer: 1 }),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UpPlan {
    block: usize,
    resnet: usize,
    has_attention: bool,
    has_upsample: bool,
}

fn map_output_block_index(index: usize) -> Option<UpPlan> {
    match index {
        0 => Some(UpPlan {
            block: 0,
            resnet: 0,
            has_attention: true,
            has_upsample: false,
        }),
        1 => Some(UpPlan {
            block: 0,
            resnet: 1,
            has_attention: true,
            has_upsample: false,
        }),
        2 => Some(UpPlan {
            block: 0,
            resnet: 2,
            has_attention: true,
            has_upsample: true,
        }),
        3 => Some(UpPlan {
            block: 1,
            resnet: 0,
            has_attention: true,
            has_upsample: false,
        }),
        4 => Some(UpPlan {
            block: 1,
            resnet: 1,
            has_attention: true,
            has_upsample: false,
        }),
        5 => Some(UpPlan {
            block: 1,
            resnet: 2,
            has_attention: true,
            has_upsample: true,
        }),
        6 => Some(UpPlan {
            block: 2,
            resnet: 0,
            has_attention: false,
            has_upsample: false,
        }),
        7 => Some(UpPlan {
            block: 2,
            resnet: 1,
            has_attention: false,
            has_upsample: false,
        }),
        8 => Some(UpPlan {
            block: 2,
            resnet: 2,
            has_attention: false,
            has_upsample: false,
        }),
        _ => None,
    }
}

fn split_index(value: &str) -> Option<(usize, &str)> {
    let (index, rest) = value.split_once('.')?;
    Some((index.parse().ok()?, rest))
}

fn unsupported_path<T>(
    name: &str,
    reason: impl Into<String>,
) -> Result<T, SdxlOriginalUnetMappingError> {
    Err(SdxlOriginalUnetMappingError::UnsupportedBlockPath {
        name: name.to_owned(),
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        SdxlOriginalUnetMappingDecision, SdxlOriginalUnetMappingError, map_original_sdxl_unet_key,
    };

    fn mapped(name: &str) -> String {
        match map_original_sdxl_unet_key(name).unwrap() {
            SdxlOriginalUnetMappingDecision::Map(mapped) => mapped.target_name,
            other => panic!("expected mapped key, got {other:?}"),
        }
    }

    #[test]
    fn maps_original_input_conv_time_embedding_and_output_layers() {
        assert_eq!(
            mapped("model.diffusion_model.input_blocks.0.0.weight"),
            "conv_in.weight"
        );
        assert_eq!(
            mapped("model.diffusion_model.time_embed.0.bias"),
            "time_embedding.linear_1.bias"
        );
        assert_eq!(
            mapped("model.diffusion_model.time_embed.2.weight"),
            "time_embedding.linear_2.weight"
        );
        assert_eq!(
            mapped("model.diffusion_model.out.0.weight"),
            "conv_norm_out.weight"
        );
        assert_eq!(mapped("model.diffusion_model.out.2.bias"), "conv_out.bias");
    }

    #[test]
    fn maps_representative_down_blocks() {
        assert_eq!(
            mapped("model.diffusion_model.input_blocks.1.0.in_layers.0.weight"),
            "down_blocks.0.resnets.0.norm1.weight"
        );
        assert_eq!(
            mapped("model.diffusion_model.input_blocks.3.0.op.weight"),
            "down_blocks.0.downsamplers.0.conv.weight"
        );
        assert_eq!(
            mapped("model.diffusion_model.input_blocks.4.0.emb_layers.1.bias"),
            "down_blocks.1.resnets.0.time_emb_proj.bias"
        );
        assert_eq!(
            mapped("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"),
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight"
        );
        assert_eq!(
            mapped("model.diffusion_model.input_blocks.7.0.out_layers.3.weight"),
            "down_blocks.2.resnets.0.conv2.weight"
        );
    }

    #[test]
    fn maps_representative_middle_block() {
        assert_eq!(
            mapped("model.diffusion_model.middle_block.0.in_layers.2.weight"),
            "mid_block.resnets.0.conv1.weight"
        );
        assert_eq!(
            mapped("model.diffusion_model.middle_block.1.proj_in.weight"),
            "mid_block.attentions.0.proj_in.weight"
        );
        assert_eq!(
            mapped("model.diffusion_model.middle_block.2.skip_connection.weight"),
            "mid_block.resnets.1.conv_shortcut.weight"
        );
    }

    #[test]
    fn maps_representative_up_blocks() {
        assert_eq!(
            mapped("model.diffusion_model.output_blocks.0.0.skip_connection.weight"),
            "up_blocks.0.resnets.0.conv_shortcut.weight"
        );
        assert_eq!(
            mapped(
                "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn1.to_q.weight"
            ),
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
        );
        assert_eq!(
            mapped("model.diffusion_model.output_blocks.3.0.out_layers.0.bias"),
            "up_blocks.1.resnets.0.norm2.bias"
        );
        assert_eq!(
            mapped("model.diffusion_model.output_blocks.5.2.conv.weight"),
            "up_blocks.1.upsamplers.0.conv.weight"
        );
    }

    #[test]
    fn label_embedding_is_explicitly_ignored_for_candle_example_path() {
        let decision =
            map_original_sdxl_unet_key("model.diffusion_model.label_emb.0.0.weight").unwrap();

        match decision {
            SdxlOriginalUnetMappingDecision::Ignore { reason } => {
                assert!(reason.contains("Candle SDXL example path"));
                assert!(reason.contains("added-conditioning"));
            }
            other => panic!("expected ignored label_emb, got {other:?}"),
        }
    }

    #[test]
    fn fails_closed_for_unknown_or_malformed_original_paths() {
        let err = map_original_sdxl_unet_key("conditioner.embedders.0.foo").unwrap_err();
        assert!(matches!(
            err,
            SdxlOriginalUnetMappingError::NotOriginalUnet { .. }
        ));

        let err = map_original_sdxl_unet_key("model.diffusion_model.input_blocks.99.0.weight")
            .unwrap_err();
        assert!(matches!(
            err,
            SdxlOriginalUnetMappingError::UnsupportedBlockIndex { index: 99, .. }
        ));

        let err = map_original_sdxl_unet_key("model.diffusion_model.input_blocks.1.0.foo.weight")
            .unwrap_err();
        assert!(matches!(
            err,
            SdxlOriginalUnetMappingError::UnsupportedBlockPath { .. }
        ));
    }
}
