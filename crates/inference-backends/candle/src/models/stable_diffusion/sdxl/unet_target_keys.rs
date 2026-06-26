#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdxlUnetTargetFamily {
    ConvIn,
    TimeEmbedding,
    DownBlockResnet,
    DownBlockAttention,
    DownBlockDownsample,
    MidBlockResnet,
    MidBlockAttention,
    UpBlockResnet,
    UpBlockAttention,
    UpBlockUpsample,
    ConvNormOut,
    ConvOut,
}

#[cfg(test)]
impl SdxlUnetTargetFamily {
    pub(crate) fn prefix(self) -> &'static str {
        match self {
            Self::ConvIn => "conv_in.",
            Self::TimeEmbedding => "time_embedding.",
            Self::DownBlockResnet => "down_blocks.<block>.resnets.<layer>.",
            Self::DownBlockAttention => "down_blocks.<block>.attentions.<layer>.",
            Self::DownBlockDownsample => "down_blocks.<block>.downsamplers.0.conv.",
            Self::MidBlockResnet => "mid_block.resnets.<layer>.",
            Self::MidBlockAttention => "mid_block.attentions.<layer>.",
            Self::UpBlockResnet => "up_blocks.<block>.resnets.<layer>.",
            Self::UpBlockAttention => "up_blocks.<block>.attentions.<layer>.",
            Self::UpBlockUpsample => "up_blocks.<block>.upsamplers.0.conv.",
            Self::ConvNormOut => "conv_norm_out.",
            Self::ConvOut => "conv_out.",
        }
    }
}

pub(crate) fn classify_sdxl_unet_target_key(name: &str) -> Option<SdxlUnetTargetFamily> {
    if name.starts_with("conv_in.") {
        return Some(SdxlUnetTargetFamily::ConvIn);
    }
    if name.starts_with("time_embedding.") {
        return Some(SdxlUnetTargetFamily::TimeEmbedding);
    }
    if name.starts_with("down_blocks.") {
        if name.contains(".resnets.") {
            return Some(SdxlUnetTargetFamily::DownBlockResnet);
        }
        if name.contains(".attentions.") {
            return Some(SdxlUnetTargetFamily::DownBlockAttention);
        }
        if name.contains(".downsamplers.0.conv.") {
            return Some(SdxlUnetTargetFamily::DownBlockDownsample);
        }
    }
    if name.starts_with("mid_block.") {
        if name.contains(".resnets.") {
            return Some(SdxlUnetTargetFamily::MidBlockResnet);
        }
        if name.contains(".attentions.") {
            return Some(SdxlUnetTargetFamily::MidBlockAttention);
        }
    }
    if name.starts_with("up_blocks.") {
        if name.contains(".resnets.") {
            return Some(SdxlUnetTargetFamily::UpBlockResnet);
        }
        if name.contains(".attentions.") {
            return Some(SdxlUnetTargetFamily::UpBlockAttention);
        }
        if name.contains(".upsamplers.0.conv.") {
            return Some(SdxlUnetTargetFamily::UpBlockUpsample);
        }
    }
    if name.starts_with("conv_norm_out.") {
        return Some(SdxlUnetTargetFamily::ConvNormOut);
    }
    if name.starts_with("conv_out.") {
        return Some(SdxlUnetTargetFamily::ConvOut);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{SdxlUnetTargetFamily, classify_sdxl_unet_target_key};

    #[test]
    fn classifies_candle_sdxl_example_target_key_families() {
        let cases = [
            ("conv_in.weight", SdxlUnetTargetFamily::ConvIn),
            (
                "time_embedding.linear_1.weight",
                SdxlUnetTargetFamily::TimeEmbedding,
            ),
            (
                "down_blocks.1.resnets.0.conv1.weight",
                SdxlUnetTargetFamily::DownBlockResnet,
            ),
            (
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight",
                SdxlUnetTargetFamily::DownBlockAttention,
            ),
            (
                "down_blocks.1.downsamplers.0.conv.weight",
                SdxlUnetTargetFamily::DownBlockDownsample,
            ),
            (
                "mid_block.resnets.1.time_emb_proj.bias",
                SdxlUnetTargetFamily::MidBlockResnet,
            ),
            (
                "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight",
                SdxlUnetTargetFamily::MidBlockAttention,
            ),
            (
                "up_blocks.0.resnets.2.conv_shortcut.weight",
                SdxlUnetTargetFamily::UpBlockResnet,
            ),
            (
                "up_blocks.0.attentions.2.proj_in.weight",
                SdxlUnetTargetFamily::UpBlockAttention,
            ),
            (
                "up_blocks.1.upsamplers.0.conv.bias",
                SdxlUnetTargetFamily::UpBlockUpsample,
            ),
            ("conv_norm_out.weight", SdxlUnetTargetFamily::ConvNormOut),
            ("conv_out.bias", SdxlUnetTargetFamily::ConvOut),
        ];

        for (name, family) in cases {
            assert_eq!(classify_sdxl_unet_target_key(name), Some(family), "{name}");
            assert!(!family.prefix().is_empty());
        }
    }

    #[test]
    fn rejects_non_unet_target_keys() {
        assert_eq!(
            classify_sdxl_unet_target_key("text_model.embeddings.token_embedding.weight"),
            None
        );
        assert_eq!(classify_sdxl_unet_target_key("label_emb.0.0.weight"), None);
    }
}
