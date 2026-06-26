use super::checkpoint_import::SdxlConvertedComponent;
use super::unet_key_mapping::map_original_sdxl_unet_key;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdxlMappedTensor {
    pub(crate) component: SdxlConvertedComponent,
    pub(crate) target_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlTensorMappingError {
    OriginalUnetUnsupported { name: String },
    UnknownRequiredFamily { name: String },
    Ignored,
}

impl std::fmt::Display for SdxlTensorMappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OriginalUnetUnsupported { name } => write!(
                f,
                "original SDXL UNet tensor `{name}` requires model.diffusion_model.* to Candle example UNet key mapping"
            ),
            Self::UnknownRequiredFamily { name } => {
                write!(
                    f,
                    "checkpoint tensor `{name}` is not mapped to any Candle example split component"
                )
            }
            Self::Ignored => f.write_str("ignored checkpoint tensor"),
        }
    }
}

impl std::error::Error for SdxlTensorMappingError {}

pub(crate) fn map_sdxl_checkpoint_tensor(
    name: &str,
) -> Result<SdxlMappedTensor, SdxlTensorMappingError> {
    if name.starts_with("model_ema.") {
        return Err(SdxlTensorMappingError::Ignored);
    }

    if let Some(target_name) = map_diffusers_unet_name(name) {
        return Ok(SdxlMappedTensor {
            component: SdxlConvertedComponent::Unet,
            target_name,
        });
    }

    if name.starts_with("model.diffusion_model.") {
        let _ = map_original_sdxl_unet_key(name);
        return Err(SdxlTensorMappingError::OriginalUnetUnsupported {
            name: name.to_owned(),
        });
    }

    if let Some(target_name) = name.strip_prefix("first_stage_model.") {
        return Ok(SdxlMappedTensor {
            component: SdxlConvertedComponent::Vae,
            target_name: target_name.to_owned(),
        });
    }

    if let Some(target_name) = name.strip_prefix("conditioner.embedders.0.") {
        return Ok(SdxlMappedTensor {
            component: SdxlConvertedComponent::ClipL,
            target_name: target_name.to_owned(),
        });
    }

    for prefix in ["conditioner.embedders.1.model.", "conditioner.embedders.1."] {
        if let Some(target_name) = name.strip_prefix(prefix) {
            return Ok(SdxlMappedTensor {
                component: SdxlConvertedComponent::ClipG,
                target_name: target_name.to_owned(),
            });
        }
    }

    Err(SdxlTensorMappingError::UnknownRequiredFamily {
        name: name.to_owned(),
    })
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
        let mapped = map_sdxl_checkpoint_tensor("unet.down_blocks.0.resnets.0.conv1.weight")
            .expect("diffusers key maps");

        assert_eq!(mapped.component, SdxlConvertedComponent::Unet);
        assert_eq!(mapped.target_name, "down_blocks.0.resnets.0.conv1.weight");
    }

    #[test]
    fn maps_original_text_and_vae_to_component_local_keys() {
        let clip_l = map_sdxl_checkpoint_tensor(
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        )
        .unwrap();
        let clip_g = map_sdxl_checkpoint_tensor(
            "conditioner.embedders.1.model.transformer.text_model.embeddings.token_embedding.weight",
        )
        .unwrap();
        let vae = map_sdxl_checkpoint_tensor("first_stage_model.decoder.conv_in.weight").unwrap();

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
    fn rejects_original_unet_until_full_mapping_exists() {
        let err = map_sdxl_checkpoint_tensor("model.diffusion_model.input_blocks.0.0.weight")
            .unwrap_err();

        assert!(matches!(
            err,
            SdxlTensorMappingError::OriginalUnetUnsupported { .. }
        ));
    }
}
