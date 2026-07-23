use std::path::{Path, PathBuf};

use super::checkpoint_inventory::{SdxlCheckpointFamily, SdxlCheckpointInventory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdxlCheckpointRole {
    Diffusion,
    TextEncoder,
    Vae,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlCheckpointRoleProjection {
    DiffusersUnet,
    OriginalCheckpoint {
        recognized_families: Vec<SdxlCheckpointFamily>,
    },
}

pub(crate) fn project_checkpoint_role(
    path: &Path,
    role: SdxlCheckpointRole,
    inventory: &SdxlCheckpointInventory,
) -> Result<SdxlCheckpointRoleProjection, SdxlCheckpointProjectionError> {
    match role {
        SdxlCheckpointRole::Diffusion => project_diffusion_role(path, inventory),
        SdxlCheckpointRole::TextEncoder => project_text_role(path, inventory),
        SdxlCheckpointRole::Vae => project_vae_role(path, inventory),
    }
}

fn project_diffusion_role(
    path: &Path,
    inventory: &SdxlCheckpointInventory,
) -> Result<SdxlCheckpointRoleProjection, SdxlCheckpointProjectionError> {
    reject_unknown_families(path, SdxlCheckpointRole::Diffusion, inventory)?;

    let required = [
        SdxlCheckpointFamily::OriginalDiffusionInputBlocks,
        SdxlCheckpointFamily::OriginalDiffusionMiddleBlock,
        SdxlCheckpointFamily::OriginalDiffusionOutputBlocks,
        SdxlCheckpointFamily::OriginalDiffusionTimeEmbed,
        SdxlCheckpointFamily::OriginalDiffusionOut,
        SdxlCheckpointFamily::OriginalDiffusionLabelEmb,
    ];
    let has_original_diffusion = required.iter().any(|family| inventory.contains(*family));
    let has_diffusers_unet = inventory.contains(SdxlCheckpointFamily::DiffusersUnet);

    if has_original_diffusion && has_diffusers_unet {
        return Err(SdxlCheckpointProjectionError::UnsupportedLayout {
            path: path.to_path_buf(),
            role: SdxlCheckpointRole::Diffusion,
            family: format!(
                "{} + {}",
                SdxlCheckpointFamily::OriginalDiffusionInputBlocks.prefix(),
                SdxlCheckpointFamily::DiffusersUnet.prefix()
            ),
            reason:
                "mixes original checkpoint and diffusers UNet prefixes; refusing ambiguous layout"
                    .to_string(),
        });
    }

    if has_diffusers_unet {
        return Ok(SdxlCheckpointRoleProjection::DiffusersUnet);
    }

    for family in required {
        if !inventory.contains(family) {
            return Err(SdxlCheckpointProjectionError::UnsupportedLayout {
                path: path.to_path_buf(),
                role: SdxlCheckpointRole::Diffusion,
                family: family.prefix().to_string(),
                reason: "is required for original SDXL diffusion checkpoint projection".to_string(),
            });
        }
    }

    Ok(SdxlCheckpointRoleProjection::OriginalCheckpoint {
        recognized_families: required.to_vec(),
    })
}

fn project_text_role(
    path: &Path,
    inventory: &SdxlCheckpointInventory,
) -> Result<SdxlCheckpointRoleProjection, SdxlCheckpointProjectionError> {
    reject_unknown_families(path, SdxlCheckpointRole::TextEncoder, inventory)?;

    if inventory.contains(SdxlCheckpointFamily::OriginalTextConditionerEmbedders) {
        return Ok(SdxlCheckpointRoleProjection::OriginalCheckpoint {
            recognized_families: vec![SdxlCheckpointFamily::OriginalTextConditionerEmbedders],
        });
    }
    Err(SdxlCheckpointProjectionError::UnsupportedLayout {
        path: path.to_path_buf(),
        role: SdxlCheckpointRole::TextEncoder,
        family: SdxlCheckpointFamily::OriginalTextConditionerEmbedders
            .prefix()
            .to_string(),
        reason: "is required for SDXL text encoder checkpoint projection".to_string(),
    })
}

fn project_vae_role(
    path: &Path,
    inventory: &SdxlCheckpointInventory,
) -> Result<SdxlCheckpointRoleProjection, SdxlCheckpointProjectionError> {
    reject_unknown_families(path, SdxlCheckpointRole::Vae, inventory)?;

    if inventory.contains(SdxlCheckpointFamily::OriginalVaeFirstStageModel) {
        return Ok(SdxlCheckpointRoleProjection::OriginalCheckpoint {
            recognized_families: vec![SdxlCheckpointFamily::OriginalVaeFirstStageModel],
        });
    }
    Err(SdxlCheckpointProjectionError::UnsupportedLayout {
        path: path.to_path_buf(),
        role: SdxlCheckpointRole::Vae,
        family: SdxlCheckpointFamily::OriginalVaeFirstStageModel
            .prefix()
            .to_string(),
        reason: "is required for SDXL VAE checkpoint projection".to_string(),
    })
}

fn reject_unknown_families(
    path: &Path,
    role: SdxlCheckpointRole,
    inventory: &SdxlCheckpointInventory,
) -> Result<(), SdxlCheckpointProjectionError> {
    if let Some(unknown) = inventory.unknown_families().first() {
        return Err(SdxlCheckpointProjectionError::UnsupportedLayout {
            path: path.to_path_buf(),
            role,
            family: unknown.clone(),
            reason:
                "is not mapped, ignored, or supported for Candle-private SDXL checkpoint projection"
                    .to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlCheckpointProjectionError {
    UnsupportedLayout {
        path: PathBuf,
        role: SdxlCheckpointRole,
        family: String,
        reason: String,
    },
}

impl std::fmt::Display for SdxlCheckpointProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLayout {
                path,
                role,
                family,
                reason,
            } => write!(
                f,
                "unsupported SDXL checkpoint layout for {role:?} at {}: family `{family}` {reason}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SdxlCheckpointProjectionError {}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        SdxlCheckpointProjectionError, SdxlCheckpointRole, SdxlCheckpointRoleProjection,
        project_checkpoint_role,
    };
    use crate::models::stable_diffusion::sdxl::checkpoint_inventory::{
        SdxlCheckpointFamily, SdxlCheckpointInventory,
    };

    #[test]
    fn diffusion_projection_recognizes_label_emb_as_required_original_family() {
        let inventory = SdxlCheckpointInventory::from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.label_emb.0.0.weight",
        ]);

        let projection = project_checkpoint_role(
            Path::new("/models/sdxl.safetensors"),
            SdxlCheckpointRole::Diffusion,
            &inventory,
        )
        .unwrap();

        assert_eq!(
            projection,
            SdxlCheckpointRoleProjection::OriginalCheckpoint {
                recognized_families: vec![
                    SdxlCheckpointFamily::OriginalDiffusionInputBlocks,
                    SdxlCheckpointFamily::OriginalDiffusionMiddleBlock,
                    SdxlCheckpointFamily::OriginalDiffusionOutputBlocks,
                    SdxlCheckpointFamily::OriginalDiffusionTimeEmbed,
                    SdxlCheckpointFamily::OriginalDiffusionOut,
                    SdxlCheckpointFamily::OriginalDiffusionLabelEmb,
                ],
            }
        );
    }

    #[test]
    fn diffusion_projection_rejects_unknown_families_with_path_and_prefix() {
        let inventory = SdxlCheckpointInventory::from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "odd.family.weight",
        ]);

        let err = project_checkpoint_role(
            Path::new("/models/original.safetensors"),
            SdxlCheckpointRole::Diffusion,
            &inventory,
        )
        .unwrap_err();

        assert_eq!(
            err,
            SdxlCheckpointProjectionError::UnsupportedLayout {
                path: Path::new("/models/original.safetensors").to_path_buf(),
                role: SdxlCheckpointRole::Diffusion,
                family: "odd.family.".to_string(),
                reason: "is not mapped, ignored, or supported for Candle-private SDXL checkpoint projection".to_string(),
            }
        );
        assert!(err.to_string().contains("/models/original.safetensors"));
        assert!(err.to_string().contains("odd.family."));
    }

    #[test]
    fn diffusion_projection_allows_diffusers_layout_with_unrelated_diffusers_keys() {
        let inventory = SdxlCheckpointInventory::from_names([
            "down_blocks.0.resnets.0.conv1.weight",
            "time_embedding.linear_1.weight",
        ]);

        let projection = project_checkpoint_role(
            Path::new("/models/unet.safetensors"),
            SdxlCheckpointRole::Diffusion,
            &inventory,
        )
        .unwrap();

        assert_eq!(projection, SdxlCheckpointRoleProjection::DiffusersUnet);
    }

    #[test]
    fn diffusion_projection_rejects_mixed_original_and_diffusers_layout() {
        let inventory = SdxlCheckpointInventory::from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "down_blocks.0.resnets.0.conv1.weight",
        ]);

        let err = project_checkpoint_role(
            Path::new("/models/ambiguous.safetensors"),
            SdxlCheckpointRole::Diffusion,
            &inventory,
        )
        .unwrap_err();

        assert!(err.to_string().contains("ambiguous layout"));
        assert!(
            err.to_string()
                .contains("model.diffusion_model.input_blocks.")
        );
        assert!(err.to_string().contains("diffusers-unet"));
    }

    #[test]
    fn vae_projection_requires_first_stage_family() {
        let inventory = SdxlCheckpointInventory::from_names([
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        ]);

        let err = project_checkpoint_role(
            Path::new("/models/original.safetensors"),
            SdxlCheckpointRole::Vae,
            &inventory,
        )
        .unwrap_err();

        assert!(err.to_string().contains("family `first_stage_model.`"));
    }
}
