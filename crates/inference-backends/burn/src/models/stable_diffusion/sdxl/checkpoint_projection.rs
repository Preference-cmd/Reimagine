//! Burn-native SDXL checkpoint role projection.
//!
//! Validates that a checkpoint's tensor families can produce each required
//! component role and returns the tensor-to-role mapping.

use std::collections::BTreeSet;

use super::checkpoint_inventory::{
    BurnCheckpointFamily, BurnCheckpointInventory, BurnCheckpointInventoryError,
};

/// Projected component role from a source checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum BurnCheckpointRole {
    Diffusion,
    TextEncoder,
    TextEncoder2,
    Vae,
}

/// Result of projecting a checkpoint into component roles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BurnCheckpointProjection {
    /// Which component types this checkpoint can produce.
    pub roles: BTreeSet<BurnCheckpointRole>,
    /// Human-readable projection notes or warnings.
    pub notes: Vec<String>,
}

/// Project roles from an already-built inventory.
pub(crate) fn project_from_inventory(
    inventory: &BurnCheckpointInventory,
) -> Result<BurnCheckpointProjection, BurnProjectionError> {
    let mut roles = BTreeSet::new();
    let mut notes = Vec::new();
    let mut missing = Vec::new();

    // Diffusion: original LDM format OR diffusers-format UNet
    let has_original_diff = inventory.contains(BurnCheckpointFamily::OriginalDiffusionInputBlocks)
        || inventory.contains(BurnCheckpointFamily::OriginalDiffusionMiddleBlock)
        || inventory.contains(BurnCheckpointFamily::OriginalDiffusionOutputBlocks)
        || inventory.contains(BurnCheckpointFamily::OriginalDiffusionTimeEmbed)
        || inventory.contains(BurnCheckpointFamily::OriginalDiffusionOut)
        || inventory.contains(BurnCheckpointFamily::OriginalDiffusionLabelEmb);
    let has_diffusers = inventory.contains(BurnCheckpointFamily::DiffusersUnet);
    if has_original_diff || has_diffusers {
        roles.insert(BurnCheckpointRole::Diffusion);
        if has_original_diff {
            notes.push("diffusion: original LDM format".to_string());
        }
        if has_diffusers {
            notes.push("diffusion: diffusers-format UNet".to_string());
        }
    } else {
        missing.push("diffusion (no UNet families found)".to_string());
    }

    // Text encoder: CLIP-L (embedders 0)
    if inventory.contains(BurnCheckpointFamily::OriginalTextEmbedders0) {
        roles.insert(BurnCheckpointRole::TextEncoder);
        notes.push("text_encoder: CLIP-L".to_string());
    } else {
        missing.push("text_encoder (no conditioner.embedders.0 keys)".to_string());
    }

    // Text encoder 2: OpenCLIP-G (embedders 1) — optional for SDXL
    if inventory.contains(BurnCheckpointFamily::OriginalTextEmbedders1) {
        roles.insert(BurnCheckpointRole::TextEncoder2);
        notes.push("text_encoder_2: OpenCLIP-G".to_string());
    } else {
        notes.push("text_encoder_2: absent (single-CLIP layout)".to_string());
    }

    // VAE
    if inventory.contains(BurnCheckpointFamily::OriginalVaeFirstStageModel) {
        roles.insert(BurnCheckpointRole::Vae);
    } else {
        missing.push("vae (no first_stage_model keys)".to_string());
    }

    // Unknown families — warn but don't fail
    let unknown = inventory.unknown_families();
    if !unknown.is_empty() {
        notes.push(format!("ignored unknown families: {:?}", unknown));
    }

    if !missing.is_empty() {
        return Err(BurnProjectionError::MissingRoles(missing));
    }

    Ok(BurnCheckpointProjection { roles, notes })
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) enum BurnProjectionError {
    /// Checkpoint could not be read or header was invalid.
    Inventory(BurnCheckpointInventoryError),
    /// Required component roles are missing from the checkpoint.
    MissingRoles(Vec<String>),
}

impl From<BurnCheckpointInventoryError> for BurnProjectionError {
    fn from(err: BurnCheckpointInventoryError) -> Self {
        Self::Inventory(err)
    }
}

impl std::fmt::Display for BurnProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inventory(err) => write!(f, "{err}"),
            Self::MissingRoles(missing) => {
                write!(
                    f,
                    "checkpoint is missing required roles: {}",
                    missing.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for BurnProjectionError {}

/// Number of text encoder components in the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextEncoderCount {
    One,
    Two,
}

pub(crate) fn text_encoder_count(projection: &BurnCheckpointProjection) -> TextEncoderCount {
    if projection.roles.contains(&BurnCheckpointRole::TextEncoder2) {
        TextEncoderCount::Two
    } else {
        TextEncoderCount::One
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::models::stable_diffusion::sdxl::checkpoint_inventory::BurnCheckpointInventory;

    fn temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "burn-checkpoint-projection-{}-{nonce}",
            std::process::id()
        ))
    }

    fn write_header_only_safetensors(path: &Path, names: &[&str]) {
        let entries = names
            .iter()
            .map(|name| {
                format!("\"{name}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}")
            })
            .collect::<Vec<_>>()
            .join(",");
        let header = format!("{{{entries}}}");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn full_sdxl_checkpoint_projects_all_4_roles() {
        let names = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.0.resblocks.0.weight",
            "model.diffusion_model.output_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "conditioner.embedders.1.model.text_projection.weight",
            "first_stage_model.decoder.conv_in.weight",
        ];
        let inv = BurnCheckpointInventory::from_names(names);
        let projection = project_from_inventory(&inv).unwrap();

        assert!(projection.roles.contains(&BurnCheckpointRole::Diffusion));
        assert!(projection.roles.contains(&BurnCheckpointRole::TextEncoder));
        assert!(projection.roles.contains(&BurnCheckpointRole::TextEncoder2));
        assert!(projection.roles.contains(&BurnCheckpointRole::Vae));
        assert_eq!(text_encoder_count(&projection), TextEncoderCount::Two);
    }

    #[test]
    fn single_clip_checkpoint_projects_3_roles() {
        let names = [
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.resnets.0.conv1.weight",
            "up_blocks.0.resnets.0.conv1.weight",
            "mid_block.resnets.0.conv1.weight",
            "conv_norm_out.weight",
            "conv_out.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ];
        let inv = BurnCheckpointInventory::from_names(names);
        let projection = project_from_inventory(&inv).unwrap();

        assert!(projection.roles.contains(&BurnCheckpointRole::Diffusion));
        assert!(projection.roles.contains(&BurnCheckpointRole::TextEncoder));
        assert!(!projection.roles.contains(&BurnCheckpointRole::TextEncoder2));
        assert!(projection.roles.contains(&BurnCheckpointRole::Vae));
        assert_eq!(text_encoder_count(&projection), TextEncoderCount::One);
    }

    #[test]
    fn missing_diffusion_is_rejected() {
        let names = [
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ];
        let inv = BurnCheckpointInventory::from_names(names);
        let err = project_from_inventory(&inv).unwrap_err();
        assert!(
            matches!(&err, BurnProjectionError::MissingRoles(m) if m.iter().any(|s| s.contains("diffusion")))
        );
    }

    #[test]
    fn missing_vae_is_rejected() {
        let names = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.out.2.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        ];
        let inv = BurnCheckpointInventory::from_names(names);
        let err = project_from_inventory(&inv).unwrap_err();
        assert!(
            matches!(&err, BurnProjectionError::MissingRoles(m) if m.iter().any(|s| s.contains("vae")))
        );
    }

    #[test]
    fn projects_from_file() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.safetensors");
        write_header_only_safetensors(
            &path,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
                "model.diffusion_model.time_embed.0.weight",
                "model.diffusion_model.out.2.weight",
                "model.diffusion_model.label_emb.0.0.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "first_stage_model.decoder.conv_in.weight",
                "first_stage_model.encoder.conv_in.weight",
            ],
        );

        let inv = BurnCheckpointInventory::from_path(&path).unwrap();
        let projection = project_from_inventory(&inv).unwrap();
        assert!(projection.roles.contains(&BurnCheckpointRole::Diffusion));
        assert!(projection.roles.contains(&BurnCheckpointRole::TextEncoder));
        assert!(projection.roles.contains(&BurnCheckpointRole::Vae));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn diffusers_unet_is_detected() {
        let names = [
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.resnets.0.conv1.weight",
            "up_blocks.0.resnets.0.conv1.weight",
            "mid_block.resnets.0.conv1.weight",
            "conv_norm_out.weight",
            "conv_out.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ];
        let inv = BurnCheckpointInventory::from_names(names);
        let projection = project_from_inventory(&inv).unwrap();
        assert!(projection.roles.contains(&BurnCheckpointRole::Diffusion));
    }
}
