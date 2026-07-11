//! Public entry point for Burn-native SDXL checkpoint import.
//!
//! Orchestrates: inventory → projection → writer.
//! Callers: `ModelService` and the `POST /models/convert` HTTP handler.

use std::path::Path;

use super::checkpoint_inventory::BurnCheckpointInventory;
use super::checkpoint_projection::{project_from_inventory, BurnCheckpointProjection};
use super::checkpoint_writer::write_real_checkpoint_components;
use super::conversion::{BurnSdxlConversionError, BurnSdxlConversionReport};

/// Import a single SDXL safetensors checkpoint into the Burn-native component
/// layout under `model_root`.
///
/// Pipeline: read header → inventory → projection → write components.
pub fn execute_real_burn_sdxl_checkpoint_import(
    source_path: &Path,
    model_id: &str,
    model_root: &Path,
) -> Result<BurnSdxlConversionReport, BurnSdxlConversionError> {
    let inventory = BurnCheckpointInventory::from_path(source_path)?;
    let projection = project_from_inventory(&inventory)?;
    write_real_checkpoint_components(source_path, model_id, model_root, &projection)
}

/// Validate without writing — returns the projection or an error.
#[allow(dead_code)]
pub(crate) fn validate_checkpoint_for_import(
    source_path: &Path,
) -> Result<BurnCheckpointProjection, BurnSdxlConversionError> {
    let inventory = BurnCheckpointInventory::from_path(source_path)?;
    let projection = project_from_inventory(&inventory)?;
    Ok(projection)
}

// ---------------------------------------------------------------------------
// Error conversions
// ---------------------------------------------------------------------------

impl From<super::checkpoint_inventory::BurnCheckpointInventoryError>
    for BurnSdxlConversionError
{
    fn from(err: super::checkpoint_inventory::BurnCheckpointInventoryError) -> Self {
        use super::checkpoint_inventory::BurnCheckpointInventoryError;
        match err {
            BurnCheckpointInventoryError::Io { path, reason } => BurnSdxlConversionError::InvalidComponentSet {
                reason: format!("I/O error at `{}`: {reason}", path.display()),
            },
            BurnCheckpointInventoryError::InvalidHeader { path, reason } => BurnSdxlConversionError::InvalidComponentSet {
                reason: format!("invalid header at `{}`: {reason}", path.display()),
            },
        }
    }
}

impl From<super::checkpoint_projection::BurnProjectionError> for BurnSdxlConversionError {
    fn from(err: super::checkpoint_projection::BurnProjectionError) -> Self {
        use super::checkpoint_projection::BurnProjectionError;
        match err {
            BurnProjectionError::Inventory(inv_err) => inv_err.into(),
            BurnProjectionError::MissingRoles(missing) => BurnSdxlConversionError::InvalidComponentSet {
                reason: format!("checkpoint is missing required roles: {}", missing.join(", ")),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::stable_diffusion::sdxl::checkpoint_inventory::BurnCheckpointFamily;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    use safetensors::tensor::{Dtype, View};

    #[derive(Clone)]
    struct TestView {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for TestView {
        fn dtype(&self) -> Dtype {
            self.dtype
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> std::borrow::Cow<'_, [u8]> {
            std::borrow::Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("burn-ckpt-import-{}-{nonce}", std::process::id()))
    }

    fn make_ckpt(path: &Path, names: &[&str]) {
        let mut map: BTreeMap<String, TestView> = BTreeMap::new();
        for name in names {
            map.insert(
                name.to_string(),
                TestView {
                    dtype: Dtype::F32,
                    shape: vec![1],
                    data: vec![0u8; 4],
                },
            );
        }
        safetensors::serialize_to_file(map, None, path).unwrap();
    }

    #[test]
    fn end_to_end_single_clip() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("m.safetensors");
        make_ckpt(&ckpt, &[
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.out.2.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ]);

        let mr = dir.join("models");
        let report = execute_real_burn_sdxl_checkpoint_import(&ckpt, "tm", &mr).unwrap();
        assert!(mr.join("diffusion_model/tm.safetensors").is_file());
        assert!(mr.join("vae/tm.safetensors").is_file());
        assert!(mr.join("clip/tm.safetensors").is_file());
        assert_eq!(report.output_components.len(), 3);
        assert_eq!(report.mapped_tensor_count, 4);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn end_to_end_dual_clip() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("sdxl.safetensors");
        make_ckpt(&ckpt, &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.time_embed.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "conditioner.embedders.1.model.text_projection.weight",
            "first_stage_model.decoder.conv_in.weight",
        ]);

        let mr = dir.join("models");
        let report = execute_real_burn_sdxl_checkpoint_import(&ckpt, "sdxl-t", &mr).unwrap();
        assert!(mr.join("diffusion_model/sdxl-t.safetensors").is_file());
        assert!(mr.join("vae/sdxl-t.safetensors").is_file());
        assert!(mr.join("clip/sdxl-t/clip-l.safetensors").is_file());
        assert!(mr.join("clip/sdxl-t/clip-g.safetensors").is_file());
        assert_eq!(report.output_components.len(), 4);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn validate_rejects_incomplete() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("partial.safetensors");
        make_ckpt(&ckpt, &[
            "model.diffusion_model.time_embed.0.weight",
        ]);

        let err = validate_checkpoint_for_import(&ckpt).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("vae") || msg.contains("text_encoder"), "error: {msg}");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn inventory_detects_all_sdxl_families() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("full.safetensors");
        let mut map: BTreeMap<String, TestView> = BTreeMap::new();
        for name in [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.0.resblocks.0.weight",
            "model.diffusion_model.output_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "conditioner.embedders.1.model.text_projection.weight",
            "first_stage_model.decoder.conv_in.weight",
        ] {
            map.insert(
                name.to_string(),
                TestView {
                    dtype: Dtype::F32,
                    shape: vec![1],
                    data: vec![0u8; 4],
                },
            );
        }
        safetensors::serialize_to_file(map, None, &ckpt).unwrap();

        let inv = BurnCheckpointInventory::from_path(&ckpt).unwrap();
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionInputBlocks));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionMiddleBlock));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionOutputBlocks));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionTimeEmbed));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionOut));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionLabelEmb));
        assert!(inv.contains(BurnCheckpointFamily::OriginalTextEmbedders0));
        assert!(inv.contains(BurnCheckpointFamily::OriginalTextEmbedders1));
        assert!(inv.contains(BurnCheckpointFamily::OriginalVaeFirstStageModel));
        assert!(inv.unknown_families().is_empty());

        fs::remove_dir_all(&dir).unwrap();
    }
}
