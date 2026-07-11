//! Burn-native SDXL checkpoint inventory.
//!
//! Reads only the safetensors header (no tensor data) and classifies each
//! tensor by family prefix. Used by [`BurnCheckpointProjection`] to determine
//! whether a source checkpoint can produce all required component roles.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Recognized tensor family prefix in an SDXL safetensors checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum BurnCheckpointFamily {
    /// `model.diffusion_model.input_blocks.*`
    OriginalDiffusionInputBlocks,
    /// `model.diffusion_model.middle_block.*`
    OriginalDiffusionMiddleBlock,
    /// `model.diffusion_model.output_blocks.*`
    OriginalDiffusionOutputBlocks,
    /// `model.diffusion_model.time_embed.*`
    OriginalDiffusionTimeEmbed,
    /// `model.diffusion_model.out.*`
    OriginalDiffusionOut,
    /// `model.diffusion_model.label_emb.*`
    OriginalDiffusionLabelEmb,
    /// Diffusers-format UNet keys: `unet.*`, `conv_in.*`, `down_blocks.*`, etc.
    DiffusersUnet,
    /// `conditioner.embedders.0.*` — CLIP-L text encoder
    OriginalTextEmbedders0,
    /// `conditioner.embedders.1.*` or `conditioner.embedders.1.model.*` — OpenCLIP-G
    OriginalTextEmbedders1,
    /// `first_stage_model.*` — VAE
    OriginalVaeFirstStageModel,
    /// `model_ema.*` — ignored (exponential moving average weights)
    IgnoredModelEma,
}

impl BurnCheckpointFamily {
    /// String prefix used to classify a tensor key into this family.
    #[allow(dead_code)]
    pub(crate) fn prefix(self) -> &'static str {
        match self {
            Self::OriginalDiffusionInputBlocks => "model.diffusion_model.input_blocks.",
            Self::OriginalDiffusionMiddleBlock => "model.diffusion_model.middle_block.",
            Self::OriginalDiffusionOutputBlocks => "model.diffusion_model.output_blocks.",
            Self::OriginalDiffusionTimeEmbed => "model.diffusion_model.time_embed.",
            Self::OriginalDiffusionOut => "model.diffusion_model.out.",
            Self::OriginalDiffusionLabelEmb => "model.diffusion_model.label_emb.",
            Self::DiffusersUnet => "diffusers-unet",
            Self::OriginalTextEmbedders0 => "conditioner.embedders.0.",
            Self::OriginalTextEmbedders1 => "conditioner.embedders.1.",
            Self::OriginalVaeFirstStageModel => "first_stage_model.",
            Self::IgnoredModelEma => "model_ema.",
        }
    }
}

/// Inventory of tensor families found in a checkpoint's safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BurnCheckpointInventory {
    family_counts: BTreeMap<BurnCheckpointFamily, usize>,
    unknown_families: Vec<String>,
}

impl BurnCheckpointInventory {
    /// Build inventory from an iterator of tensor names (no I/O).
    pub(crate) fn from_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut family_counts = BTreeMap::new();
        let mut unknown_families = BTreeSet::new();

        for name in names {
            if let Some(family) = classify_name(name) {
                *family_counts.entry(family).or_insert(0) += 1;
            } else if let Some(prefix) = unknown_family_prefix(name) {
                unknown_families.insert(prefix);
            }
        }

        Self {
            family_counts,
            unknown_families: unknown_families.into_iter().collect(),
        }
    }

    /// Read the safetensors header from `path` and build an inventory.
    ///
    /// Only reads the 8-byte length prefix and JSON header — tensor data is
    /// never loaded.
    pub(crate) fn from_path(path: &Path) -> Result<Self, BurnCheckpointInventoryError> {
        let names = read_safetensors_header_names(path)?;
        Ok(Self::from_names(names.iter().map(String::as_str)))
    }

    pub(crate) fn count(&self, family: BurnCheckpointFamily) -> usize {
        self.family_counts.get(&family).copied().unwrap_or(0)
    }

    pub(crate) fn contains(&self, family: BurnCheckpointFamily) -> bool {
        self.count(family) > 0
    }

    pub(crate) fn unknown_families(&self) -> &[String] {
        &self.unknown_families
    }

    /// Returns true when the checkpoint has at least one recognized family
    /// for each of the base SDXL component roles: diffusion, text encoder
    /// (CLIP-L or CLIP-G), and VAE.
    #[allow(dead_code)]
    pub(crate) fn has_sdxl_roles(&self) -> bool {
        let has_diff = self.contains(BurnCheckpointFamily::OriginalDiffusionInputBlocks)
            || self.contains(BurnCheckpointFamily::DiffusersUnet);
        let has_text = self.contains(BurnCheckpointFamily::OriginalTextEmbedders0)
            || self.contains(BurnCheckpointFamily::OriginalTextEmbedders1);
        let has_vae = self.contains(BurnCheckpointFamily::OriginalVaeFirstStageModel);
        has_diff && has_text && has_vae
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Error reading or parsing a checkpoint's safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BurnCheckpointInventoryError {
    Io { path: PathBuf, reason: String },
    InvalidHeader { path: PathBuf, reason: String },
}

impl std::fmt::Display for BurnCheckpointInventoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, reason } => {
                write!(f, "failed to read checkpoint inventory from {}: {reason}", path.display())
            }
            Self::InvalidHeader { path, reason } => {
                write!(
                    f,
                    "invalid checkpoint safetensors header at {}: {reason}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for BurnCheckpointInventoryError {}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

fn classify_name(name: &str) -> Option<BurnCheckpointFamily> {
    // Original LDM format (diffusion)
    if name.starts_with("model.diffusion_model.input_blocks.") {
        return Some(BurnCheckpointFamily::OriginalDiffusionInputBlocks);
    }
    if name.starts_with("model.diffusion_model.middle_block.") {
        return Some(BurnCheckpointFamily::OriginalDiffusionMiddleBlock);
    }
    if name.starts_with("model.diffusion_model.output_blocks.") {
        return Some(BurnCheckpointFamily::OriginalDiffusionOutputBlocks);
    }
    if name.starts_with("model.diffusion_model.time_embed.") {
        return Some(BurnCheckpointFamily::OriginalDiffusionTimeEmbed);
    }
    if name.starts_with("model.diffusion_model.out.") {
        return Some(BurnCheckpointFamily::OriginalDiffusionOut);
    }
    if name.starts_with("model.diffusion_model.label_emb.") {
        return Some(BurnCheckpointFamily::OriginalDiffusionLabelEmb);
    }

    // Diffusers-format UNet
    if name.starts_with("unet.")
        || name.starts_with("diffusion_model.")
        || name == "conv_in.weight"
        || name == "conv_in.bias"
        || name.starts_with("time_embedding.")
        || name.starts_with("down_blocks.")
        || name.starts_with("up_blocks.")
        || name.starts_with("mid_block.")
        || name.starts_with("conv_norm_out.")
        || name.starts_with("conv_out.")
        || name.starts_with("class_embedding.")
    {
        return Some(BurnCheckpointFamily::DiffusersUnet);
    }

    // Text encoders — order matters: embedders.1 before embedders.0 for
    // prefix matching? No — same length prefix, but we check 1 first to
    // avoid confusion. Actually both start the same way, but we can only
    // match one. Since they're at the same level, we check .1 first.
    if name.starts_with("conditioner.embedders.1.") {
        return Some(BurnCheckpointFamily::OriginalTextEmbedders1);
    }
    if name.starts_with("conditioner.embedders.0.") {
        return Some(BurnCheckpointFamily::OriginalTextEmbedders0);
    }

    // VAE
    if name.starts_with("first_stage_model.") {
        return Some(BurnCheckpointFamily::OriginalVaeFirstStageModel);
    }

    // Ignored
    if name.starts_with("model_ema.") {
        return Some(BurnCheckpointFamily::IgnoredModelEma);
    }

    None
}

fn unknown_family_prefix(name: &str) -> Option<String> {
    let mut segments = name.split('.');
    let first = segments.next()?;
    let second = segments.next();
    match second {
        Some(second) if segments.next().is_some() => Some(format!("{first}.{second}.")),
        Some(_) | None => Some(format!("{first}.")),
    }
}

// ---------------------------------------------------------------------------
// Safetensors header reader (data-free)
// ---------------------------------------------------------------------------

fn read_safetensors_header_names(path: &Path) -> Result<Vec<String>, BurnCheckpointInventoryError> {
    let mut file = std::fs::File::open(path).map_err(|err| BurnCheckpointInventoryError::Io {
        path: path.to_path_buf(),
        reason: err.to_string(),
    })?;

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|err| BurnCheckpointInventoryError::Io {
            path: path.to_path_buf(),
            reason: format!("failed to read header length: {err}"),
        })?;
    let header_len = u64::from_le_bytes(len_bytes);

    let file_len = file
        .metadata()
        .map_err(|err| BurnCheckpointInventoryError::Io {
            path: path.to_path_buf(),
            reason: format!("failed to inspect file metadata: {err}"),
        })?
        .len();

    if header_len > file_len.saturating_sub(8) {
        return Err(BurnCheckpointInventoryError::InvalidHeader {
            path: path.to_path_buf(),
            reason: format!("header length {header_len} exceeds file size {file_len}"),
        });
    }
    if header_len > 64 * 1024 * 1024 {
        return Err(BurnCheckpointInventoryError::InvalidHeader {
            path: path.to_path_buf(),
            reason: format!("header too large to inspect safely: {header_len} bytes"),
        });
    }

    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|err| BurnCheckpointInventoryError::Io {
            path: path.to_path_buf(),
            reason: format!("failed to read header bytes: {err}"),
        })?;

    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|err| {
        BurnCheckpointInventoryError::InvalidHeader {
            path: path.to_path_buf(),
            reason: err.to_string(),
        }
    })?;

    let object = value.as_object().ok_or_else(|| {
        BurnCheckpointInventoryError::InvalidHeader {
            path: path.to_path_buf(),
            reason: "header is not a JSON object".to_string(),
        }
    })?;

    Ok(object
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "burn-checkpoint-inventory-{}-{nonce}",
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
    fn classifies_original_checkpoint_families() {
        let inv = BurnCheckpointInventory::from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "conditioner.embedders.1.model.text_projection.weight",
            "first_stage_model.decoder.conv_in.weight",
            "model_ema.decay",
        ]);

        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionInputBlocks));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionMiddleBlock));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionOutputBlocks));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionTimeEmbed));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionOut));
        assert!(inv.contains(BurnCheckpointFamily::OriginalDiffusionLabelEmb));
        assert!(inv.contains(BurnCheckpointFamily::OriginalTextEmbedders0));
        assert!(inv.contains(BurnCheckpointFamily::OriginalTextEmbedders1));
        assert!(inv.contains(BurnCheckpointFamily::OriginalVaeFirstStageModel));
        assert!(inv.contains(BurnCheckpointFamily::IgnoredModelEma));
        assert!(inv.unknown_families().is_empty(), "{inv:?}");
        assert!(inv.has_sdxl_roles());
    }

    #[test]
    fn classifies_diffusers_unet() {
        let inv = BurnCheckpointInventory::from_names([
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.resnets.0.conv1.weight",
            "up_blocks.0.resnets.0.conv1.weight",
            "mid_block.resnets.0.conv1.weight",
            "conv_norm_out.weight",
            "conv_out.weight",
        ]);

        assert!(inv.contains(BurnCheckpointFamily::DiffusersUnet));
        // Single CLIP text encoder (SD1.5/FLUX style)
        assert!(!inv.contains(BurnCheckpointFamily::OriginalTextEmbedders0));
        assert!(!inv.contains(BurnCheckpointFamily::OriginalTextEmbedders1));
        assert!(!inv.has_sdxl_roles());
    }

    #[test]
    fn reports_unknown_families() {
        let inv = BurnCheckpointInventory::from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "mystery.branch.weight",
            "mystery.branch.bias",
            "another_thing.value",
        ]);

        assert_eq!(
            inv.unknown_families(),
            &["another_thing.".to_string(), "mystery.branch.".to_string()]
        );
    }

    #[test]
    fn reads_inventory_from_header_only_safetensors() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.safetensors");
        write_header_only_safetensors(
            &path,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "conditioner.embedders.1.model.text_projection.weight",
                "first_stage_model.decoder.conv_in.weight",
            ],
        );

        let inv = BurnCheckpointInventory::from_path(&path).unwrap();
        assert_eq!(
            inv.count(BurnCheckpointFamily::OriginalDiffusionInputBlocks),
            1
        );
        assert_eq!(
            inv.count(BurnCheckpointFamily::OriginalTextEmbedders0),
            1
        );
        assert_eq!(
            inv.count(BurnCheckpointFamily::OriginalTextEmbedders1),
            1
        );
        assert_eq!(
            inv.count(BurnCheckpointFamily::OriginalVaeFirstStageModel),
            1
        );
        assert!(inv.unknown_families().is_empty(), "{inv:?}");
        assert!(inv.has_sdxl_roles());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_clip_checkpoint_has_embeddder0_but_not_embeder1() {
        let inv = BurnCheckpointInventory::from_names([
            "conv_in.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ]);
        assert!(inv.contains(BurnCheckpointFamily::OriginalTextEmbedders0));
        assert!(!inv.contains(BurnCheckpointFamily::OriginalTextEmbedders1));
    }
}
