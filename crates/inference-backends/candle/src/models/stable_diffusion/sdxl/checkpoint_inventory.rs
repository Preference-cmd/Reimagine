use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SdxlCheckpointFamily {
    OriginalDiffusionInputBlocks,
    OriginalDiffusionMiddleBlock,
    OriginalDiffusionOutputBlocks,
    OriginalDiffusionTimeEmbed,
    OriginalDiffusionOut,
    OriginalDiffusionLabelEmb,
    OriginalTextConditionerEmbedders,
    OriginalVaeFirstStageModel,
    DiffusersUnet,
    IgnoredModelEma,
}

impl SdxlCheckpointFamily {
    pub(crate) fn prefix(self) -> &'static str {
        match self {
            Self::OriginalDiffusionInputBlocks => "model.diffusion_model.input_blocks.",
            Self::OriginalDiffusionMiddleBlock => "model.diffusion_model.middle_block.",
            Self::OriginalDiffusionOutputBlocks => "model.diffusion_model.output_blocks.",
            Self::OriginalDiffusionTimeEmbed => "model.diffusion_model.time_embed.",
            Self::OriginalDiffusionOut => "model.diffusion_model.out.",
            Self::OriginalDiffusionLabelEmb => "model.diffusion_model.label_emb.",
            Self::OriginalTextConditionerEmbedders => "conditioner.embedders.",
            Self::OriginalVaeFirstStageModel => "first_stage_model.",
            Self::DiffusersUnet => "diffusers-unet",
            Self::IgnoredModelEma => "model_ema.",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdxlCheckpointInventory {
    family_counts: BTreeMap<SdxlCheckpointFamily, usize>,
    unknown_families: Vec<String>,
}

impl SdxlCheckpointInventory {
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

    pub(crate) fn from_path(path: &Path) -> Result<Self, SdxlCheckpointInventoryError> {
        let names = read_safetensors_header_names(path)?;
        Ok(Self::from_names(names.iter().map(String::as_str)))
    }

    pub(crate) fn count(&self, family: SdxlCheckpointFamily) -> usize {
        self.family_counts.get(&family).copied().unwrap_or(0)
    }

    pub(crate) fn contains(&self, family: SdxlCheckpointFamily) -> bool {
        self.count(family) > 0
    }

    pub(crate) fn unknown_families(&self) -> &[String] {
        &self.unknown_families
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlCheckpointInventoryError {
    Io { path: PathBuf, reason: String },
    InvalidHeader { path: PathBuf, reason: String },
}

impl std::fmt::Display for SdxlCheckpointInventoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, reason } => write!(
                f,
                "failed to read SDXL checkpoint inventory from {}: {reason}",
                path.display()
            ),
            Self::InvalidHeader { path, reason } => write!(
                f,
                "invalid SDXL checkpoint safetensors header at {}: {reason}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SdxlCheckpointInventoryError {}

fn classify_name(name: &str) -> Option<SdxlCheckpointFamily> {
    if name.starts_with("model.diffusion_model.input_blocks.") {
        return Some(SdxlCheckpointFamily::OriginalDiffusionInputBlocks);
    }
    if name.starts_with("model.diffusion_model.middle_block.") {
        return Some(SdxlCheckpointFamily::OriginalDiffusionMiddleBlock);
    }
    if name.starts_with("model.diffusion_model.output_blocks.") {
        return Some(SdxlCheckpointFamily::OriginalDiffusionOutputBlocks);
    }
    if name.starts_with("model.diffusion_model.time_embed.") {
        return Some(SdxlCheckpointFamily::OriginalDiffusionTimeEmbed);
    }
    if name.starts_with("model.diffusion_model.out.") {
        return Some(SdxlCheckpointFamily::OriginalDiffusionOut);
    }
    if name.starts_with("model.diffusion_model.label_emb.") {
        return Some(SdxlCheckpointFamily::OriginalDiffusionLabelEmb);
    }
    if name.starts_with("conditioner.embedders.") {
        return Some(SdxlCheckpointFamily::OriginalTextConditionerEmbedders);
    }
    if name.starts_with("first_stage_model.") {
        return Some(SdxlCheckpointFamily::OriginalVaeFirstStageModel);
    }
    if name.starts_with("model_ema.") {
        return Some(SdxlCheckpointFamily::IgnoredModelEma);
    }
    if name.starts_with("unet.")
        || name.starts_with("diffusion_model.")
        || name == "conv_in.weight"
        || name.starts_with("time_embedding.")
        || name.starts_with("down_blocks.")
        || name.starts_with("up_blocks.")
        || name.starts_with("mid_block.")
        || name.starts_with("conv_norm_out.")
        || name.starts_with("conv_out.")
        || name.starts_with("class_embedding.")
    {
        return Some(SdxlCheckpointFamily::DiffusersUnet);
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

fn read_safetensors_header_names(path: &Path) -> Result<Vec<String>, SdxlCheckpointInventoryError> {
    let mut file = std::fs::File::open(path).map_err(|err| SdxlCheckpointInventoryError::Io {
        path: path.to_path_buf(),
        reason: err.to_string(),
    })?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|err| SdxlCheckpointInventoryError::Io {
            path: path.to_path_buf(),
            reason: format!("failed to read header length: {err}"),
        })?;
    let header_len = u64::from_le_bytes(len_bytes);
    let file_len = file
        .metadata()
        .map_err(|err| SdxlCheckpointInventoryError::Io {
            path: path.to_path_buf(),
            reason: format!("failed to inspect file metadata: {err}"),
        })?
        .len();
    if header_len > file_len.saturating_sub(8) {
        return Err(SdxlCheckpointInventoryError::InvalidHeader {
            path: path.to_path_buf(),
            reason: format!("header length {header_len} exceeds file size {file_len}"),
        });
    }
    if header_len > 64 * 1024 * 1024 {
        return Err(SdxlCheckpointInventoryError::InvalidHeader {
            path: path.to_path_buf(),
            reason: format!("header is too large to inspect safely: {header_len} bytes"),
        });
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|err| SdxlCheckpointInventoryError::Io {
            path: path.to_path_buf(),
            reason: format!("failed to read header bytes: {err}"),
        })?;
    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|err| {
        SdxlCheckpointInventoryError::InvalidHeader {
            path: path.to_path_buf(),
            reason: err.to_string(),
        }
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| SdxlCheckpointInventoryError::InvalidHeader {
            path: path.to_path_buf(),
            reason: "header is not a JSON object".to_string(),
        })?;
    Ok(object
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::{SdxlCheckpointFamily, SdxlCheckpointInventory};

    fn unique_temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "reimagine-sdxl-checkpoint-inventory-{}-{nonce}",
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
    fn classifies_original_checkpoint_families_without_loading_payloads() {
        let inventory = SdxlCheckpointInventory::from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
            "model_ema.decay",
        ]);

        assert!(inventory.contains(SdxlCheckpointFamily::OriginalDiffusionInputBlocks));
        assert!(inventory.contains(SdxlCheckpointFamily::OriginalDiffusionMiddleBlock));
        assert!(inventory.contains(SdxlCheckpointFamily::OriginalDiffusionOutputBlocks));
        assert!(inventory.contains(SdxlCheckpointFamily::OriginalDiffusionTimeEmbed));
        assert!(inventory.contains(SdxlCheckpointFamily::OriginalDiffusionOut));
        assert!(inventory.contains(SdxlCheckpointFamily::OriginalDiffusionLabelEmb));
        assert!(inventory.contains(SdxlCheckpointFamily::OriginalTextConditionerEmbedders));
        assert!(inventory.contains(SdxlCheckpointFamily::OriginalVaeFirstStageModel));
        assert!(inventory.contains(SdxlCheckpointFamily::IgnoredModelEma));
        assert!(inventory.unknown_families().is_empty(), "{inventory:?}");
    }

    #[test]
    fn reports_unknown_tensor_prefix_families_precisely() {
        let inventory = SdxlCheckpointInventory::from_names([
            "model.diffusion_model.input_blocks.0.0.weight",
            "mystery.branch.weight",
            "mystery.branch.bias",
            "another_thing.value",
        ]);

        assert_eq!(
            inventory.unknown_families(),
            &["another_thing.".to_string(), "mystery.branch.".to_string()]
        );
    }

    #[test]
    fn reads_inventory_from_header_only_safetensors() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.safetensors");
        write_header_only_safetensors(
            &path,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
                "model.diffusion_model.label_emb.0.0.weight",
                "conditioner.embedders.1.model.transformer.text_model.embeddings.token_embedding.weight",
                "first_stage_model.decoder.conv_in.weight",
            ],
        );

        let inventory = SdxlCheckpointInventory::from_path(&path).unwrap();

        assert_eq!(
            inventory.count(SdxlCheckpointFamily::OriginalDiffusionInputBlocks),
            1
        );
        assert_eq!(
            inventory.count(SdxlCheckpointFamily::OriginalDiffusionLabelEmb),
            1
        );
        assert_eq!(
            inventory.count(SdxlCheckpointFamily::OriginalTextConditionerEmbedders),
            1
        );
        assert_eq!(
            inventory.count(SdxlCheckpointFamily::OriginalVaeFirstStageModel),
            1
        );
        assert!(inventory.unknown_families().is_empty(), "{inventory:?}");

        let _ = fs::remove_dir_all(&dir);
    }
}
