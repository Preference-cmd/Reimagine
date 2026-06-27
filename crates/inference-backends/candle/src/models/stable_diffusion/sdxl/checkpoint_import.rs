use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::checkpoint_inventory::{
    SdxlCheckpointFamily, SdxlCheckpointInventory, SdxlCheckpointInventoryError,
};
use super::checkpoint_projection::{
    SdxlCheckpointProjectionError, SdxlCheckpointRole, SdxlCheckpointRoleProjection,
    project_checkpoint_role,
};
use super::checkpoint_writer::{SdxlCheckpointWriterError, write_sdxl_checkpoint_components};

pub const CANDLE_EXAMPLE_SPLIT_LAYOUT: &str = "candle_example_split";
pub const SDXL_CHECKPOINT_IMPORT_CONVERTER_VERSION: &str =
    "reimagine.candle.sdxl_checkpoint_import.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdxlCheckpointImportRequest {
    source_model_id: String,
    source_path: PathBuf,
    source_fingerprint: String,
    source_format: String,
    output_root: PathBuf,
    created_at: String,
}

impl SdxlCheckpointImportRequest {
    pub fn new(
        source_model_id: impl Into<String>,
        source_path: impl Into<PathBuf>,
        source_fingerprint: impl Into<String>,
        source_format: impl Into<String>,
        output_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            source_model_id: source_model_id.into(),
            source_path: source_path.into(),
            source_fingerprint: source_fingerprint.into(),
            source_format: source_format.into(),
            output_root: output_root.into(),
            created_at: "now".to_owned(),
        }
    }

    pub fn with_created_at(mut self, created_at: impl Into<String>) -> Self {
        self.created_at = created_at.into();
        self
    }

    pub fn source_model_id(&self) -> &str {
        &self.source_model_id
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn source_fingerprint(&self) -> &str {
        &self.source_fingerprint
    }

    pub fn source_format(&self) -> &str {
        &self.source_format
    }

    pub fn output_root(&self) -> &Path {
        &self.output_root
    }

    pub fn conversion_dir(&self) -> PathBuf {
        self.output_root
            .join(safe_path_segment(&self.source_model_id))
            .join(safe_path_segment(&self.source_fingerprint))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdxlCheckpointImportResult {
    conversion_dir: PathBuf,
    conversion_manifest_path: PathBuf,
    conversion_manifest: SdxlCheckpointConversionManifest,
    reused_existing: bool,
}

impl SdxlCheckpointImportResult {
    pub fn conversion_dir(&self) -> &Path {
        &self.conversion_dir
    }

    pub fn conversion_manifest_path(&self) -> &Path {
        &self.conversion_manifest_path
    }

    pub fn conversion_manifest(&self) -> &SdxlCheckpointConversionManifest {
        &self.conversion_manifest
    }

    pub fn component_path(&self, component: SdxlConvertedComponent) -> PathBuf {
        self.conversion_dir
            .join(self.conversion_manifest.component_path(component))
    }

    pub fn reused_existing(&self) -> bool {
        self.reused_existing
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdxlCheckpointConversionManifest {
    source_model_id: String,
    source_path: String,
    source_fingerprint: String,
    source_format: String,
    target_layout: String,
    model_series: String,
    variant: String,
    converter_version: String,
    components: BTreeMap<String, String>,
    warnings: Vec<String>,
    ignored_families: Vec<SdxlIgnoredFamily>,
    created_at: String,
}

impl SdxlCheckpointConversionManifest {
    pub fn for_request(request: &SdxlCheckpointImportRequest) -> Self {
        Self {
            source_model_id: request.source_model_id.clone(),
            source_path: request.source_path.display().to_string(),
            source_fingerprint: request.source_fingerprint.clone(),
            source_format: request.source_format.clone(),
            target_layout: CANDLE_EXAMPLE_SPLIT_LAYOUT.to_owned(),
            model_series: "stable_diffusion".to_owned(),
            variant: "sdxl".to_owned(),
            converter_version: SDXL_CHECKPOINT_IMPORT_CONVERTER_VERSION.to_owned(),
            components: SdxlConvertedComponent::all()
                .into_iter()
                .map(|component| {
                    (
                        component.manifest_key().to_owned(),
                        component.relative_path().to_owned(),
                    )
                })
                .collect(),
            warnings: Vec::new(),
            ignored_families: vec![SdxlIgnoredFamily {
                family: SdxlCheckpointFamily::IgnoredModelEma.prefix().to_owned(),
                reason: "EMA weights are not part of Candle example split execution".to_owned(),
            }],
            created_at: request.created_at.clone(),
        }
    }

    pub fn source_model_id(&self) -> &str {
        &self.source_model_id
    }

    pub fn source_fingerprint(&self) -> &str {
        &self.source_fingerprint
    }

    pub fn target_layout(&self) -> &str {
        &self.target_layout
    }

    pub fn converter_version(&self) -> &str {
        &self.converter_version
    }

    pub fn component_path(&self, component: SdxlConvertedComponent) -> &str {
        self.components
            .get(component.manifest_key())
            .expect("conversion manifest contains all known components")
    }

    fn is_compatible_with(&self, request: &SdxlCheckpointImportRequest) -> bool {
        self.source_model_id == request.source_model_id
            && self.source_fingerprint == request.source_fingerprint
            && self.source_format == request.source_format
            && self.target_layout == CANDLE_EXAMPLE_SPLIT_LAYOUT
            && self.model_series == "stable_diffusion"
            && self.variant == "sdxl"
            && self.converter_version == SDXL_CHECKPOINT_IMPORT_CONVERTER_VERSION
            && SdxlConvertedComponent::all()
                .into_iter()
                .all(|component| self.components.contains_key(component.manifest_key()))
    }

    /// Merge additional ignored families into the manifest.
    ///
    /// New entries whose `family` is not already present are appended;
    /// existing entries are preserved so caller-provided defaults keep
    /// priority.
    pub(crate) fn merge_ignored_families(
        &mut self,
        additional: impl IntoIterator<Item = SdxlIgnoredFamily>,
    ) {
        for family in additional {
            if !self
                .ignored_families
                .iter()
                .any(|f| f.family == family.family)
            {
                self.ignored_families.push(family);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdxlIgnoredFamily {
    pub family: String,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SdxlConvertedComponent {
    Unet,
    Vae,
    ClipL,
    ClipG,
}

impl SdxlConvertedComponent {
    pub fn all() -> [Self; 4] {
        [Self::Unet, Self::Vae, Self::ClipL, Self::ClipG]
    }

    pub fn manifest_key(self) -> &'static str {
        match self {
            Self::Unet => "unet",
            Self::Vae => "vae",
            Self::ClipL => "text_encoder",
            Self::ClipG => "text_encoder_2",
        }
    }

    pub fn metadata_component(self) -> &'static str {
        match self {
            Self::Unet => "unet",
            Self::Vae => "vae",
            Self::ClipL => "clip_l",
            Self::ClipG => "clip_g",
        }
    }

    pub fn relative_path(self) -> &'static str {
        match self {
            Self::Unet => "unet/model.safetensors",
            Self::Vae => "vae/model.safetensors",
            Self::ClipL => "text_encoder/model.safetensors",
            Self::ClipG => "text_encoder_2/model.safetensors",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdxlCheckpointImportError {
    Inventory { path: PathBuf, reason: String },
    Projection { path: PathBuf, reason: String },
    UnsupportedMapping { path: PathBuf, reason: String },
    WriteComponents { path: PathBuf, reason: String },
    Io { path: PathBuf, reason: String },
    ConversionManifestInvalid { path: PathBuf, reason: String },
}

impl std::fmt::Display for SdxlCheckpointImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inventory { path, reason } => write!(
                f,
                "failed to inspect SDXL checkpoint import input {}: {reason}",
                path.display()
            ),
            Self::Projection { path, reason } => write!(
                f,
                "failed to project SDXL checkpoint import input {}: {reason}",
                path.display()
            ),
            Self::UnsupportedMapping { path, reason } => write!(
                f,
                "unsupported SDXL checkpoint import mapping for {}: {reason}",
                path.display()
            ),
            Self::WriteComponents { path, reason } => write!(
                f,
                "failed to write SDXL checkpoint import components under {}: {reason}",
                path.display()
            ),
            Self::Io { path, reason } => {
                write!(
                    f,
                    "failed SDXL checkpoint import IO at {}: {reason}",
                    path.display()
                )
            }
            Self::ConversionManifestInvalid { path, reason } => write!(
                f,
                "invalid SDXL checkpoint conversion manifest at {}: {reason}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SdxlCheckpointImportError {}

impl From<SdxlCheckpointInventoryError> for SdxlCheckpointImportError {
    fn from(value: SdxlCheckpointInventoryError) -> Self {
        let path = match &value {
            SdxlCheckpointInventoryError::Io { path, .. }
            | SdxlCheckpointInventoryError::InvalidHeader { path, .. } => path.clone(),
        };
        Self::Inventory {
            path,
            reason: value.to_string(),
        }
    }
}

impl From<SdxlCheckpointProjectionError> for SdxlCheckpointImportError {
    fn from(value: SdxlCheckpointProjectionError) -> Self {
        let path = match &value {
            SdxlCheckpointProjectionError::UnsupportedLayout { path, .. } => path.clone(),
        };
        Self::Projection {
            path,
            reason: value.to_string(),
        }
    }
}

pub async fn import_sdxl_checkpoint_to_candle_example_split(
    request: SdxlCheckpointImportRequest,
) -> Result<SdxlCheckpointImportResult, SdxlCheckpointImportError> {
    if let Some(result) = load_existing_conversion(&request).await? {
        return Ok(result);
    }

    let inventory = SdxlCheckpointInventory::from_path(request.source_path())?;
    validate_supported_projection(request.source_path(), &inventory)?;

    write_conversion(&request).await
}

async fn write_conversion(
    request: &SdxlCheckpointImportRequest,
) -> Result<SdxlCheckpointImportResult, SdxlCheckpointImportError> {
    let conversion_dir = request.conversion_dir();
    let staging_dir = staging_conversion_dir(&conversion_dir);
    if tokio::fs::try_exists(&staging_dir)
        .await
        .map_err(|error| SdxlCheckpointImportError::Io {
            path: staging_dir.clone(),
            reason: error.to_string(),
        })?
    {
        tokio::fs::remove_dir_all(&staging_dir)
            .await
            .map_err(|error| SdxlCheckpointImportError::Io {
                path: staging_dir.clone(),
                reason: error.to_string(),
            })?;
    }

    let base_manifest = SdxlCheckpointConversionManifest::for_request(request);
    let write_result = {
        let staging_dir = staging_dir.clone();
        let source_path = request.source_path().to_path_buf();
        let manifest = base_manifest.clone();
        tokio::task::spawn_blocking(move || {
            let plan = write_sdxl_checkpoint_components(&source_path, &staging_dir, |component| {
                manifest.component_path(component).to_owned()
            })?;
            Ok::<_, SdxlCheckpointWriterError>(plan)
        })
        .await
        .map_err(|error| SdxlCheckpointImportError::WriteComponents {
            path: request.source_path().to_path_buf(),
            reason: error.to_string(),
        })?
    };

    let plan = match write_result {
        Ok(plan) => plan,
        Err(error) => {
            let _ = tokio::fs::remove_dir_all(&staging_dir).await;
            return Err(map_writer_error(error, request.source_path(), &staging_dir));
        }
    };

    // Build the final manifest, merging in any ignored families
    // discovered during the write pass (e.g. label_emb.*).
    let mut manifest = base_manifest;
    manifest.merge_ignored_families(plan.ignored_families().iter().cloned());

    let manifest_path = staging_dir.join("conversion.json");
    tokio::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).map_err(|error| {
            SdxlCheckpointImportError::ConversionManifestInvalid {
                path: manifest_path.clone(),
                reason: error.to_string(),
            }
        })?,
    )
    .await
    .map_err(|error| SdxlCheckpointImportError::Io {
        path: manifest_path.clone(),
        reason: error.to_string(),
    })?;

    if let Some(parent) = conversion_dir.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| SdxlCheckpointImportError::Io {
                path: parent.to_path_buf(),
                reason: error.to_string(),
            })?;
    }
    if tokio::fs::try_exists(&conversion_dir)
        .await
        .map_err(|error| SdxlCheckpointImportError::Io {
            path: conversion_dir.clone(),
            reason: error.to_string(),
        })?
    {
        tokio::fs::remove_dir_all(&conversion_dir)
            .await
            .map_err(|error| SdxlCheckpointImportError::Io {
                path: conversion_dir.clone(),
                reason: error.to_string(),
            })?;
    }
    tokio::fs::rename(&staging_dir, &conversion_dir)
        .await
        .map_err(|error| SdxlCheckpointImportError::Io {
            path: conversion_dir.clone(),
            reason: error.to_string(),
        })?;

    Ok(SdxlCheckpointImportResult {
        conversion_manifest_path: conversion_dir.join("conversion.json"),
        conversion_dir,
        conversion_manifest: manifest,
        reused_existing: false,
    })
}

fn map_writer_error(
    error: SdxlCheckpointWriterError,
    source_path: &Path,
    staging_dir: &Path,
) -> SdxlCheckpointImportError {
    match error {
        SdxlCheckpointWriterError::UnsupportedMapping { reason, .. } => {
            SdxlCheckpointImportError::UnsupportedMapping {
                path: source_path.to_path_buf(),
                reason,
            }
        }
        other => SdxlCheckpointImportError::WriteComponents {
            path: staging_dir.to_path_buf(),
            reason: other.to_string(),
        },
    }
}

async fn load_existing_conversion(
    request: &SdxlCheckpointImportRequest,
) -> Result<Option<SdxlCheckpointImportResult>, SdxlCheckpointImportError> {
    let conversion_dir = request.conversion_dir();
    let manifest_path = conversion_dir.join("conversion.json");
    if !tokio::fs::try_exists(&manifest_path)
        .await
        .map_err(|error| SdxlCheckpointImportError::Io {
            path: manifest_path.clone(),
            reason: error.to_string(),
        })?
    {
        return Ok(None);
    }

    let bytes =
        tokio::fs::read(&manifest_path)
            .await
            .map_err(|error| SdxlCheckpointImportError::Io {
                path: manifest_path.clone(),
                reason: error.to_string(),
            })?;
    let manifest =
        serde_json::from_slice::<SdxlCheckpointConversionManifest>(&bytes).map_err(|error| {
            SdxlCheckpointImportError::ConversionManifestInvalid {
                path: manifest_path.clone(),
                reason: error.to_string(),
            }
        })?;

    if !manifest.is_compatible_with(request) {
        return Err(SdxlCheckpointImportError::ConversionManifestInvalid {
            path: manifest_path,
            reason:
                "manifest does not match requested source fingerprint, target layout, or converter"
                    .to_owned(),
        });
    }

    for component in SdxlConvertedComponent::all() {
        let component_path = conversion_dir.join(manifest.component_path(component));
        if !tokio::fs::try_exists(&component_path)
            .await
            .map_err(|error| SdxlCheckpointImportError::Io {
                path: component_path.clone(),
                reason: error.to_string(),
            })?
        {
            return Err(SdxlCheckpointImportError::ConversionManifestInvalid {
                path: manifest_path,
                reason: format!(
                    "component `{}` is missing at {}",
                    component.manifest_key(),
                    component_path.display()
                ),
            });
        }
    }

    Ok(Some(SdxlCheckpointImportResult {
        conversion_dir,
        conversion_manifest_path: manifest_path,
        conversion_manifest: manifest,
        reused_existing: true,
    }))
}

fn validate_supported_projection(
    path: &Path,
    inventory: &SdxlCheckpointInventory,
) -> Result<(), SdxlCheckpointImportError> {
    for role in [
        SdxlCheckpointRole::Diffusion,
        SdxlCheckpointRole::TextEncoder,
        SdxlCheckpointRole::Vae,
    ] {
        match project_checkpoint_role(path, role, inventory)? {
            SdxlCheckpointRoleProjection::OriginalCheckpoint { .. } => {}
            SdxlCheckpointRoleProjection::DiffusersUnet => {
                if role != SdxlCheckpointRole::Diffusion {
                    return Err(SdxlCheckpointImportError::UnsupportedMapping {
                        path: path.to_path_buf(),
                        reason: "diffusers-style projection is only valid for the UNet component"
                            .to_owned(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn safe_path_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => ch,
            _ => '_',
        })
        .collect()
}

fn staging_conversion_dir(conversion_dir: &Path) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    conversion_dir.with_extension(format!("tmp-{}-{nonce}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    #[cfg(test)]
    use candle_core::Device;

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "reimagine-sdxl-checkpoint-import-{name}-{}-{nonce}",
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

    fn write_real_safetensors(path: &Path, names: &[&str]) {
        use candle_core::{DType, Device, Tensor};
        use std::collections::HashMap;

        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut tensors = HashMap::new();
        for (idx, name) in names.iter().enumerate() {
            let tensor = Tensor::from_vec(vec![idx as f32], (1,), &Device::Cpu).unwrap();
            assert_eq!(tensor.dtype(), DType::F32);
            tensors.insert((*name).to_owned(), tensor);
        }
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    fn complete_diffusers_split_names() -> Vec<&'static str> {
        let mut names = vec![
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.resnets.0.norm1.weight",
            "down_blocks.1.attentions.0.proj_in.weight",
            "down_blocks.0.downsamplers.0.conv.weight",
            "mid_block.resnets.0.norm1.weight",
            "mid_block.attentions.0.proj_in.weight",
            "up_blocks.0.resnets.0.conv_shortcut.weight",
            "up_blocks.0.attentions.0.proj_in.weight",
            "up_blocks.0.upsamplers.0.conv.weight",
            "conv_norm_out.weight",
            "conv_out.weight",
        ];
        names.extend(required_clip_source_names("conditioner.embedders.0."));
        names.extend(required_clip_source_names("conditioner.embedders.1.model."));
        names.extend(required_vae_source_names());
        names
    }

    fn complete_original_checkpoint_names() -> Vec<&'static str> {
        let mut names = vec![
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
            "model.diffusion_model.input_blocks.4.1.proj_in.weight",
            "model.diffusion_model.input_blocks.3.0.op.weight",
            "model.diffusion_model.middle_block.0.in_layers.0.weight",
            "model.diffusion_model.middle_block.1.proj_in.weight",
            "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
            "model.diffusion_model.output_blocks.0.1.proj_in.weight",
            "model.diffusion_model.output_blocks.2.2.conv.weight",
            "model.diffusion_model.out.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.label_emb.0.0.weight",
        ];
        names.extend(required_clip_source_names("conditioner.embedders.0."));
        names.extend(required_clip_source_names("conditioner.embedders.1.model."));
        names.extend(required_vae_source_names());
        names
    }

    fn required_clip_source_names(prefix: &'static str) -> Vec<&'static str> {
        let targets = [
            "transformer.text_model.embeddings.token_embedding.weight",
            "transformer.text_model.embeddings.position_embedding.weight",
            "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
            "transformer.text_model.encoder.layers.0.self_attn.v_proj.weight",
            "transformer.text_model.encoder.layers.0.self_attn.out_proj.weight",
            "transformer.text_model.encoder.layers.0.layer_norm1.weight",
            "transformer.text_model.final_layer_norm.weight",
        ];
        targets
            .into_iter()
            .map(|target| Box::leak(format!("{prefix}{target}").into_boxed_str()) as &'static str)
            .collect()
    }

    fn required_vae_source_names() -> Vec<&'static str> {
        let targets = [
            "encoder.conv_in.weight",
            "encoder.conv_out.weight",
            "encoder.conv_norm_out.weight",
            "decoder.conv_in.weight",
            "decoder.conv_out.weight",
            "decoder.conv_norm_out.weight",
            "quant_conv.weight",
            "post_quant_conv.weight",
        ];
        targets
            .into_iter()
            .map(|target| {
                Box::leak(format!("first_stage_model.{target}").into_boxed_str()) as &'static str
            })
            .collect()
    }

    fn request(base: &Path, source: &Path) -> SdxlCheckpointImportRequest {
        SdxlCheckpointImportRequest::new(
            "sdxl/base 1.0",
            source,
            "sha256:abcd/ef",
            "safetensors",
            base.join("models/converted"),
        )
        .with_created_at("2026-06-26T00:00:00Z")
    }

    #[test]
    fn conversion_manifest_records_candle_example_split_components() {
        let base = temp_dir("manifest-shape");
        let source = base.join("models/checkpoints/sdxl.safetensors");
        let request = request(&base, &source);

        let manifest = SdxlCheckpointConversionManifest::for_request(&request);

        assert_eq!(manifest.source_model_id(), "sdxl/base 1.0");
        assert_eq!(manifest.source_fingerprint(), "sha256:abcd/ef");
        assert_eq!(manifest.target_layout(), CANDLE_EXAMPLE_SPLIT_LAYOUT);
        assert_eq!(
            manifest.converter_version(),
            SDXL_CHECKPOINT_IMPORT_CONVERTER_VERSION
        );
        assert_eq!(
            manifest.component_path(SdxlConvertedComponent::Unet),
            "unet/model.safetensors"
        );
        assert_eq!(
            manifest.component_path(SdxlConvertedComponent::ClipG),
            "text_encoder_2/model.safetensors"
        );
    }

    #[test]
    fn staging_conversion_paths_are_siblings_and_unique() {
        let base = temp_dir("staging-path");
        let conversion_dir = base.join("models/converted/sdxl_base/sha256_abcd");

        let first = staging_conversion_dir(&conversion_dir);
        std::thread::sleep(std::time::Duration::from_nanos(1));
        let second = staging_conversion_dir(&conversion_dir);

        assert_eq!(first.parent(), conversion_dir.parent());
        assert_eq!(second.parent(), conversion_dir.parent());
        assert_ne!(first, conversion_dir);
        assert_ne!(second, conversion_dir);
        assert_ne!(first, second);
        assert!(
            first
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("sha256_abcd.tmp-")
        );
    }

    #[tokio::test]
    async fn existing_complete_conversion_is_reused() {
        let base = temp_dir("existing");
        let source = base.join("models/checkpoints/sdxl.safetensors");
        let request = request(&base, &source);
        let conversion_dir = request.conversion_dir();
        tokio::fs::create_dir_all(&conversion_dir).await.unwrap();
        let manifest = SdxlCheckpointConversionManifest::for_request(&request);
        for component in SdxlConvertedComponent::all() {
            let path = conversion_dir.join(manifest.component_path(component));
            tokio::fs::create_dir_all(path.parent().unwrap())
                .await
                .unwrap();
            tokio::fs::write(path, b"component").await.unwrap();
        }
        tokio::fs::write(
            conversion_dir.join("conversion.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .await
        .unwrap();

        let result = import_sdxl_checkpoint_to_candle_example_split(request)
            .await
            .unwrap();

        assert!(result.reused_existing());
        assert_eq!(result.conversion_dir(), conversion_dir);

        let _ = tokio::fs::remove_dir_all(base).await;
    }

    #[tokio::test]
    async fn original_checkpoint_converts_split_components_with_details() {
        let base = temp_dir("original-convert");
        let source = base.join("models/checkpoints/sdxl.safetensors");
        tokio::fs::create_dir_all(source.parent().unwrap())
            .await
            .unwrap();
        write_real_safetensors(&source, &complete_original_checkpoint_names());
        let request = request(&base, &source);

        let result = import_sdxl_checkpoint_to_candle_example_split(request)
            .await
            .unwrap();

        assert!(!result.reused_existing());
        assert!(result.conversion_manifest_path().is_file());
        assert!(
            result
                .component_path(SdxlConvertedComponent::Unet)
                .is_file()
        );
        assert!(
            result
                .component_path(SdxlConvertedComponent::ClipL)
                .is_file()
        );
        assert!(
            result
                .component_path(SdxlConvertedComponent::ClipG)
                .is_file()
        );
        assert!(result.component_path(SdxlConvertedComponent::Vae).is_file());

        // Verify the conversion manifest records label_emb.* as an ignored family.
        let manifest = result.conversion_manifest();
        assert!(
            manifest
                .ignored_families
                .iter()
                .any(|f| f.family.starts_with("model.diffusion_model.label_emb")),
            "ignored_families should contain label_emb.*: {manifest:#?}"
        );

        // Verify the mapped UNet content.
        let unet = candle_core::safetensors::load(
            result.component_path(SdxlConvertedComponent::Unet),
            &Device::Cpu,
        )
        .unwrap();
        assert!(unet.contains_key("conv_in.weight"));
        assert!(unet.contains_key("time_embedding.linear_1.weight"));

        let _ = tokio::fs::remove_dir_all(base).await;
    }

    #[tokio::test]
    async fn original_checkpoint_with_compvis_vae_routes_through_vae_key_mapping() {
        // End-to-end exercise of real-inference/07a1: the source checkpoint
        // uses compvis/LDM VAE keys (`encoder.down.0.block.0.norm1.weight`
        // etc.) and the import pipeline must produce a diffusers-layout
        // `vae/model.safetensors` whose keys Candle's `AutoEncoderKL`
        // can load.
        let base = temp_dir("original-compvis-vae");
        let source = base.join("models/checkpoints/sdxl.safetensors");
        tokio::fs::create_dir_all(source.parent().unwrap())
            .await
            .unwrap();
        let mut names = complete_original_checkpoint_names();
        // Drop the diffusers-format VAE keys from the standard fixture and
        // supply a compvis-format VAE surface instead.
        names.retain(|name| !name.starts_with("first_stage_model."));
        names.extend(compvis_vae_source_names_for_import());
        write_real_safetensors(&source, &names);
        let request = request(&base, &source);

        let result = import_sdxl_checkpoint_to_candle_example_split(request)
            .await
            .expect("compvis VAE keys should round-trip through the import pipeline");

        assert!(!result.reused_existing());
        assert!(result.component_path(SdxlConvertedComponent::Vae).is_file());

        let vae = candle_core::safetensors::load(
            result.component_path(SdxlConvertedComponent::Vae),
            &Device::Cpu,
        )
        .expect("vae/model.safetensors must be readable as safetensors");

        // Resnet + downsample + mid resnet + mid attention + norm_out
        // targets must all be present in diffusers layout.
        assert!(vae.contains_key("encoder.down_blocks.0.resnets.0.norm1.weight"));
        assert!(vae.contains_key("encoder.down_blocks.0.downsamplers.0.conv.weight"));
        assert!(vae.contains_key("encoder.mid_block.resnets.0.norm1.weight"));
        assert!(vae.contains_key("encoder.mid_block.attentions.0.group_norm.weight"));
        assert!(vae.contains_key("encoder.conv_norm_out.weight"));
        // Symmetric decoder mappings.
        assert!(vae.contains_key("decoder.up_blocks.0.resnets.0.norm1.weight"));
        assert!(vae.contains_key("decoder.mid_block.resnets.0.norm1.weight"));
        assert!(vae.contains_key("decoder.conv_norm_out.weight"));
        // quant_conv / post_quant_conv unchanged.
        assert!(vae.contains_key("quant_conv.weight"));
        assert!(vae.contains_key("post_quant_conv.weight"));
        // Compvis-style keys must not appear in the output.
        assert!(!vae.contains_key("encoder.down.0.block.0.norm1.weight"));
        assert!(!vae.contains_key("encoder.norm_out.weight"));
        assert!(!vae.contains_key("encoder.mid.attn_1.norm.weight"));

        let _ = tokio::fs::remove_dir_all(base).await;
    }

    fn compvis_vae_source_names_for_import() -> Vec<&'static str> {
        vec![
            "first_stage_model.encoder.conv_in.weight",
            "first_stage_model.encoder.conv_out.weight",
            "first_stage_model.encoder.norm_out.weight",
            "first_stage_model.encoder.down.0.block.0.norm1.weight",
            "first_stage_model.encoder.down.0.block.0.conv1.weight",
            "first_stage_model.encoder.down.0.block.0.norm2.weight",
            "first_stage_model.encoder.down.0.block.0.conv2.weight",
            "first_stage_model.encoder.down.0.downsample.conv.weight",
            "first_stage_model.encoder.mid.block_1.norm1.weight",
            "first_stage_model.encoder.mid.attn_1.norm.weight",
            "first_stage_model.decoder.conv_in.weight",
            "first_stage_model.decoder.conv_out.weight",
            "first_stage_model.decoder.norm_out.weight",
            "first_stage_model.decoder.up.0.block.0.norm1.weight",
            "first_stage_model.decoder.mid.block_1.norm1.weight",
            "first_stage_model.quant_conv.weight",
            "first_stage_model.post_quant_conv.weight",
        ]
    }

    #[tokio::test]
    async fn unsupported_original_block_index_still_fails() {
        let base = temp_dir("unsupported-block-idx");
        let source = base.join("models/checkpoints/sdxl.safetensors");
        tokio::fs::create_dir_all(source.parent().unwrap())
            .await
            .unwrap();
        // Provide all 6 required original checkpoint families so
        // projection passes, then use an unsupported block index
        // to trigger a mapping error.
        write_real_safetensors(
            &source,
            &[
                "model.diffusion_model.input_blocks.99.0.weight",
                "model.diffusion_model.middle_block.0.in_layers.0.weight",
                "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
                "model.diffusion_model.time_embed.0.weight",
                "model.diffusion_model.out.2.weight",
                "model.diffusion_model.label_emb.0.0.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "first_stage_model.decoder.conv_in.weight",
            ],
        );
        let request = request(&base, &source);
        let conversion_dir = request.conversion_dir();

        let error = import_sdxl_checkpoint_to_candle_example_split(request)
            .await
            .unwrap_err();

        assert!(
            matches!(error, SdxlCheckpointImportError::UnsupportedMapping { .. }),
            "{error:?}"
        );
        assert!(
            !tokio::fs::try_exists(conversion_dir.join("conversion.json"))
                .await
                .unwrap()
        );

        let _ = tokio::fs::remove_dir_all(base).await;
    }

    #[tokio::test]
    async fn supported_checkpoint_writes_split_components_and_conversion_manifest() {
        let base = temp_dir("supported");
        let source = base.join("models/checkpoints/sdxl.safetensors");
        write_real_safetensors(&source, &complete_diffusers_split_names());
        let first_request = request(&base, &source);

        let result = import_sdxl_checkpoint_to_candle_example_split(first_request)
            .await
            .unwrap();

        assert!(!result.reused_existing());
        assert!(result.conversion_manifest_path().is_file());
        assert!(
            result
                .component_path(SdxlConvertedComponent::Unet)
                .is_file()
        );
        assert!(
            result
                .component_path(SdxlConvertedComponent::ClipL)
                .is_file()
        );
        assert!(
            result
                .component_path(SdxlConvertedComponent::ClipG)
                .is_file()
        );
        assert!(result.component_path(SdxlConvertedComponent::Vae).is_file());

        let second = import_sdxl_checkpoint_to_candle_example_split(request(&base, &source))
            .await
            .unwrap();
        assert!(second.reused_existing());

        let _ = tokio::fs::remove_dir_all(base).await;
    }

    #[tokio::test]
    async fn unknown_checkpoint_family_fails_precisely() {
        let base = temp_dir("unknown-family");
        let source = base.join("models/checkpoints/sdxl.safetensors");
        tokio::fs::create_dir_all(source.parent().unwrap())
            .await
            .unwrap();
        write_header_only_safetensors(
            &source,
            &[
                "model.diffusion_model.input_blocks.0.0.weight",
                "model.diffusion_model.middle_block.1.weight",
                "model.diffusion_model.output_blocks.0.0.weight",
                "model.diffusion_model.time_embed.0.weight",
                "model.diffusion_model.out.2.weight",
                "model.diffusion_model.label_emb.0.0.weight",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
                "first_stage_model.decoder.conv_in.weight",
                "surprise.family.weight",
            ],
        );

        let error = import_sdxl_checkpoint_to_candle_example_split(request(&base, &source))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("surprise.family."));

        let _ = tokio::fs::remove_dir_all(base).await;
    }
}
