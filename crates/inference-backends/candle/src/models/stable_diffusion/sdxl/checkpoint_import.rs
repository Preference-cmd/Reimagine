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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdxlIgnoredFamily {
    family: String,
    reason: String,
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

    Err(SdxlCheckpointImportError::UnsupportedMapping {
        path: request.source_path().to_path_buf(),
        reason: "original SDXL checkpoint tensor remapping to Candle example split component keys is not implemented yet".to_owned(),
    })
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
                return Err(SdxlCheckpointImportError::UnsupportedMapping {
                    path: path.to_path_buf(),
                    reason: "diffusers-style UNet-only source is already a split component candidate and is not an original checkpoint bundle import input".to_owned(),
                });
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

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
    async fn unsupported_original_mapping_fails_before_creating_conversion_manifest() {
        let base = temp_dir("unsupported");
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
