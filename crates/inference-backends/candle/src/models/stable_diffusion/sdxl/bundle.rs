use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::Device;
use reimagine_core::model::{ModelId, ModelRole};
use reimagine_inference::{
    BackendPayloadKey, ModelFormat, ModelSourceKind, ResolvedInferenceModelSource,
    ResolvedInferenceModelSourceSet,
};

use super::diffusion_graph::{SdxlDiffusionGraph, load_diffusion_graph};
use super::diffusion_sources::{
    SdxlDiffusionSourceError, SdxlDiffusionSources, resolve_diffusion_sources,
};
use super::text::{SdxlTextEncoderGraph, TextEncoderError};
use super::tokenizer::SdxlTokenizer;
use super::vae_sources::{SdxlVaeSourceError, SdxlVaeSources, resolve_vae_sources};
use crate::error::CandleBackendError;

/// Source validation and bundle construction errors.
///
/// The V1 path treats "missing / unreadable / unsupported artifact" as a
/// backend execution error so the runtime surfaces a useful message
/// instead of pretending the model was loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleLoadError {
    MissingArtifact {
        path: PathBuf,
    },
    NotAFile {
        path: PathBuf,
    },
    Unreadable {
        path: PathBuf,
        reason: String,
    },
    UnsupportedFormat {
        path: PathBuf,
        expected_extension: String,
        actual_extension: String,
    },
    EmptySourceSet,
    TokenizerLoadFailed {
        reason: String,
    },
    TextEncoderLoadFailed {
        reason: String,
    },
    DiffusionSourceLoadFailed {
        reason: String,
    },
    VaeSourceLoadFailed {
        reason: String,
    },
}

impl std::fmt::Display for BundleLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingArtifact { path } => {
                write!(f, "model artifact missing: {}", path.display())
            }
            Self::NotAFile { path } => {
                write!(f, "model source is not a regular file: {}", path.display())
            }
            Self::Unreadable { path, reason } => write!(
                f,
                "model artifact unreadable: {} ({})",
                path.display(),
                reason
            ),
            Self::UnsupportedFormat {
                path,
                expected_extension,
                actual_extension,
            } => write!(
                f,
                "model source format mismatch at {}: expected extension `.{expected_extension}`, got `.{actual_extension}`",
                path.display()
            ),
            Self::EmptySourceSet => write!(f, "model source set cannot be empty"),
            Self::TokenizerLoadFailed { reason } => {
                write!(f, "SDXL tokenizer load failed: {reason}")
            }
            Self::TextEncoderLoadFailed { reason } => {
                write!(f, "SDXL text encoder load failed: {reason}")
            }
            Self::DiffusionSourceLoadFailed { reason } => {
                write!(f, "SDXL diffusion source load failed: {reason}")
            }
            Self::VaeSourceLoadFailed { reason } => {
                write!(f, "SDXL VAE source load failed: {reason}")
            }
        }
    }
}

impl std::error::Error for BundleLoadError {}

impl From<BundleLoadError> for CandleBackendError {
    fn from(err: BundleLoadError) -> Self {
        CandleBackendError::InvalidRequest(err.to_string())
    }
}

/// Backend-owned SDXL bundle entry.
///
/// One bundle can back a single resolved checkpoint and exposes three
/// lightweight `BackendPayloadKey` handles for `model`, `clip`, and
/// `vae`. The same `Device` is shared across the bundle and propagated
/// to later kernels.
#[derive(Debug)]
pub struct LoadedSdxlBundle {
    pub model_id: ModelId,
    pub source_path: PathBuf,
    source_set: ResolvedInferenceModelSourceSet,
    pub source_format: ModelFormat,
    pub device: Arc<Device>,
    pub(crate) diffusion_sources: SdxlDiffusionSources,
    pub(crate) vae_sources: SdxlVaeSources,
    pub(crate) diffusion_graph: Mutex<Option<Arc<dyn SdxlDiffusionGraph>>>,
    pub tokenizer: SdxlTokenizer,
    pub text_encoder: SdxlTextEncoderGraph,
    pub model_payload_key: BackendPayloadKey,
    pub clip_payload_key: BackendPayloadKey,
    pub vae_payload_key: BackendPayloadKey,
}

impl LoadedSdxlBundle {
    /// V1 constructor with single source_path — builds a CheckpointBundle source_set internally.
    pub fn from_resolved(
        model_id: ModelId,
        source_path: PathBuf,
        format: ModelFormat,
        device: Arc<Device>,
    ) -> Result<Arc<Self>, BundleLoadError> {
        let source = ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            source_path.clone(),
            format,
        );
        let source_set = ResolvedInferenceModelSourceSet::new(source);
        Self::from_resolved_with_source_set(model_id, source_set, format, device)
    }

    /// Constructor accepting a full source_set (supports split components).
    pub fn from_resolved_with_source_set(
        model_id: ModelId,
        source_set: ResolvedInferenceModelSourceSet,
        format: ModelFormat,
        device: Arc<Device>,
    ) -> Result<Arc<Self>, BundleLoadError> {
        if source_set.sources().is_empty() {
            return Err(BundleLoadError::EmptySourceSet);
        }
        for source in source_set.sources() {
            validate_source(source.path(), source.format())?;
        }
        let primary_path = source_set
            .sources()
            .first()
            .map(|s| s.path().clone())
            .unwrap_or_default();
        let diffusion_sources = resolve_diffusion_sources(&source_set).map_err(|e| {
            BundleLoadError::DiffusionSourceLoadFailed {
                reason: e.to_string(),
            }
        })?;
        let vae_sources =
            resolve_vae_sources(&source_set).map_err(|e| BundleLoadError::VaeSourceLoadFailed {
                reason: e.to_string(),
            })?;
        let model_payload_key =
            BackendPayloadKey::new(format!("bundle:{}:model", model_id.as_str()));
        let clip_payload_key = BackendPayloadKey::new(format!("bundle:{}:clip", model_id.as_str()));
        let vae_payload_key = BackendPayloadKey::new(format!("bundle:{}:vae", model_id.as_str()));
        let tokenizer = SdxlTokenizer::from_source(&source_set, &primary_path).map_err(|e| {
            BundleLoadError::TokenizerLoadFailed {
                reason: e.to_string(),
            }
        })?;
        let text_encoder = SdxlTextEncoderGraph::load(&source_set, &primary_path, device.clone())
            .map_err(|e| BundleLoadError::TextEncoderLoadFailed {
            reason: e.to_string(),
        })?;
        Ok(Arc::new(Self {
            model_id,
            source_path: primary_path,
            source_set,
            source_format: format,
            device,
            diffusion_sources,
            vae_sources,
            diffusion_graph: Mutex::new(None),
            tokenizer,
            text_encoder,
            model_payload_key,
            clip_payload_key,
            vae_payload_key,
        }))
    }

    #[doc(hidden)]
    pub(crate) fn from_resolved_with_test_text_projection(
        model_id: ModelId,
        source_set: ResolvedInferenceModelSourceSet,
        format: ModelFormat,
        device: Arc<Device>,
    ) -> Result<Arc<Self>, BundleLoadError> {
        if source_set.sources().is_empty() {
            return Err(BundleLoadError::EmptySourceSet);
        }
        for source in source_set.sources() {
            validate_source(source.path(), source.format())?;
        }
        let primary_path = source_set
            .sources()
            .first()
            .map(|s| s.path().clone())
            .unwrap_or_default();
        let diffusion_sources = resolve_diffusion_sources(&source_set).map_err(|e| {
            BundleLoadError::DiffusionSourceLoadFailed {
                reason: e.to_string(),
            }
        })?;
        let vae_sources =
            resolve_vae_sources(&source_set).map_err(|e| BundleLoadError::VaeSourceLoadFailed {
                reason: e.to_string(),
            })?;
        let model_payload_key =
            BackendPayloadKey::new(format!("bundle:{}:model", model_id.as_str()));
        let clip_payload_key = BackendPayloadKey::new(format!("bundle:{}:clip", model_id.as_str()));
        let vae_payload_key = BackendPayloadKey::new(format!("bundle:{}:vae", model_id.as_str()));
        let tokenizer = SdxlTokenizer::from_source(&source_set, &primary_path).map_err(|e| {
            BundleLoadError::TokenizerLoadFailed {
                reason: e.to_string(),
            }
        })?;
        let text_encoder = SdxlTextEncoderGraph::load_test_projection(&source_set, device.clone())
            .map_err(|e| BundleLoadError::TextEncoderLoadFailed {
                reason: e.to_string(),
            })?;
        Ok(Arc::new(Self {
            model_id,
            source_path: primary_path,
            source_set,
            source_format: format,
            device,
            diffusion_sources,
            vae_sources,
            diffusion_graph: Mutex::new(None),
            tokenizer,
            text_encoder,
            model_payload_key,
            clip_payload_key,
            vae_payload_key,
        }))
    }

    pub(crate) fn materialize_diffusion_graph(
        &self,
    ) -> Result<Arc<dyn SdxlDiffusionGraph>, CandleBackendError> {
        let mut guard = self.diffusion_graph.lock().map_err(|_| {
            CandleBackendError::InvalidRequest(
                "diffusion.sample SDXL diffusion graph cache lock is poisoned".to_string(),
            )
        })?;
        if let Some(graph) = guard.as_ref() {
            return Ok(graph.clone());
        }
        let graph = load_diffusion_graph(&self.diffusion_sources, self.device.as_ref())?;
        *guard = Some(graph.clone());
        Ok(graph)
    }

    pub(crate) fn vae_sources(&self) -> &SdxlVaeSources {
        &self.vae_sources
    }

    #[cfg(test)]
    pub(crate) fn install_test_diffusion_graph(&self, graph: Arc<dyn SdxlDiffusionGraph>) {
        let mut guard = self.diffusion_graph.lock().unwrap();
        *guard = Some(graph);
    }
}

impl From<TextEncoderError> for BundleLoadError {
    fn from(err: TextEncoderError) -> Self {
        Self::TextEncoderLoadFailed {
            reason: err.to_string(),
        }
    }
}

impl From<SdxlDiffusionSourceError> for BundleLoadError {
    fn from(err: SdxlDiffusionSourceError) -> Self {
        Self::DiffusionSourceLoadFailed {
            reason: err.to_string(),
        }
    }
}

impl From<SdxlVaeSourceError> for BundleLoadError {
    fn from(err: SdxlVaeSourceError) -> Self {
        Self::VaeSourceLoadFailed {
            reason: err.to_string(),
        }
    }
}

use crate::graph::LoadedModelGraph;

impl LoadedModelGraph for LoadedSdxlBundle {
    fn source_set(&self) -> &ResolvedInferenceModelSourceSet {
        &self.source_set
    }

    fn component_graph_metadata(&self) -> Option<&str> {
        Some("stable_diffusion/sdxl")
    }

    fn check_compatible(&self, incoming: &ResolvedInferenceModelSourceSet) -> Result<(), String> {
        if incoming.sources().len() != self.source_set.sources().len() {
            return Err(format!(
                "source count mismatch: cached {} vs requested {}",
                self.source_set.sources().len(),
                incoming.sources().len()
            ));
        }
        for (i, (existing, incoming)) in self
            .source_set
            .sources()
            .iter()
            .zip(incoming.sources().iter())
            .enumerate()
        {
            if existing.path() != incoming.path() {
                return Err(format!(
                    "source path mismatch at index {}: cached {:?} vs requested {:?}",
                    i,
                    existing.path(),
                    incoming.path()
                ));
            }
            if existing.kind() != incoming.kind() {
                return Err(format!(
                    "source kind mismatch at index {}: cached {:?} vs requested {:?}",
                    i,
                    existing.kind(),
                    incoming.kind()
                ));
            }
            if existing.role() != incoming.role() {
                return Err(format!(
                    "source role mismatch at index {}: cached {:?} vs requested {:?}",
                    i,
                    existing.role(),
                    incoming.role()
                ));
            }
            if existing.format() != incoming.format() {
                return Err(format!(
                    "source format mismatch at index {}: cached {:?} vs requested {:?}",
                    i,
                    existing.format(),
                    incoming.format()
                ));
            }
        }
        Ok(())
    }
}

/// Validate the resolved model source path against the resolved format.
///
/// V1 only checks that the file exists, is readable, and has an
/// extension that matches the resolved [`ModelFormat`]. Header-level
/// format validation belongs to the kernel that actually parses the
/// artifact.
pub fn validate_source(path: &Path, format: ModelFormat) -> Result<(), BundleLoadError> {
    if !path.exists() {
        return Err(BundleLoadError::MissingArtifact {
            path: path.to_path_buf(),
        });
    }
    let metadata = std::fs::metadata(path).map_err(|err| BundleLoadError::Unreadable {
        path: path.to_path_buf(),
        reason: err.to_string(),
    })?;
    if !metadata.is_file() {
        return Err(BundleLoadError::NotAFile {
            path: path.to_path_buf(),
        });
    }
    // Open the file to confirm read permission and that the OS will let
    // a kernel load it later. We do not retain the handle.
    std::fs::File::open(path).map_err(|err| BundleLoadError::Unreadable {
        path: path.to_path_buf(),
        reason: err.to_string(),
    })?;

    let expected_extension = expected_extension_for(format);
    if let Some(expected) = expected_extension {
        let actual = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if actual != expected {
            return Err(BundleLoadError::UnsupportedFormat {
                path: path.to_path_buf(),
                expected_extension: expected.to_string(),
                actual_extension: actual,
            });
        }
    }
    Ok(())
}

fn expected_extension_for(format: ModelFormat) -> Option<&'static str> {
    match format {
        ModelFormat::SafeTensors => Some("safetensors"),
        ModelFormat::Gguf => Some("gguf"),
        ModelFormat::Onnx => Some("onnx"),
        ModelFormat::PyTorch => Some("pt"),
        ModelFormat::Other => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let process = std::process::id();
        std::env::temp_dir().join(format!("reimagine-sdxl-bundle-{process}-{nonce}-{counter}"))
    }

    fn write_placeholder(dir: &Path, filename: &str) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let path = dir.join(filename);
        fs::write(&path, b"placeholder").unwrap();
        path
    }

    fn write_unrelated_safetensors(dir: &Path, filename: &str) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let path = dir.join(filename);
        let tensor = Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("unet.unrelated.weight", tensor);
        candle_core::safetensors::save(&tensors, &path).unwrap();
        path
    }

    #[test]
    fn validate_source_accepts_existing_readable_safetensors_file() {
        let dir = unique_temp_dir();
        let path = write_placeholder(&dir, "model.safetensors");
        validate_source(&path, ModelFormat::SafeTensors).expect("should pass");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_source_rejects_missing_file() {
        let path = unique_temp_dir().join("does-not-exist.safetensors");
        let err = validate_source(&path, ModelFormat::SafeTensors).unwrap_err();
        assert!(matches!(err, BundleLoadError::MissingArtifact { .. }));
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn validate_source_rejects_directory() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let err = validate_source(&dir, ModelFormat::SafeTensors).unwrap_err();
        assert!(matches!(err, BundleLoadError::NotAFile { .. }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_source_rejects_extension_mismatch() {
        let dir = unique_temp_dir();
        let path = write_placeholder(&dir, "model.pt");
        let err = validate_source(&path, ModelFormat::SafeTensors).unwrap_err();
        match err {
            BundleLoadError::UnsupportedFormat {
                expected_extension,
                actual_extension,
                ..
            } => {
                assert_eq!(expected_extension, "safetensors");
                assert_eq!(actual_extension, "pt");
            }
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_source_skips_extension_check_for_other_format() {
        let dir = unique_temp_dir();
        let path = write_placeholder(&dir, "model.bin");
        validate_source(&path, ModelFormat::Other)
            .expect("Other format should not check extension");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_builds_arc_handle_with_three_payload_keys() {
        let dir = unique_temp_dir();
        let path = write_placeholder(&dir, "sdxl.safetensors");
        let model_id = ModelId::new("sdxl-base-1.0");
        let device = Arc::new(Device::Cpu);
        let source = ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            path,
            ModelFormat::SafeTensors,
        );
        let bundle = LoadedSdxlBundle::from_resolved_with_test_text_projection(
            model_id.clone(),
            ResolvedInferenceModelSourceSet::new(source),
            ModelFormat::SafeTensors,
            device,
        )
        .expect("bundle");
        assert_eq!(bundle.model_id, model_id);
        assert_eq!(bundle.source_format, ModelFormat::SafeTensors);
        assert_eq!(
            bundle.model_payload_key.as_str(),
            "bundle:sdxl-base-1.0:model"
        );
        assert_eq!(
            bundle.clip_payload_key.as_str(),
            "bundle:sdxl-base-1.0:clip"
        );
        assert_eq!(bundle.vae_payload_key.as_str(), "bundle:sdxl-base-1.0:vae");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_rejects_placeholder_checkpoint_without_text_encoder_weights() {
        let dir = unique_temp_dir();
        let path = write_placeholder(&dir, "sdxl.safetensors");
        let err = LoadedSdxlBundle::from_resolved(
            ModelId::new("sdxl-placeholder"),
            path,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .unwrap_err();

        match err {
            BundleLoadError::TextEncoderLoadFailed { reason } => {
                assert!(reason.contains("text encoder weights"), "reason: {reason}");
            }
            other => panic!("expected TextEncoderLoadFailed, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_reports_missing_text_encoder_prefix_for_valid_safetensors() {
        let dir = unique_temp_dir();
        let path = write_unrelated_safetensors(&dir, "sdxl.safetensors");
        let err = LoadedSdxlBundle::from_resolved(
            ModelId::new("sdxl-no-text-prefix"),
            path,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .unwrap_err();

        match err {
            BundleLoadError::TextEncoderLoadFailed { reason } => {
                assert!(
                    reason.contains(
                        "missing clip_l text encoder weights; no supported key prefix found"
                    ),
                    "reason: {reason}"
                );
            }
            other => panic!("expected TextEncoderLoadFailed, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_propagates_validation_error() {
        let dir = unique_temp_dir();
        let bundle = LoadedSdxlBundle::from_resolved(
            ModelId::new("missing"),
            dir.join("nope.safetensors"),
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        );
        assert!(matches!(
            bundle,
            Err(BundleLoadError::MissingArtifact { .. })
        ));
    }

    #[test]
    fn from_resolved_reports_missing_explicit_tokenizer_without_fallback() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let checkpoint_path = write_placeholder(&dir, "sdxl.safetensors");
        let text_encoder_path = write_placeholder(&dir, "clip.safetensors");
        let missing_tokenizer = dir.join("missing-tokenizer");
        let source_set = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            checkpoint_path,
            ModelFormat::SafeTensors,
        ))
        .with_source(
            ResolvedInferenceModelSource::new(
                ModelSourceKind::SplitComponent,
                ModelRole::TextEncoder,
                text_encoder_path,
                ModelFormat::SafeTensors,
            )
            .with_metadata(missing_tokenizer.display().to_string()),
        );

        let err = LoadedSdxlBundle::from_resolved_with_source_set(
            ModelId::new("sdxl-explicit-tokenizer"),
            source_set,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .unwrap_err();

        match err {
            BundleLoadError::TokenizerLoadFailed { reason } => {
                assert!(reason.contains("tokenizer resource not found"));
                assert!(reason.contains("missing-tokenizer"));
            }
            other => panic!("expected TokenizerLoadFailed, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_reports_malformed_sidecar_tokenizer_without_fallback() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let checkpoint_path = write_placeholder(&dir, "sdxl.safetensors");
        fs::write(dir.join("tokenizer.json"), b"not-json").unwrap();
        let source_set = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            checkpoint_path,
            ModelFormat::SafeTensors,
        ));

        let err = LoadedSdxlBundle::from_resolved_with_source_set(
            ModelId::new("sdxl-malformed-sidecar-tokenizer"),
            source_set,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .unwrap_err();

        match err {
            BundleLoadError::TokenizerLoadFailed { reason } => {
                assert!(reason.contains("failed to load tokenizer"));
                assert!(reason.contains("tokenizer.json"));
            }
            other => panic!("expected TokenizerLoadFailed, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_accepts_split_text_encoder_component_metadata() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let checkpoint_path = write_placeholder(&dir, "sdxl.safetensors");
        let clip_l_path = write_placeholder(&dir, "clip_l.safetensors");
        let clip_g_path = write_placeholder(&dir, "clip_g.safetensors");
        let source_set = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            checkpoint_path,
            ModelFormat::SafeTensors,
        ))
        .with_source(
            ResolvedInferenceModelSource::new(
                ModelSourceKind::SplitComponent,
                ModelRole::TextEncoder,
                clip_l_path,
                ModelFormat::SafeTensors,
            )
            .with_metadata("component=clip_l"),
        )
        .with_source(
            ResolvedInferenceModelSource::new(
                ModelSourceKind::SplitComponent,
                ModelRole::TextEncoder,
                clip_g_path,
                ModelFormat::SafeTensors,
            )
            .with_metadata("component=clip_g"),
        );

        LoadedSdxlBundle::from_resolved_with_test_text_projection(
            ModelId::new("sdxl-split-text-encoder"),
            source_set,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .expect("split text encoder component metadata should not be treated as tokenizer paths");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_reports_incomplete_split_text_encoder_sources() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let checkpoint_path = write_placeholder(&dir, "sdxl.safetensors");
        let clip_l_path = write_placeholder(&dir, "clip_l.safetensors");
        let source_set = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            checkpoint_path,
            ModelFormat::SafeTensors,
        ))
        .with_source(
            ResolvedInferenceModelSource::new(
                ModelSourceKind::SplitComponent,
                ModelRole::TextEncoder,
                clip_l_path,
                ModelFormat::SafeTensors,
            )
            .with_metadata("component=clip_l"),
        );

        let err = LoadedSdxlBundle::from_resolved_with_source_set(
            ModelId::new("sdxl-incomplete-text-encoder"),
            source_set,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .unwrap_err();

        match err {
            BundleLoadError::TextEncoderLoadFailed { reason } => {
                assert!(reason.contains("missing clip_g"), "reason: {reason}");
            }
            other => panic!("expected TextEncoderLoadFailed, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_accepts_split_vae_component_metadata() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let checkpoint_path = write_placeholder(&dir, "sdxl.safetensors");
        let vae_path = write_placeholder(&dir, "vae.safetensors");
        let source_set = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            checkpoint_path,
            ModelFormat::SafeTensors,
        ))
        .with_source(
            ResolvedInferenceModelSource::new(
                ModelSourceKind::SplitComponent,
                ModelRole::Vae,
                vae_path,
                ModelFormat::SafeTensors,
            )
            .with_metadata("component=vae"),
        );

        let bundle = LoadedSdxlBundle::from_resolved_with_test_text_projection(
            ModelId::new("sdxl-split-vae"),
            source_set,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .expect("split VAE component metadata should resolve");

        assert!(matches!(bundle.vae_sources, SdxlVaeSources::Split { .. }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_resolved_reports_missing_split_vae_metadata() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let checkpoint_path = write_placeholder(&dir, "sdxl.safetensors");
        let vae_path = write_placeholder(&dir, "vae.safetensors");
        let source_set = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            checkpoint_path,
            ModelFormat::SafeTensors,
        ))
        .with_source(ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::Vae,
            vae_path,
            ModelFormat::SafeTensors,
        ));

        let err = LoadedSdxlBundle::from_resolved_with_test_text_projection(
            ModelId::new("sdxl-missing-vae-metadata"),
            source_set,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .unwrap_err();

        match err {
            BundleLoadError::VaeSourceLoadFailed { reason } => {
                assert!(reason.contains("component=vae"), "reason: {reason}");
            }
            other => panic!("expected VaeSourceLoadFailed, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bundle_load_error_display_contains_useful_path() {
        let err = BundleLoadError::MissingArtifact {
            path: PathBuf::from("/models/missing.safetensors"),
        };
        let msg = err.to_string();
        assert!(msg.contains("missing"));
        assert!(msg.contains("/models/missing.safetensors"));
    }

    fn make_test_bundle(source_set: ResolvedInferenceModelSourceSet) -> Arc<LoadedSdxlBundle> {
        fs::create_dir_all(unique_temp_dir()).unwrap();
        for source in source_set.sources() {
            if let Some(parent) = source.path().parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(source.path(), b"dummy").unwrap();
        }
        let device = Arc::new(Device::Cpu);
        LoadedSdxlBundle::from_resolved_with_test_text_projection(
            ModelId::new("test-model"),
            source_set,
            ModelFormat::SafeTensors,
            device,
        )
        .unwrap()
    }

    #[test]
    fn check_compatible_identical_sources() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");
        fs::write(&path, b"dummy").unwrap();
        let src = ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            path,
            ModelFormat::SafeTensors,
        );
        let set = ResolvedInferenceModelSourceSet::new(src);
        let bundle = make_test_bundle(set.clone());
        assert!(bundle.check_compatible(&set).is_ok());
    }

    #[test]
    fn check_compatible_path_mismatch() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path_a = dir.join("a.safetensors");
        let path_b = dir.join("b.safetensors");
        fs::write(&path_a, b"dummy").unwrap();
        fs::write(&path_b, b"dummy").unwrap();
        let set1 = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            path_a,
            ModelFormat::SafeTensors,
        ));
        let set2 = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            path_b,
            ModelFormat::SafeTensors,
        ));
        let bundle = make_test_bundle(set1);
        assert!(bundle.check_compatible(&set2).is_err());
    }

    #[test]
    fn check_compatible_kind_mismatch() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");
        fs::write(&path, b"dummy").unwrap();
        let set1 = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            path.clone(),
            ModelFormat::SafeTensors,
        ));
        let set2 = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::CheckpointBundle,
            path,
            ModelFormat::SafeTensors,
        ));
        let bundle = make_test_bundle(set1);
        assert!(bundle.check_compatible(&set2).is_err());
    }

    #[test]
    fn check_compatible_count_mismatch() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");
        let path_unet = dir.join("unet.safetensors");
        let path_clip = dir.join("clip.safetensors");
        fs::write(&path, b"dummy").unwrap();
        fs::write(&path_unet, b"dummy").unwrap();
        fs::write(&path_clip, b"dummy").unwrap();
        let set1 = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            path,
            ModelFormat::SafeTensors,
        ));
        let set2 = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::DiffusionModel,
            path_unet,
            ModelFormat::SafeTensors,
        ))
        .with_source(ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::TextEncoder,
            path_clip,
            ModelFormat::SafeTensors,
        ));
        let bundle = make_test_bundle(set1);
        assert!(bundle.check_compatible(&set2).is_err());
    }

    #[test]
    fn materialize_diffusion_graph_reports_original_checkpoint_adapter_gap() {
        let Some(weights) = std::env::var_os("REIMAGINE_SDXL_REAL_WEIGHTS").map(PathBuf::from)
        else {
            eprintln!(
                "skipping original checkpoint materialization diagnostic; set REIMAGINE_SDXL_REAL_WEIGHTS to a local SDXL checkpoint"
            );
            return;
        };
        if !weights.exists() {
            eprintln!(
                "skipping original checkpoint materialization diagnostic; missing {}",
                weights.display()
            );
            return;
        }

        let source = ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            weights,
            ModelFormat::SafeTensors,
        );
        let source_set = ResolvedInferenceModelSourceSet::new(source);
        let bundle = LoadedSdxlBundle::from_resolved_with_test_text_projection(
            ModelId::new("sdxl-original-checkpoint"),
            source_set,
            ModelFormat::SafeTensors,
            Arc::new(Device::Cpu),
        )
        .unwrap();

        let err = bundle.materialize_diffusion_graph().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("original checkpoint diffusion layout"),
            "msg: {msg}"
        );
        assert!(msg.contains("model.diffusion_model"), "msg: {msg}");
        assert!(msg.contains("key adapter is not implemented"), "msg: {msg}");
    }
}
