use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::Device;
use reimagine_core::model::ModelId;
use reimagine_inference::{BackendPayloadKey, ModelFormat, ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet};

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
            source_path.clone(),
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
        for source in source_set.sources() {
            validate_source(source.path(), format)?;
        }
        let primary_path = source_set.sources().first()
            .map(|s| s.path().clone())
            .unwrap_or_default();
        let model_payload_key = BackendPayloadKey::new(format!("bundle:{}:model", model_id.as_str()));
        let clip_payload_key = BackendPayloadKey::new(format!("bundle:{}:clip", model_id.as_str()));
        let vae_payload_key = BackendPayloadKey::new(format!("bundle:{}:vae", model_id.as_str()));
        Ok(Arc::new(Self {
            model_id,
            source_path: primary_path,
            source_set,
            source_format: format,
            device,
            model_payload_key,
            clip_payload_key,
            vae_payload_key,
        }))
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
        for (i, (existing, incoming)) in self.source_set.sources().iter()
            .zip(incoming.sources().iter()).enumerate()
        {
            if existing.path() != incoming.path() {
                return Err(format!(
                    "source path mismatch at index {}: cached {:?} vs requested {:?}",
                    i, existing.path(), incoming.path()
                ));
            }
            if existing.kind() != incoming.kind() {
                return Err(format!(
                    "source kind mismatch at index {}: cached {:?} vs requested {:?}",
                    i, existing.kind(), incoming.kind()
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
        let bundle = LoadedSdxlBundle::from_resolved(
            model_id.clone(),
            path,
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
    fn bundle_load_error_display_contains_useful_path() {
        let err = BundleLoadError::MissingArtifact {
            path: PathBuf::from("/models/missing.safetensors"),
        };
        let msg = err.to_string();
        assert!(msg.contains("missing"));
        assert!(msg.contains("/models/missing.safetensors"));
    }
}
