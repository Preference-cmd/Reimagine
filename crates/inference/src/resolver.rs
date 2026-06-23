//! Model resolution capability consumed by executors.
//!
//! The model resolver is a capability injected into executor adapters
//! so they can translate a workflow-level [`ModelRef`] into a
//! [`ResolvedInferenceModel`] without depending on `model-manager`.

use std::path::PathBuf;

use reimagine_core::model::{ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant};

use crate::inference_error::InferenceError;

pub mod source;
pub use source::{ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet};

/// Resolved, backend-neutral model metadata.
///
/// This shape reuses stable [`reimagine_core`] semantic types but
/// deliberately excludes model-manager implementation details
/// (scan roots, classification rules, fingerprint state, manifest
/// diagnostics).
///
/// A backend adapter uses the `source_path` to locate the file on
/// disk and the semantic fields to decide which kernel to load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedInferenceModel {
    model_id: ModelId,
    series: ModelSeries,
    variant: ModelVariant,
    role: ModelRole,
    source_path: PathBuf,
    format: ModelFormat,
    metadata: Option<String>,
    source_set: Option<ResolvedInferenceModelSourceSet>,
}

impl ResolvedInferenceModel {
    pub fn new(
        model_id: ModelId,
        series: ModelSeries,
        variant: ModelVariant,
        role: ModelRole,
        source_path: impl Into<PathBuf>,
        format: ModelFormat,
    ) -> Self {
        Self {
            model_id,
            series,
            variant,
            role,
            source_path: source_path.into(),
            format,
            metadata: None,
            source_set: None,
        }
    }

    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    pub fn source_set(&self) -> Option<&ResolvedInferenceModelSourceSet> {
        self.source_set.as_ref()
    }

    pub fn with_source_set(mut self, source_set: ResolvedInferenceModelSourceSet) -> Self {
        self.source_set = Some(source_set);
        self
    }

    pub fn to_checkpoint_bundle_source_set(&self) -> ResolvedInferenceModelSourceSet {
        let mut source = ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            self.source_path.clone(),
        );
        if let Some(ref meta) = self.metadata {
            source = source.with_metadata(meta.as_str());
        }
        ResolvedInferenceModelSourceSet::new(source)
    }

    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    pub fn series(&self) -> &ModelSeries {
        &self.series
    }

    pub fn variant(&self) -> &ModelVariant {
        &self.variant
    }

    pub fn role(&self) -> ModelRole {
        self.role
    }

    pub fn source_path(&self) -> &std::path::Path {
        &self.source_path
    }

    pub fn format(&self) -> ModelFormat {
        self.format
    }

    pub fn metadata(&self) -> Option<&str> {
        self.metadata.as_deref()
    }
}

/// Stable, backend-neutral model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    SafeTensors,
    PyTorch,
    Onnx,
    Gguf,
    Other,
}

/// Model resolver capability.
///
/// Executable adapters (e.g. `builtin.checkpoint_loader`) call
/// `resolve` to translate a workflow `ModelRef` into a
/// `ResolvedInferenceModel`. The resolver is injected by `app-host`
/// and may internally delegate to `model-manager`, but the
/// inference layer only sees `ResolvedInferenceModel`.
#[async_trait::async_trait]
pub trait ModelResolver: Send + Sync + 'static {
    async fn resolve(&self, model_ref: &ModelRef)
    -> Result<ResolvedInferenceModel, InferenceError>;
}
