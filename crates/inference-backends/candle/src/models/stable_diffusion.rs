//! Stable Diffusion model family and loaded-bundle dispatch.
//!
//! `LoadedModelBundle` is the backend-local wrapper that
//! `CandleModelCache` stores. V1 only has the SDXL variant behind
//! `StableDiffusionSdxl(Arc<LoadedSdxlBundle>)`; later milestones add
//! SDXL refiner, SDXL Lightning, etc. behind the same enum.

use std::path::Path;
use std::sync::Arc;

use candle_core::Device;
use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_inference::ModelFormat;
use reimagine_inference::{
    ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};

use crate::error::CandleBackendError;
use crate::models::stable_diffusion::sdxl::LoadedSdxlBundle;

/// Backend-local loaded model bundle.
///
/// One variant per supported model family. Operation modules
/// pattern-match on the variant to dispatch kernel work; the cache
/// stores this enum and never sees a family-specific payload.
#[derive(Debug, Clone)]
pub enum LoadedModelBundle {
    StableDiffusionSdxl(Arc<LoadedSdxlBundle>),
    /// Test-only placeholder bundle used to verify that unsupported model
    /// families produce precise backend diagnostics instead of panicking.
    #[cfg(test)]
    TestPlaceholder,
}

impl LoadedModelBundle {
    /// Family and variant of the wrapped bundle, useful for kernel
    /// dispatch and diagnostics.
    pub fn family_label(&self) -> &'static str {
        match self {
            Self::StableDiffusionSdxl(_) => "stable_diffusion/sdxl",
            #[cfg(test)]
            Self::TestPlaceholder => "test/placeholder",
        }
    }

    /// Load a bundle for the given resolved model, dispatching on
    /// `series` and `variant`. Returns a useful backend error when
    /// the backend has no loader for the requested family.
    pub fn load(
        model_id: ModelId,
        series: &ModelSeries,
        variant: &ModelVariant,
        source_path: &Path,
        format: ModelFormat,
        device: Arc<Device>,
    ) -> Result<Arc<Self>, CandleBackendError> {
        let source = ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            source_path.to_path_buf(),
            format,
        );
        let source_set = ResolvedInferenceModelSourceSet::new(source);
        Self::load_from_source_set(model_id, series, variant, &source_set, format, device)
    }

    /// Load from a multi-source set (checkpoint bundle or split components).
    pub fn load_from_source_set(
        model_id: ModelId,
        series: &ModelSeries,
        variant: &ModelVariant,
        source_set: &ResolvedInferenceModelSourceSet,
        format: ModelFormat,
        device: Arc<Device>,
    ) -> Result<Arc<Self>, CandleBackendError> {
        if series.as_str() == "stable_diffusion" && variant.as_str() == "sdxl" {
            let sdxl = LoadedSdxlBundle::from_resolved_with_source_set(
                model_id,
                source_set.clone(),
                format,
                device,
            )?;
            Ok(Arc::new(Self::StableDiffusionSdxl(sdxl)))
        } else {
            Err(CandleBackendError::UnsupportedModelFamily {
                model_id: model_id.as_str().to_string(),
                series: series.as_str().to_string(),
                variant: variant.as_str().to_string(),
            })
        }
    }

    pub(crate) fn load_from_source_set_with_test_text_projection(
        model_id: ModelId,
        series: &ModelSeries,
        variant: &ModelVariant,
        source_set: &ResolvedInferenceModelSourceSet,
        format: ModelFormat,
        device: Arc<Device>,
    ) -> Result<Arc<Self>, CandleBackendError> {
        if series.as_str() == "stable_diffusion" && variant.as_str() == "sdxl" {
            let sdxl = LoadedSdxlBundle::from_resolved_with_test_text_projection(
                model_id,
                source_set.clone(),
                format,
                device,
            )?;
            Ok(Arc::new(Self::StableDiffusionSdxl(sdxl)))
        } else {
            Self::load_from_source_set(model_id, series, variant, source_set, format, device)
        }
    }
}

pub mod sdxl;
