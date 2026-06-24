use std::sync::Arc;

use reimagine_config::AppPaths;
use reimagine_core::model::ModelRef;
use reimagine_inference::{
    InferenceError, ModelFormat, ModelResolver, ModelSourceKind, ResolvedInferenceModel,
    ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};

use crate::ModelService;

pub(crate) struct ModelResolverAdapter {
    model_service: Arc<ModelService>,
    app_paths: AppPaths,
}

impl ModelResolverAdapter {
    pub(crate) fn new(model_service: Arc<ModelService>, app_paths: AppPaths) -> Self {
        Self {
            model_service,
            app_paths,
        }
    }
}

#[async_trait::async_trait]
impl ModelResolver for ModelResolverAdapter {
    async fn resolve(
        &self,
        model_ref: &ModelRef,
    ) -> Result<ResolvedInferenceModel, InferenceError> {
        let resolution = self
            .model_service
            .resolve_descriptor(model_ref)
            .await
            .map_err(|error| InferenceError::ModelResolutionFailed {
                message: error.to_string(),
            })?;

        let Some(descriptor) = resolution.into_value() else {
            return Err(InferenceError::ModelResolutionFailed {
                message: format!("model ref {} could not be resolved", model_ref.id()),
            });
        };

        let manifest = self.model_service.cached_manifest().ok_or_else(|| {
            InferenceError::ModelResolutionFailed {
                message: "model manifest not cached after resolution".to_string(),
            }
        })?;

        let source_path = reimagine_model_manager::resolve_source_path(
            &manifest,
            descriptor.source(),
            self.app_paths.models_dir(),
        )
        .ok_or_else(|| InferenceError::ModelResolutionFailed {
            message: format!(
                "could not resolve source path for model {}",
                descriptor.id()
            ),
        })?;

        let resolved = ResolvedInferenceModel::new(
            descriptor.id().clone(),
            descriptor.model_series().clone(),
            descriptor.variant().clone(),
            model_ref.role(),
            source_path.clone(),
            map_model_format(descriptor.format()),
        );

        let source_set = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
            ModelSourceKind::CheckpointBundle,
            model_ref.role(),
            source_path,
            map_model_format(descriptor.format()),
        ));

        Ok(resolved.with_source_set(source_set))
    }
}

fn map_model_format(format: reimagine_model_manager::ModelFormat) -> ModelFormat {
    match format {
        reimagine_model_manager::ModelFormat::Safetensors => ModelFormat::SafeTensors,
        reimagine_model_manager::ModelFormat::Gguf => ModelFormat::Gguf,
        reimagine_model_manager::ModelFormat::Ckpt => ModelFormat::PyTorch,
        reimagine_model_manager::ModelFormat::Unknown => ModelFormat::Other,
    }
}
