use std::collections::BTreeMap;
use std::sync::Arc;

use reimagine_config::AppPaths;
use reimagine_core::model::{ModelRef, ModelRole};
use reimagine_inference::{
    InferenceError, ModelFormat, ModelResolver, ModelSourceKind, ResolvedInferenceModel,
    ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};
use reimagine_model_manager::ResolvedComponent;

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
            .resolve_descriptor_with_components(model_ref)
            .await
            .map_err(|error| InferenceError::ModelResolutionFailed {
                message: error.to_string(),
            })?;

        let error_diagnostics = resolution
            .report()
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.severity().is_error())
            .map(|diagnostic| format!("{}: {}", diagnostic.code().as_str(), diagnostic.message()))
            .collect::<Vec<_>>();
        if !error_diagnostics.is_empty() {
            return Err(InferenceError::ModelResolutionFailed {
                message: format!(
                    "model ref {} failed model resolution diagnostics: {}",
                    model_ref.id(),
                    error_diagnostics.join("; ")
                ),
            });
        }

        let Some(view) = resolution.into_value() else {
            return Err(InferenceError::ModelResolutionFailed {
                message: format!("model ref {} could not be resolved", model_ref.id()),
            });
        };

        let descriptor = view.descriptor();
        let manifest = self.model_service.cached_manifest().ok_or_else(|| {
            InferenceError::ModelResolutionFailed {
                message: "model manifest not cached after resolution".to_string(),
            }
        })?;

        let (source_path, format) = primary_source_path_and_format(
            &manifest,
            descriptor,
            view.components(),
            model_ref.role(),
            self.app_paths.models_dir(),
        )
        .map_err(|message| InferenceError::ModelResolutionFailed { message })?;

        let resolved = ResolvedInferenceModel::new(
            descriptor.id().clone(),
            descriptor.model_series().clone(),
            descriptor.variant().clone(),
            model_ref.role(),
            source_path,
            map_model_format(format),
        );

        let source_set = if descriptor.components().is_empty() {
            ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
                ModelSourceKind::CheckpointBundle,
                model_ref.role(),
                resolved.source_path().to_path_buf(),
                resolved.format(),
            ))
        } else {
            let sources = view
                .components()
                .iter()
                .map(|component| {
                    ResolvedInferenceModelSource::new(
                        ModelSourceKind::SplitComponent,
                        component.role(),
                        component.path().to_path_buf(),
                        map_model_format(component.format()),
                    )
                    .with_metadata(serialize_metadata(component.metadata()))
                })
                .collect::<Vec<_>>();
            ResolvedInferenceModelSourceSet::from_sources(sources)
        };

        Ok(resolved.with_source_set(source_set))
    }
}

/// Resolve the primary `source_path` and `format` for a resolved model.
///
/// For a descriptor with no components (the legacy single-source /
/// checkpoint-bundle shape) this falls through to the descriptor's
/// primary `source()` path.
///
/// For a split descriptor, a `CheckpointBundle` request means "load the whole
/// component graph" and may use any component as the legacy primary path
/// because the complete source set carries the executable sources. Requests
/// for a concrete component role still prefer a matching component so the
/// resolved model's legacy path/format stays intuitive in diagnostics.
fn primary_source_path_and_format(
    manifest: &reimagine_model_manager::ModelManifest,
    descriptor: &reimagine_model_manager::ModelDescriptor,
    components: &[ResolvedComponent],
    requested_role: ModelRole,
    models_dir: &std::path::Path,
) -> Result<(std::path::PathBuf, reimagine_model_manager::ModelFormat), String> {
    if !components.is_empty() {
        if let Some(matching) = components
            .iter()
            .find(|component| component.role() == requested_role)
        {
            return Ok((matching.path().to_path_buf(), matching.format()));
        }
        if requested_role == ModelRole::CheckpointBundle {
            let primary = components
                .first()
                .expect("non-empty component list has a first component");
            return Ok((primary.path().to_path_buf(), primary.format()));
        }
        let available_roles: Vec<String> = components
            .iter()
            .map(|component| format!("{:?}", component.role()))
            .collect();
        return Err(format!(
            "model {} does not declare a split component for requested role `{:?}`; available component roles: [{}]",
            descriptor.id(),
            requested_role,
            available_roles.join(", "),
        ));
    }
    let path =
        reimagine_model_manager::resolve_source_path(manifest, descriptor.source(), models_dir)
            .ok_or_else(|| {
                format!(
                    "could not resolve primary source path for model {}",
                    descriptor.id()
                )
            })?;
    Ok((path, descriptor.format()))
}

fn serialize_metadata(metadata: &BTreeMap<String, String>) -> String {
    metadata
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn map_model_format(format: reimagine_model_manager::ModelFormat) -> ModelFormat {
    match format {
        reimagine_model_manager::ModelFormat::Safetensors => ModelFormat::SafeTensors,
        reimagine_model_manager::ModelFormat::Gguf => ModelFormat::Gguf,
        reimagine_model_manager::ModelFormat::Ckpt => ModelFormat::PyTorch,
        reimagine_model_manager::ModelFormat::Unknown => ModelFormat::Other,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use reimagine_config::AppPaths;
    use reimagine_core::model::{ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant};
    use reimagine_inference::ModelFormat as InferenceModelFormat;
    use reimagine_model_manager::{
        ModelComponentSource, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelRootId,
        ModelSource, ModelSourceStatus,
    };

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "reimagine-app-host-resolver-{name}-{}",
            std::process::id()
        ))
    }

    fn split_sdxl_descriptor() -> ModelDescriptor {
        ModelDescriptor::new(
            ModelId::new("sdxl-base-1.0"),
            ModelSeries::new("stable_diffusion"),
            ModelVariant::new("sdxl"),
            vec![
                ModelRole::CheckpointBundle,
                ModelRole::DiffusionModel,
                ModelRole::TextEncoder,
                ModelRole::Vae,
            ],
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/manifest.safetensors",
            ),
            ModelFormat::Safetensors,
        )
        .with_source_status(ModelSourceStatus::Available)
        .with_size_bytes(7)
        .with_observed_size_bytes(7)
        .with_component(
            ModelComponentSource::new(
                ModelRole::DiffusionModel,
                ModelSource::relative(
                    ModelRootId::new("base"),
                    "sdxl-base-1.0/unet/model.safetensors",
                ),
                ModelFormat::Safetensors,
            )
            .with_metadata("component", "unet"),
        )
        .with_component(
            ModelComponentSource::new(
                ModelRole::TextEncoder,
                ModelSource::relative(
                    ModelRootId::new("base"),
                    "sdxl-base-1.0/text_encoder/model.safetensors",
                ),
                ModelFormat::Safetensors,
            )
            .with_metadata("component", "clip_l"),
        )
        .with_component(
            ModelComponentSource::new(
                ModelRole::TextEncoder,
                ModelSource::relative(
                    ModelRootId::new("base"),
                    "sdxl-base-1.0/text_encoder_2/model.safetensors",
                ),
                ModelFormat::Safetensors,
            )
            .with_metadata("component", "clip_g"),
        )
        .with_component(
            ModelComponentSource::new(
                ModelRole::Vae,
                ModelSource::relative(
                    ModelRootId::new("base"),
                    "sdxl-base-1.0/vae/model.safetensors",
                ),
                ModelFormat::Safetensors,
            )
            .with_metadata("component", "vae"),
        )
    }

    fn legacy_descriptor() -> ModelDescriptor {
        ModelDescriptor::new(
            ModelId::new("sdxl-base-1.0"),
            ModelSeries::new("stable_diffusion"),
            ModelVariant::new("sdxl"),
            vec![ModelRole::CheckpointBundle],
            ModelSource::relative(
                ModelRootId::new("base"),
                "sdxl-base-1.0/checkpoint.safetensors",
            ),
            ModelFormat::Safetensors,
        )
        .with_source_status(ModelSourceStatus::Available)
        .with_size_bytes(7)
        .with_observed_size_bytes(7)
    }

    fn model_ref_for(role: ModelRole) -> ModelRef {
        ModelRef::new(
            ModelId::new("sdxl-base-1.0"),
            ModelSeries::new("stable_diffusion"),
            ModelVariant::new("sdxl"),
            role,
        )
    }

    #[tokio::test]
    async fn split_descriptor_projects_each_component_as_a_split_source() {
        let base = temp_dir("split");
        let paths = AppPaths::new(&base);
        tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
        for relative in [
            "sdxl-base-1.0/manifest.safetensors",
            "sdxl-base-1.0/unet/model.safetensors",
            "sdxl-base-1.0/text_encoder/model.safetensors",
            "sdxl-base-1.0/text_encoder_2/model.safetensors",
            "sdxl-base-1.0/vae/model.safetensors",
        ] {
            let path = paths.models_dir().join(relative);
            tokio::fs::create_dir_all(path.parent().unwrap())
                .await
                .unwrap();
            tokio::fs::write(&path, b"weights").await.unwrap();
        }

        let manifest = ModelManifest::new()
            .with_root(ModelRoot::base_models())
            .with_model(split_sdxl_descriptor());
        let service = ModelService::new(paths.clone());
        service
            .save_manifest(&manifest)
            .await
            .expect("save manifest");

        let adapter = ModelResolverAdapter::new(std::sync::Arc::new(service), paths.clone());

        let resolved = adapter
            .resolve(&model_ref_for(ModelRole::TextEncoder))
            .await
            .expect("split descriptor should resolve");

        assert_eq!(resolved.role(), ModelRole::TextEncoder);
        let source_set = resolved
            .source_set()
            .expect("split descriptor should carry a source set");
        assert_eq!(source_set.sources().len(), 4);
        assert!(!source_set.is_checkpoint_bundle());

        let diffusion_source = source_set
            .sources()
            .iter()
            .find(|s| s.role() == ModelRole::DiffusionModel)
            .expect("diffusion source");
        assert_eq!(diffusion_source.kind(), ModelSourceKind::SplitComponent);
        assert_eq!(diffusion_source.format(), InferenceModelFormat::SafeTensors);
        assert_eq!(diffusion_source.metadata(), Some("component=unet"));

        let clip_l_source = source_set
            .sources()
            .iter()
            .filter(|s| s.role() == ModelRole::TextEncoder)
            .find(|s| s.metadata() == Some("component=clip_l"))
            .expect("clip_l source");
        assert_eq!(clip_l_source.kind(), ModelSourceKind::SplitComponent);

        let clip_g_source = source_set
            .sources()
            .iter()
            .filter(|s| s.role() == ModelRole::TextEncoder)
            .find(|s| s.metadata() == Some("component=clip_g"))
            .expect("clip_g source");
        assert_eq!(clip_g_source.kind(), ModelSourceKind::SplitComponent);

        let vae_source = source_set
            .sources()
            .iter()
            .find(|s| s.role() == ModelRole::Vae)
            .expect("vae source");
        assert_eq!(vae_source.metadata(), Some("component=vae"));

        let _ = tokio::fs::remove_dir_all(&base).await;
    }

    #[tokio::test]
    async fn legacy_single_source_descriptor_projects_a_checkpoint_bundle_source_set() {
        let base = temp_dir("legacy");
        let paths = AppPaths::new(&base);
        tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
        let path = paths
            .models_dir()
            .join("sdxl-base-1.0/checkpoint.safetensors");
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&path, b"weights").await.unwrap();

        let manifest = ModelManifest::new()
            .with_root(ModelRoot::base_models())
            .with_model(legacy_descriptor());
        let service = ModelService::new(paths.clone());
        service
            .save_manifest(&manifest)
            .await
            .expect("save manifest");

        let adapter = ModelResolverAdapter::new(std::sync::Arc::new(service), paths.clone());

        let resolved = adapter
            .resolve(&model_ref_for(ModelRole::CheckpointBundle))
            .await
            .expect("legacy descriptor should resolve");

        let source_set = resolved
            .source_set()
            .expect("legacy descriptor should carry a source set");
        assert_eq!(source_set.sources().len(), 1);
        assert!(source_set.is_checkpoint_bundle());
        assert_eq!(
            source_set.sources()[0].kind(),
            ModelSourceKind::CheckpointBundle
        );
        assert_eq!(source_set.sources()[0].role(), ModelRole::CheckpointBundle);
        assert_eq!(
            source_set.sources()[0].format(),
            InferenceModelFormat::SafeTensors
        );
        assert!(source_set.sources()[0].metadata().is_none());

        let _ = tokio::fs::remove_dir_all(&base).await;
    }

    #[tokio::test]
    async fn checkpoint_bundle_ref_for_split_descriptor_loads_the_component_graph() {
        let base = temp_dir("checkpoint-bundle-split");
        let paths = AppPaths::new(&base);
        tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
        for relative in [
            "sdxl-base-1.0/manifest.safetensors",
            "sdxl-base-1.0/unet/model.safetensors",
            "sdxl-base-1.0/text_encoder/model.safetensors",
            "sdxl-base-1.0/text_encoder_2/model.safetensors",
            "sdxl-base-1.0/vae/model.safetensors",
        ] {
            let path = paths.models_dir().join(relative);
            tokio::fs::create_dir_all(path.parent().unwrap())
                .await
                .unwrap();
            tokio::fs::write(&path, b"weights").await.unwrap();
        }

        let manifest = ModelManifest::new()
            .with_root(ModelRoot::base_models())
            .with_model(split_sdxl_descriptor());
        let service = ModelService::new(paths.clone());
        service
            .save_manifest(&manifest)
            .await
            .expect("save manifest");

        let adapter = ModelResolverAdapter::new(std::sync::Arc::new(service), paths.clone());

        let resolved = adapter
            .resolve(&model_ref_for(ModelRole::CheckpointBundle))
            .await
            .expect("checkpoint bundle model ref should resolve the split component graph");

        assert_eq!(resolved.role(), ModelRole::CheckpointBundle);
        assert!(
            resolved
                .source_path()
                .ends_with("sdxl-base-1.0/unet/model.safetensors"),
            "legacy primary source path should come from the first component, got {:?}",
            resolved.source_path()
        );
        let source_set = resolved
            .source_set()
            .expect("split descriptor should carry source set");
        assert_eq!(source_set.sources().len(), 4);
        assert!(
            source_set
                .sources()
                .iter()
                .all(|source| source.kind() == ModelSourceKind::SplitComponent)
        );
        assert!(
            source_set
                .sources()
                .iter()
                .any(|source| source.metadata() == Some("component=clip_l"))
        );
        assert!(
            source_set
                .sources()
                .iter()
                .any(|source| source.metadata() == Some("component=clip_g"))
        );

        let _ = tokio::fs::remove_dir_all(&base).await;
    }

    #[tokio::test]
    async fn split_descriptor_with_missing_component_reports_model_resolution_failed() {
        let base = temp_dir("missing-component");
        let paths = AppPaths::new(&base);
        tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
        for relative in [
            "sdxl-base-1.0/manifest.safetensors",
            "sdxl-base-1.0/unet/model.safetensors",
            "sdxl-base-1.0/text_encoder/model.safetensors",
            "sdxl-base-1.0/vae/model.safetensors",
        ] {
            let path = paths.models_dir().join(relative);
            tokio::fs::create_dir_all(path.parent().unwrap())
                .await
                .unwrap();
            tokio::fs::write(&path, b"weights").await.unwrap();
        }

        let manifest = ModelManifest::new()
            .with_root(ModelRoot::base_models())
            .with_model(split_sdxl_descriptor());
        let service = ModelService::new(paths.clone());
        service
            .save_manifest(&manifest)
            .await
            .expect("save manifest");

        let adapter = ModelResolverAdapter::new(std::sync::Arc::new(service), paths.clone());

        let err = adapter
            .resolve(&model_ref_for(ModelRole::CheckpointBundle))
            .await
            .expect_err("missing split component should block model resolution");

        match err {
            InferenceError::ModelResolutionFailed { message } => {
                assert!(
                    message.contains("MODEL_MANAGER/COMPONENT_SOURCE_MISSING"),
                    "error should preserve component diagnostic code, got: {message}"
                );
                assert!(
                    message.contains("TextEncoder:clip_g"),
                    "error should identify the missing component, got: {message}"
                );
            }
            other => panic!("expected ModelResolutionFailed, got {other:?}"),
        }

        let _ = tokio::fs::remove_dir_all(&base).await;
    }

    #[test]
    fn serialize_metadata_renders_btreemap_into_sorted_semicolon_separated_pairs() {
        let mut metadata = BTreeMap::new();
        metadata.insert("component".to_owned(), "clip_l".to_owned());
        metadata.insert("extra".to_owned(), "value".to_owned());
        let rendered = serialize_metadata(&metadata);
        assert_eq!(rendered, "component=clip_l;extra=value");
    }
}
