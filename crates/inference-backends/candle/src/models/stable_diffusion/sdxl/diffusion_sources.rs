use std::fmt;
use std::path::PathBuf;

use reimagine_core::model::ModelRole;
use reimagine_inference::{
    ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlDiffusionSources {
    Split { path: PathBuf },
    Checkpoint { path: PathBuf },
}

impl SdxlDiffusionSources {
    pub(crate) fn path(&self) -> &PathBuf {
        match self {
            Self::Split { path } | Self::Checkpoint { path } => path,
        }
    }

    pub(crate) fn fingerprint(&self) -> String {
        match self {
            Self::Split { path } => format!("split:{}", path.display()),
            Self::Checkpoint { path } => format!("checkpoint:{}", path.display()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlDiffusionSourceError {
    AmbiguousDuplicate { component: &'static str },
    MissingComponentMetadata { path: PathBuf },
    UnsupportedComponentMetadata { path: PathBuf, metadata: String },
    MissingCheckpoint,
}

impl fmt::Display for SdxlDiffusionSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AmbiguousDuplicate { component } => {
                write!(f, "ambiguous SDXL diffusion sources for {component}")
            }
            Self::MissingComponentMetadata { path } => write!(
                f,
                "SDXL diffusion split source `{}` requires metadata `component=unet` or `component=diffusion_model`",
                path.display()
            ),
            Self::UnsupportedComponentMetadata { path, metadata } => write!(
                f,
                "SDXL diffusion split source `{}` has unsupported metadata `{metadata}`; expected `component=unet` or `component=diffusion_model`",
                path.display()
            ),
            Self::MissingCheckpoint => write!(f, "missing SDXL checkpoint bundle source"),
        }
    }
}

impl std::error::Error for SdxlDiffusionSourceError {}

pub(crate) fn resolve_diffusion_sources(
    source_set: &ResolvedInferenceModelSourceSet,
) -> Result<SdxlDiffusionSources, SdxlDiffusionSourceError> {
    let mut split = None;
    let mut checkpoint = None;

    for source in source_set.sources() {
        if source.kind() == ModelSourceKind::CheckpointBundle {
            checkpoint = Some(source.path().clone());
        }

        if source.kind() != ModelSourceKind::SplitComponent
            || source.role() != ModelRole::DiffusionModel
        {
            continue;
        }

        match diffusion_component(source) {
            Some(DiffusionComponent::Unet) => {
                set_once(&mut split, source.path().clone(), "unet")?;
            }
            None if source.metadata().is_none() => {
                return Err(SdxlDiffusionSourceError::MissingComponentMetadata {
                    path: source.path().clone(),
                });
            }
            None => {
                return Err(SdxlDiffusionSourceError::UnsupportedComponentMetadata {
                    path: source.path().clone(),
                    metadata: source.metadata().unwrap_or_default().to_string(),
                });
            }
        }
    }

    split
        .map(|path| SdxlDiffusionSources::Split { path })
        .or_else(|| checkpoint.map(|path| SdxlDiffusionSources::Checkpoint { path }))
        .ok_or(SdxlDiffusionSourceError::MissingCheckpoint)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffusionComponent {
    Unet,
}

fn set_once(
    slot: &mut Option<PathBuf>,
    path: PathBuf,
    component: &'static str,
) -> Result<(), SdxlDiffusionSourceError> {
    if slot.is_some() {
        return Err(SdxlDiffusionSourceError::AmbiguousDuplicate { component });
    }
    *slot = Some(path);
    Ok(())
}

fn diffusion_component(source: &ResolvedInferenceModelSource) -> Option<DiffusionComponent> {
    source.metadata().and_then(parse_diffusion_component)
}

fn parse_diffusion_component(metadata: &str) -> Option<DiffusionComponent> {
    metadata
        .split(';')
        .flat_map(|part| part.split(','))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(key, value)| {
            let key = key.trim();
            let value = value.trim();
            match (key, value) {
                ("component", "unet") | ("component", "diffusion_model") => {
                    Some(DiffusionComponent::Unet)
                }
                _ => None,
            }
        })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use reimagine_core::model::ModelRole;
    use reimagine_inference::{
        ModelFormat, ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
    };

    use super::{SdxlDiffusionSourceError, SdxlDiffusionSources, resolve_diffusion_sources};

    fn source(
        kind: ModelSourceKind,
        role: ModelRole,
        path: &str,
        metadata: Option<&str>,
    ) -> ResolvedInferenceModelSource {
        let source = ResolvedInferenceModelSource::new(
            kind,
            role,
            PathBuf::from(path),
            ModelFormat::SafeTensors,
        );
        match metadata {
            Some(metadata) => source.with_metadata(metadata),
            None => source,
        }
    }

    fn checkpoint(path: &str) -> ResolvedInferenceModelSource {
        source(
            ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            path,
            None,
        )
    }

    fn diffusion(path: &str, metadata: Option<&str>) -> ResolvedInferenceModelSource {
        source(
            ModelSourceKind::SplitComponent,
            ModelRole::DiffusionModel,
            path,
            metadata,
        )
    }

    #[test]
    fn resolves_split_unet_over_checkpoint() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(diffusion(
                    "/models/unet/model.safetensors",
                    Some("component=unet"),
                ));

        let resolved = resolve_diffusion_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlDiffusionSources::Split {
                path: PathBuf::from("/models/unet/model.safetensors"),
            }
        );
        assert_eq!(
            resolved.path(),
            &PathBuf::from("/models/unet/model.safetensors")
        );
    }

    #[test]
    fn accepts_diffusion_model_component_alias() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(diffusion(
                    "/models/diffusion.safetensors",
                    Some("component=diffusion_model"),
                ));

        let resolved = resolve_diffusion_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlDiffusionSources::Split {
                path: PathBuf::from("/models/diffusion.safetensors"),
            }
        );
    }

    #[test]
    fn falls_back_to_checkpoint_when_no_split_diffusion_source_exists() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"));

        let resolved = resolve_diffusion_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlDiffusionSources::Checkpoint {
                path: PathBuf::from("/models/sdxl.safetensors"),
            }
        );
    }

    #[test]
    fn rejects_missing_split_component_metadata() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(diffusion("/models/unet/model.safetensors", None));

        let err = resolve_diffusion_sources(&source_set).unwrap_err();

        assert_eq!(
            err,
            SdxlDiffusionSourceError::MissingComponentMetadata {
                path: PathBuf::from("/models/unet/model.safetensors"),
            }
        );
        assert!(err.to_string().contains("component=unet"));
    }

    #[test]
    fn rejects_unsupported_split_component_metadata() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(diffusion(
                    "/models/clip.safetensors",
                    Some("component=clip_l"),
                ));

        let err = resolve_diffusion_sources(&source_set).unwrap_err();

        assert_eq!(
            err,
            SdxlDiffusionSourceError::UnsupportedComponentMetadata {
                path: PathBuf::from("/models/clip.safetensors"),
                metadata: "component=clip_l".to_string(),
            }
        );
        assert!(err.to_string().contains("component=clip_l"));
    }

    #[test]
    fn rejects_duplicate_diffusion_components() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(diffusion(
                    "/models/unet-a.safetensors",
                    Some("component=unet"),
                ))
                .with_source(diffusion(
                    "/models/unet-b.safetensors",
                    Some("component=diffusion_model"),
                ));

        let err = resolve_diffusion_sources(&source_set).unwrap_err();

        assert_eq!(
            err,
            SdxlDiffusionSourceError::AmbiguousDuplicate { component: "unet" }
        );
    }

    #[test]
    fn parses_multi_key_metadata_produced_by_app_host_projection() {
        // Pins the contract between the app-host metadata serializer
        // (which joins sorted `key=value` pairs with `;`) and the
        // Candle parser (which splits on `;` then `=`). A multi-key
        // BTreeMap must still resolve to the expected component even
        // when the trailing `component=unet` pair is preceded by
        // arbitrary metadata keys.
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(diffusion(
                    "/models/unet/model.safetensors",
                    Some("component=unet;extra=value"),
                ));

        let resolved = resolve_diffusion_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlDiffusionSources::Split {
                path: PathBuf::from("/models/unet/model.safetensors"),
            }
        );
    }
}
