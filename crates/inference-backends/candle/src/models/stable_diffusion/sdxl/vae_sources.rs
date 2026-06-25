use std::fmt;
use std::path::PathBuf;

use reimagine_core::model::ModelRole;
use reimagine_inference::{
    ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlVaeSources {
    Split { path: PathBuf },
    Checkpoint { path: PathBuf },
}

impl SdxlVaeSources {
    pub(crate) fn fingerprint(&self) -> String {
        match self {
            Self::Split { path } => format!("split:{}", path.display()),
            Self::Checkpoint { path } => format!("checkpoint:{}", path.display()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlVaeSourceError {
    AmbiguousDuplicate { component: &'static str },
    MissingComponentMetadata { path: PathBuf },
    UnsupportedComponentMetadata { path: PathBuf, metadata: String },
    MissingCheckpoint,
}

impl fmt::Display for SdxlVaeSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AmbiguousDuplicate { component } => {
                write!(f, "ambiguous SDXL VAE sources for {component}")
            }
            Self::MissingComponentMetadata { path } => write!(
                f,
                "SDXL VAE split source `{}` requires metadata `component=vae` or `component=first_stage_model`",
                path.display()
            ),
            Self::UnsupportedComponentMetadata { path, metadata } => write!(
                f,
                "SDXL VAE split source `{}` has unsupported metadata `{metadata}`; expected `component=vae` or `component=first_stage_model`",
                path.display()
            ),
            Self::MissingCheckpoint => write!(f, "missing SDXL checkpoint bundle source"),
        }
    }
}

impl std::error::Error for SdxlVaeSourceError {}

pub(crate) fn resolve_vae_sources(
    source_set: &ResolvedInferenceModelSourceSet,
) -> Result<SdxlVaeSources, SdxlVaeSourceError> {
    let mut split = None;
    let mut checkpoint = None;

    for source in source_set.sources() {
        if source.kind() == ModelSourceKind::CheckpointBundle {
            checkpoint = Some(source.path().clone());
        }

        if source.kind() != ModelSourceKind::SplitComponent || source.role() != ModelRole::Vae {
            continue;
        }

        match vae_component(source) {
            Some(VaeComponent::Vae) => set_once(&mut split, source.path().clone(), "vae")?,
            None if source.metadata().is_none() => {
                return Err(SdxlVaeSourceError::MissingComponentMetadata {
                    path: source.path().clone(),
                });
            }
            None => {
                return Err(SdxlVaeSourceError::UnsupportedComponentMetadata {
                    path: source.path().clone(),
                    metadata: source.metadata().unwrap_or_default().to_string(),
                });
            }
        }
    }

    split
        .map(|path| SdxlVaeSources::Split { path })
        .or_else(|| checkpoint.map(|path| SdxlVaeSources::Checkpoint { path }))
        .ok_or(SdxlVaeSourceError::MissingCheckpoint)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VaeComponent {
    Vae,
}

fn set_once(
    slot: &mut Option<PathBuf>,
    path: PathBuf,
    component: &'static str,
) -> Result<(), SdxlVaeSourceError> {
    if slot.is_some() {
        return Err(SdxlVaeSourceError::AmbiguousDuplicate { component });
    }
    *slot = Some(path);
    Ok(())
}

fn vae_component(source: &ResolvedInferenceModelSource) -> Option<VaeComponent> {
    source.metadata().and_then(parse_vae_component)
}

fn parse_vae_component(metadata: &str) -> Option<VaeComponent> {
    metadata
        .split(';')
        .flat_map(|part| part.split(','))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(key, value)| match (key.trim(), value.trim()) {
            ("component", "vae") | ("component", "first_stage_model") => Some(VaeComponent::Vae),
            _ => None,
        })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use reimagine_core::model::ModelRole;
    use reimagine_inference::{
        ModelFormat, ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
    };

    use super::{SdxlVaeSourceError, SdxlVaeSources, resolve_vae_sources};

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

    fn vae(path: &str, metadata: Option<&str>) -> ResolvedInferenceModelSource {
        source(
            ModelSourceKind::SplitComponent,
            ModelRole::Vae,
            path,
            metadata,
        )
    }

    #[test]
    fn resolves_split_vae_over_checkpoint() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(vae("/models/vae.safetensors", Some("component=vae")));

        let resolved = resolve_vae_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlVaeSources::Split {
                path: PathBuf::from("/models/vae.safetensors"),
            }
        );
    }

    #[test]
    fn accepts_first_stage_model_component_alias() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(vae(
                    "/models/vae.safetensors",
                    Some("component=first_stage_model"),
                ));

        let resolved = resolve_vae_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlVaeSources::Split {
                path: PathBuf::from("/models/vae.safetensors"),
            }
        );
    }

    #[test]
    fn rejects_missing_split_component_metadata() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(vae("/models/vae.safetensors", None));

        let err = resolve_vae_sources(&source_set).unwrap_err();

        assert_eq!(
            err,
            SdxlVaeSourceError::MissingComponentMetadata {
                path: PathBuf::from("/models/vae.safetensors"),
            }
        );
    }
}
