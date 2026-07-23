use std::fmt;
use std::path::PathBuf;

use reimagine_core::model::ModelRole;
use reimagine_inference::{
    ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlTextEncoderSources {
    Split { clip_l: PathBuf, clip_g: PathBuf },
    Combined { path: PathBuf },
    Checkpoint { path: PathBuf },
}

impl SdxlTextEncoderSources {
    pub(crate) fn fingerprint(&self) -> String {
        match self {
            Self::Split { clip_l, clip_g } => {
                format!("split:{}|{}", clip_l.display(), clip_g.display())
            }
            Self::Combined { path } => format!("combined:{}", path.display()),
            Self::Checkpoint { path } => format!("checkpoint:{}", path.display()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlTextSourceError {
    IncompleteSplit { missing: &'static str },
    AmbiguousDuplicate { component: &'static str },
    MissingCheckpoint,
}

impl fmt::Display for SdxlTextSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompleteSplit { missing } => {
                write!(
                    f,
                    "incomplete SDXL text encoder split sources; missing {missing}"
                )
            }
            Self::AmbiguousDuplicate { component } => {
                write!(f, "ambiguous SDXL text encoder sources for {component}")
            }
            Self::MissingCheckpoint => write!(f, "missing SDXL checkpoint bundle source"),
        }
    }
}

impl std::error::Error for SdxlTextSourceError {}

pub(crate) fn resolve_text_encoder_sources(
    source_set: &ResolvedInferenceModelSourceSet,
) -> Result<SdxlTextEncoderSources, SdxlTextSourceError> {
    let mut clip_l = None;
    let mut clip_g = None;
    let mut combined = None;
    let mut checkpoint = None;

    for source in source_set.sources() {
        if source.kind() == ModelSourceKind::CheckpointBundle {
            checkpoint = Some(source.path().clone());
        }

        if source.kind() != ModelSourceKind::SplitComponent
            || source.role() != ModelRole::TextEncoder
        {
            continue;
        }

        let Some(component) = text_encoder_component(source) else {
            continue;
        };

        match component {
            TextEncoderComponent::ClipL => {
                set_once(&mut clip_l, source.path().clone(), "clip_l")?;
            }
            TextEncoderComponent::ClipG => {
                set_once(&mut clip_g, source.path().clone(), "clip_g")?;
            }
            TextEncoderComponent::Combined => {
                set_once(
                    &mut combined,
                    source.path().clone(),
                    "text_encoder_combined",
                )?;
            }
        }
    }

    match (clip_l, clip_g) {
        (Some(clip_l), Some(clip_g)) => Ok(SdxlTextEncoderSources::Split { clip_l, clip_g }),
        (Some(_), None) => Err(SdxlTextSourceError::IncompleteSplit { missing: "clip_g" }),
        (None, Some(_)) => Err(SdxlTextSourceError::IncompleteSplit { missing: "clip_l" }),
        (None, None) => combined
            .map(|path| SdxlTextEncoderSources::Combined { path })
            .or_else(|| checkpoint.map(|path| SdxlTextEncoderSources::Checkpoint { path }))
            .ok_or(SdxlTextSourceError::MissingCheckpoint),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextEncoderComponent {
    ClipL,
    ClipG,
    Combined,
}

fn set_once(
    slot: &mut Option<PathBuf>,
    path: PathBuf,
    component: &'static str,
) -> Result<(), SdxlTextSourceError> {
    if slot.is_some() {
        return Err(SdxlTextSourceError::AmbiguousDuplicate { component });
    }
    *slot = Some(path);
    Ok(())
}

fn text_encoder_component(source: &ResolvedInferenceModelSource) -> Option<TextEncoderComponent> {
    source.metadata().and_then(parse_text_encoder_component)
}

fn parse_text_encoder_component(metadata: &str) -> Option<TextEncoderComponent> {
    metadata
        .split(';')
        .flat_map(|part| part.split(','))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(key, value)| {
            let key = key.trim();
            let value = value.trim();
            match (key, value) {
                ("component", "clip_l") | ("clip", "clip_l") => Some(TextEncoderComponent::ClipL),
                ("component", "clip_g") | ("clip", "clip_g") => Some(TextEncoderComponent::ClipG),
                ("component", "text_encoder_combined") => Some(TextEncoderComponent::Combined),
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

    use super::{SdxlTextEncoderSources, SdxlTextSourceError, resolve_text_encoder_sources};

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

    fn text_encoder(path: &str, component: &str) -> ResolvedInferenceModelSource {
        source(
            ModelSourceKind::SplitComponent,
            ModelRole::TextEncoder,
            path,
            Some(component),
        )
    }

    #[test]
    fn resolves_complete_split_text_encoder_sources() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(text_encoder(
                    "/models/text_encoder.safetensors",
                    "component=clip_l",
                ))
                .with_source(text_encoder(
                    "/models/text_encoder_2.safetensors",
                    "component=clip_g",
                ));

        let resolved = resolve_text_encoder_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlTextEncoderSources::Split {
                clip_l: PathBuf::from("/models/text_encoder.safetensors"),
                clip_g: PathBuf::from("/models/text_encoder_2.safetensors"),
            }
        );
    }

    #[test]
    fn prefers_complete_split_sources_over_combined_and_checkpoint() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(text_encoder(
                    "/models/text_encoder_combined.safetensors",
                    "component=text_encoder_combined",
                ))
                .with_source(text_encoder(
                    "/models/text_encoder.safetensors",
                    "component=clip_l",
                ))
                .with_source(text_encoder(
                    "/models/text_encoder_2.safetensors",
                    "component=clip_g",
                ));

        let resolved = resolve_text_encoder_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlTextEncoderSources::Split {
                clip_l: PathBuf::from("/models/text_encoder.safetensors"),
                clip_g: PathBuf::from("/models/text_encoder_2.safetensors"),
            }
        );
    }

    #[test]
    fn rejects_incomplete_split_sources() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(text_encoder(
                    "/models/text_encoder.safetensors",
                    "component=clip_l",
                ));

        let err = resolve_text_encoder_sources(&source_set).unwrap_err();

        assert_eq!(
            err,
            SdxlTextSourceError::IncompleteSplit { missing: "clip_g" }
        );
        assert!(err.to_string().contains("clip_g"));
    }

    #[test]
    fn rejects_duplicate_ambiguous_metadata() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(text_encoder(
                    "/models/text_encoder_a.safetensors",
                    "component=clip_l",
                ))
                .with_source(text_encoder(
                    "/models/text_encoder_b.safetensors",
                    "component=clip_l",
                ))
                .with_source(text_encoder(
                    "/models/text_encoder_2.safetensors",
                    "component=clip_g",
                ));

        let err = resolve_text_encoder_sources(&source_set).unwrap_err();

        assert_eq!(
            err,
            SdxlTextSourceError::AmbiguousDuplicate {
                component: "clip_l"
            }
        );
        assert!(err.to_string().contains("clip_l"));
    }

    #[test]
    fn accepts_compatibility_clip_alias_for_clip_l_and_clip_g() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(text_encoder(
                    "/models/text_encoder.safetensors",
                    "clip=clip_l",
                ))
                .with_source(text_encoder(
                    "/models/text_encoder_2.safetensors",
                    "clip=clip_g",
                ));

        let resolved = resolve_text_encoder_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlTextEncoderSources::Split {
                clip_l: PathBuf::from("/models/text_encoder.safetensors"),
                clip_g: PathBuf::from("/models/text_encoder_2.safetensors"),
            }
        );
    }

    #[test]
    fn resolves_combined_text_encoder_source() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(text_encoder(
                    "/models/text_encoder_combined.safetensors",
                    "component=text_encoder_combined",
                ));

        let resolved = resolve_text_encoder_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlTextEncoderSources::Combined {
                path: PathBuf::from("/models/text_encoder_combined.safetensors"),
            }
        );
    }

    #[test]
    fn falls_back_to_checkpoint_when_no_split_text_encoder_exists() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"));

        let resolved = resolve_text_encoder_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlTextEncoderSources::Checkpoint {
                path: PathBuf::from("/models/sdxl.safetensors"),
            }
        );
    }

    #[test]
    fn parses_multi_key_metadata_produced_by_app_host_projection() {
        // Pins the contract between the app-host metadata serializer
        // (which joins sorted `key=value` pairs with `;`) and the
        // Candle text-encoder parser (which splits on `;` then `=`).
        // A multi-key BTreeMap must still resolve to the expected
        // split text-encoder components even when the trailing
        // `component=clip_*` pair is preceded by arbitrary metadata
        // keys.
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(text_encoder(
                    "/models/text_encoder/model.safetensors",
                    "component=clip_l;extra=value",
                ))
                .with_source(text_encoder(
                    "/models/text_encoder_2/model.safetensors",
                    "component=clip_g;role=second",
                ));

        let resolved = resolve_text_encoder_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlTextEncoderSources::Split {
                clip_l: PathBuf::from("/models/text_encoder/model.safetensors"),
                clip_g: PathBuf::from("/models/text_encoder_2/model.safetensors"),
            }
        );
    }
}
