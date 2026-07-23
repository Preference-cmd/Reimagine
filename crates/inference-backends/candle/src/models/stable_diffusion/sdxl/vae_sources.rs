use std::fmt;
use std::path::PathBuf;

use reimagine_core::model::ModelRole;
use reimagine_inference::{
    ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};

/// Resolved SDXL VAE source shape.
///
/// V1 real decode accepts split VAE component sources only. Raw
/// single-file checkpoint VAE weights are not loaded directly; users
/// must run [`crate::models::stable_diffusion::sdxl::checkpoint_import`]
/// first to produce the Candle-compatible split layout, then VAE
/// decode consumes the split VAE file.
///
/// Split VAE components override a checkpoint-bundle VAE for the
/// decode path only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlVaeSources {
    /// A split VAE component source keyed by `component=vae` or
    /// `component=vae_decoder`. The path points at a single
    /// safetensors file with bare Candle example keys (decoder.* /
    /// post_quant_conv.* / encoder.* / quant_conv.*).
    Split { path: PathBuf },
    /// Fallback when only a checkpoint bundle is present. The split
    /// import path is required before real decode can succeed; loading
    /// reports a precise diagnostic pointing at the importer.
    Checkpoint { path: PathBuf },
}

impl SdxlVaeSources {
    #[allow(dead_code)]
    pub(crate) fn fingerprint(&self) -> String {
        match self {
            Self::Split { path } => format!("split:{}", path.display()),
            Self::Checkpoint { path } => format!("checkpoint:{}", path.display()),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn path(&self) -> &PathBuf {
        match self {
            Self::Split { path } | Self::Checkpoint { path } => path,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SdxlVaeSourceError {
    AmbiguousDuplicate {
        component: &'static str,
    },
    MissingComponentMetadata {
        path: PathBuf,
    },
    UnsupportedComponentMetadata {
        path: PathBuf,
        metadata: String,
    },
    /// No split VAE component is present; the caller is expected to
    /// run `import_sdxl_checkpoint_to_candle_example_split` to produce
    /// the split VAE file first.
    RequiresSplitImport {
        path: PathBuf,
    },
}

impl fmt::Display for SdxlVaeSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AmbiguousDuplicate { component } => {
                write!(f, "ambiguous SDXL VAE sources for {component}")
            }
            Self::MissingComponentMetadata { path } => write!(
                f,
                "SDXL VAE split source `{}` requires metadata `component=vae` or `component=vae_decoder`",
                path.display()
            ),
            Self::UnsupportedComponentMetadata { path, metadata } => write!(
                f,
                "SDXL VAE split source `{}` has unsupported metadata `{metadata}`; expected `component=vae` or `component=vae_decoder`",
                path.display()
            ),
            Self::RequiresSplitImport { path } => write!(
                f,
                "SDXL VAE decode requires a Candle-compatible split VAE source; only the original checkpoint `{}` is present. Run `import_sdxl_checkpoint_to_candle_example_split` first to produce `vae/model.safetensors` with bare Candle example keys, then re-supply it with `component=vae`",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SdxlVaeSourceError {}

/// Resolve VAE sources from the resolved inference source set.
///
/// V1 only loads split VAE components for real decode. A checkpoint
/// bundle without a split VAE companion is accepted by this
/// resolver (so existing checkpoint-only model references stay
/// parseable) but flagged via [`SdxlVaeSources::Checkpoint`] so the
/// graph facade can emit a precise "requires split import" diagnostic
/// at decode time instead of silently using placeholder weights.
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
        .ok_or(SdxlVaeSourceError::RequiresSplitImport {
            // No checkpoint either; surface a synthetic path so callers
            // see an actionable diagnostic rather than a bare
            // MissingCheckpoint string.
            path: PathBuf::from("<no checkpoint or split VAE source present>"),
        })
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
        .find_map(|(key, value)| {
            let key = key.trim();
            let value = value.trim();
            match (key, value) {
                ("component", "vae")
                | ("component", "vae_decoder")
                | ("component", "first_stage_model") => Some(VaeComponent::Vae),
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
    fn accepts_vae_decoder_component_metadata() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(vae(
                    "/models/vae.safetensors",
                    Some("component=vae_decoder"),
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

    #[test]
    fn rejects_unsupported_split_component_metadata() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(vae("/models/vae.safetensors", Some("component=clip_l")));

        let err = resolve_vae_sources(&source_set).unwrap_err();

        assert!(matches!(
            err,
            SdxlVaeSourceError::UnsupportedComponentMetadata { .. }
        ));
        assert!(err.to_string().contains("component=clip_l"));
    }

    #[test]
    fn rejects_duplicate_vae_components() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(vae("/models/vae-a.safetensors", Some("component=vae")))
                .with_source(vae(
                    "/models/vae-b.safetensors",
                    Some("component=vae_decoder"),
                ));

        let err = resolve_vae_sources(&source_set).unwrap_err();

        assert_eq!(
            err,
            SdxlVaeSourceError::AmbiguousDuplicate { component: "vae" }
        );
    }

    #[test]
    fn falls_back_to_checkpoint_when_no_split_vae_is_present() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"));

        let resolved = resolve_vae_sources(&source_set).unwrap();

        assert_eq!(
            resolved,
            SdxlVaeSources::Checkpoint {
                path: PathBuf::from("/models/sdxl.safetensors"),
            }
        );
        assert_eq!(
            resolved.fingerprint(),
            "checkpoint:/models/sdxl.safetensors"
        );
    }

    #[test]
    fn parses_multi_key_metadata_with_component_vae() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/sdxl.safetensors"))
                .with_source(vae(
                    "/models/vae.safetensors",
                    Some("component=vae;extra=value"),
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
    fn requires_split_import_when_only_placeholder_checkpoint_is_present() {
        let source_set =
            ResolvedInferenceModelSourceSet::new(checkpoint("/models/placeholder.safetensors"));

        let resolved = resolve_vae_sources(&source_set).unwrap();

        match resolved {
            SdxlVaeSources::Checkpoint { path } => {
                assert_eq!(path, PathBuf::from("/models/placeholder.safetensors"));
            }
            other => panic!("expected Checkpoint fallback, got {other:?}"),
        }
    }
}
