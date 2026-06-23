use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Kind of model source — checkpoint bundle or a split-component file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ModelSourceKind {
    CheckpointBundle,
    SplitComponent,
}

/// Backend-neutral description of one model source artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedInferenceModelSource {
    kind: ModelSourceKind,
    path: PathBuf,
    metadata: Option<String>,
}

impl ResolvedInferenceModelSource {
    pub fn new(kind: ModelSourceKind, path: PathBuf) -> Self {
        Self {
            kind,
            path,
            metadata: None,
        }
    }
    pub fn kind(&self) -> ModelSourceKind {
        self.kind
    }
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
    pub fn metadata(&self) -> Option<&str> {
        self.metadata.as_deref()
    }
    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }
}

/// A set of resolved model sources for a loadable model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedInferenceModelSourceSet {
    sources: Vec<ResolvedInferenceModelSource>,
}

impl ResolvedInferenceModelSourceSet {
    pub fn new(source: ResolvedInferenceModelSource) -> Self {
        Self {
            sources: vec![source],
        }
    }
    pub fn from_sources(sources: Vec<ResolvedInferenceModelSource>) -> Self {
        Self { sources }
    }
    pub fn sources(&self) -> &[ResolvedInferenceModelSource] {
        &self.sources
    }
    pub fn with_source(mut self, source: ResolvedInferenceModelSource) -> Self {
        self.sources.push(source);
        self
    }
    pub fn is_checkpoint_bundle(&self) -> bool {
        self.sources.len() == 1 && self.sources[0].kind() == ModelSourceKind::CheckpointBundle
    }
}
