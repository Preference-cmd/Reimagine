//! `image.save` and `image.preview` response DTOs.

use reimagine_core::model::ArtifactRef;

/// `image.save` response.
#[derive(Debug, Clone)]
pub struct ImageSaveResponse {
    artifact: ArtifactRef,
}

impl ImageSaveResponse {
    pub fn new(artifact: ArtifactRef) -> Self {
        Self { artifact }
    }

    pub fn artifact(&self) -> &ArtifactRef {
        &self.artifact
    }

    /// Consume the response and return its inner artifact reference.
    pub fn into_artifact(self) -> ArtifactRef {
        self.artifact
    }
}

/// `image.preview` response.
#[derive(Debug, Clone)]
pub struct ImagePreviewResponse {
    artifact: ArtifactRef,
}

impl ImagePreviewResponse {
    pub fn new(artifact: ArtifactRef) -> Self {
        Self { artifact }
    }

    pub fn artifact(&self) -> &ArtifactRef {
        &self.artifact
    }

    /// Consume the response and return its inner artifact reference.
    pub fn into_artifact(self) -> ArtifactRef {
        self.artifact
    }
}
