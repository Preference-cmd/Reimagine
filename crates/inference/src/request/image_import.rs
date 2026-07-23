//! `image.import` request DTO.
//!
//! Workspace-safe image source passed from the app-host input
//! resolver. The resolver rejects absolute paths and parent escapes
//! before the request reaches the backend; backends only see the
//! already-authorized [`ResolvedImageSource`].

use std::path::PathBuf;

use crate::BackendSelectionOverlay;
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

/// A workspace-safe image source description.
///
/// Lives in `inference` because the request DTO needs it, but
/// workspace safety (path validation, base-path enforcement) is
/// owned by `app-host`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedImageSource {
    path: PathBuf,
    media_type: String,
    display_name: Option<String>,
}

impl ResolvedImageSource {
    pub fn new(
        path: impl Into<PathBuf>,
        media_type: impl Into<String>,
        display_name: Option<String>,
    ) -> Self {
        Self {
            path: path.into(),
            media_type: media_type.into(),
            display_name,
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn display_name(&self) -> Option<&str> {
        self.display_name.as_deref()
    }
}

/// `image.import` request.
#[derive(Debug, Clone)]
pub struct ImageImportRequest {
    source: ResolvedImageSource,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
    backend_selection: BackendSelectionOverlay,
}

impl ImageImportRequest {
    pub fn new(
        source: ResolvedImageSource,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            source,
            run_id,
            workflow_id,
            workflow_version,
            correlation_id: None,
            node_id,
            backend_selection: BackendSelectionOverlay::new(),
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn source(&self) -> &ResolvedImageSource {
        &self.source
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn workflow_version(&self) -> WorkflowVersion {
        self.workflow_version
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// `image.import` has no upstream backend-affine handle, so it
    /// carries no affinities. The router selects a backend from
    /// config or explicit overlay like `model.load_bundle`.
    pub fn backend_affinities(&self) -> Vec<crate::BackendInstance> {
        Vec::new()
    }

    /// Per-request selection overlay supplied by the runtime.
    pub fn backend_selection_overlay(&self) -> &BackendSelectionOverlay {
        &self.backend_selection
    }

    /// Replace the request's selection overlay (for tests or
    /// runtime-pre-dispatch mutation).
    pub fn set_backend_selection_overlay(&mut self, overlay: BackendSelectionOverlay) {
        self.backend_selection = overlay;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_image_source_carries_path_media_type_and_display_name() {
        let src = ResolvedImageSource::new(
            "/workspace/input/cat.png",
            "image/png",
            Some("cat.png".to_string()),
        );
        assert_eq!(src.path().to_str().unwrap(), "/workspace/input/cat.png");
        assert_eq!(src.media_type(), "image/png");
        assert_eq!(src.display_name(), Some("cat.png"));
    }

    #[test]
    fn request_exposes_source_and_run_metadata() {
        let src = ResolvedImageSource::new("/workspace/input/cat.png", "image/png", None);
        let req = ImageImportRequest::new(
            src,
            RunId::new("run-1"),
            WorkflowId::new("wf-1"),
            WorkflowVersion::new(1),
            NodeId::new("node-1"),
        );
        assert_eq!(
            req.source().path().to_str().unwrap(),
            "/workspace/input/cat.png"
        );
        assert_eq!(req.run_id().as_str(), "run-1");
        assert_eq!(req.node_id().as_str(), "node-1");
        assert!(req.backend_affinities().is_empty());
        assert!(req.correlation_id().is_none());
    }
}
