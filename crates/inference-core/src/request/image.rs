//! `image.save` and `image.preview` request DTOs.

use crate::RuntimeImage;
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

/// Optional filename prefix for `image.save`. `None` falls back to
/// the backend's default prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilenamePrefix {
    Default,
    Custom(String),
}

impl FilenamePrefix {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Default => "reimagine",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl Default for FilenamePrefix {
    fn default() -> Self {
        Self::Default
    }
}

/// `image.save` request.
#[derive(Debug, Clone)]
pub struct ImageSaveRequest {
    image: RuntimeImage,
    filename_prefix: FilenamePrefix,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
}

impl ImageSaveRequest {
    pub fn new(
        image: RuntimeImage,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            image,
            filename_prefix: FilenamePrefix::Default,
            run_id,
            workflow_id,
            workflow_version,
            correlation_id: None,
            node_id,
        }
    }

    pub fn with_filename_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.filename_prefix = FilenamePrefix::Custom(prefix.into());
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn image(&self) -> &RuntimeImage {
        &self.image
    }

    /// Consume the request and return its [`RuntimeImage`] handle.
    pub fn into_image(self) -> RuntimeImage {
        self.image
    }

    pub fn filename_prefix(&self) -> &FilenamePrefix {
        &self.filename_prefix
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

    pub fn backend_affinities(&self) -> Vec<crate::BackendKind> {
        vec![self.image.payload().backend().clone()]
    }
}

/// `image.preview` request.
#[derive(Debug, Clone)]
pub struct ImagePreviewRequest {
    image: RuntimeImage,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
}

impl ImagePreviewRequest {
    pub fn new(
        image: RuntimeImage,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            image,
            run_id,
            workflow_id,
            workflow_version,
            correlation_id: None,
            node_id,
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn image(&self) -> &RuntimeImage {
        &self.image
    }

    /// Consume the request and return its [`RuntimeImage`] handle.
    pub fn into_image(self) -> RuntimeImage {
        self.image
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

    pub fn backend_affinities(&self) -> Vec<crate::BackendKind> {
        vec![self.image.payload().backend().clone()]
    }
}
