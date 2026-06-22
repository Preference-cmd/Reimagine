//! `text.encode` request DTO.

use std::sync::Arc;

use crate::BackendSelectionOverlay;
use crate::ExecutionValue;
use crate::RuntimeClipHandle;
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, ParamValue, RunId, WorkflowId, WorkflowVersion};

/// `text.encode` request.
///
/// Carries a [`RuntimeClipHandle`] for the loaded CLIP bundle and the
/// prompt text. The `text` slot is carried as an
/// [`ExecutionValue`] because prompts arrive through the workflow
/// input pipeline as `ExecutionValue::Param(ParamValue::String)` or
/// `ExecutionValue::Param(ParamValue::Text)`.
#[derive(Debug, Clone)]
pub struct TextEncodeRequest {
    clip: RuntimeClipHandle,
    text: Arc<ExecutionValue>,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
    backend_selection: BackendSelectionOverlay,
}

impl TextEncodeRequest {
    pub fn new(
        clip: RuntimeClipHandle,
        text: Arc<ExecutionValue>,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            clip,
            text,
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

    pub fn clip(&self) -> &RuntimeClipHandle {
        &self.clip
    }

    pub fn text(&self) -> &Arc<ExecutionValue> {
        &self.text
    }

    /// Convenience accessor: extract the prompt string from
    /// `text` when it is a `Param(String | Text)` value.
    pub fn prompt_string(&self) -> Option<String> {
        match self.text.as_ref() {
            ExecutionValue::Param(ParamValue::String(s)) => Some(s.clone()),
            ExecutionValue::Param(ParamValue::Text(s)) => Some(s.clone()),
            _ => None,
        }
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

    /// Consume the request and return its clip handle and text value.
    pub fn into_parts(self) -> (RuntimeClipHandle, Arc<ExecutionValue>) {
        (self.clip, self.text)
    }

    /// Backend affinity observed from the clip handle.
    pub fn backend_affinities(&self) -> Vec<crate::BackendInstance> {
        vec![self.clip.backend_instance().clone()]
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
