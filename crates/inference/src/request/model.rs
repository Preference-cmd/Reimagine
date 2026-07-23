//! `model.load_bundle` request DTO.

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

use crate::BackendSelectionOverlay;
use crate::resolver::ResolvedInferenceModel;

/// `model.load_bundle` request.
///
/// Carries the resolved model metadata plus the run/node/correlation
/// context the backend needs to scope its work.
#[derive(Debug, Clone)]
pub struct LoadBundleRequest {
    resolved_model: ResolvedInferenceModel,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
    backend_selection: BackendSelectionOverlay,
}

impl LoadBundleRequest {
    pub fn new(
        resolved_model: ResolvedInferenceModel,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            resolved_model,
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

    pub fn resolved_model(&self) -> &ResolvedInferenceModel {
        &self.resolved_model
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

    /// Backend affinity observed on the resolved model's handles.
    ///
    /// `LoadBundleRequest` has no executable inputs yet, so the
    /// affinity derives solely from the resolved model. Today the
    /// resolved model carries no backend affinity; V2 may grow that
    /// constraint.
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
