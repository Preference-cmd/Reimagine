//! `model.load_bundle` request DTO.

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

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
    pub fn backend_affinities(&self) -> Vec<reimagine_core::BackendKind> {
        Vec::new()
    }
}
