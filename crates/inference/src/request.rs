//! Backend-neutral inference request.
//!
//! An [`InferenceRequest`] owns all the data a backend needs to
//! execute one operation. The request is deliberately self-contained
//! so that the backend call can cross an `.await` boundary without
//! borrowing from [`NodeExecutionContext`](reimagine_runtime::NodeExecutionContext).

use std::collections::HashMap;
use std::sync::Arc;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, ParamValue, RunId, SlotId, WorkflowId, WorkflowVersion};
use reimagine_runtime::RuntimeValue;

use crate::operation::InferenceOperationId;
use crate::resolver::ResolvedInferenceModel;

/// A backend-neutral inference request.
///
/// Inputs are keyed by `SlotId`, not by positional index, so
/// multi-output nodes and multi-model operations can be expressed
/// without relying on declaration order.
///
/// The `models` vector carries resolved model metadata even for
/// single-model operations (e.g. a checkpoint loader carries one
/// entry). This lets the backend know which model is being operated
/// on without introducing a special "single model" variant.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// The stable operation identifier.
    operation_id: InferenceOperationId,
    /// Input values keyed by `SlotId`.
    inputs: HashMap<SlotId, Arc<RuntimeValue>>,
    /// Typed node parameters keyed by `SlotId`.
    params: HashMap<SlotId, ParamValue>,
    /// Resolved model context. One entry for single-model operations;
    /// multiple for future multi-model operations (base + LoRA, etc).
    models: Vec<ResolvedInferenceModel>,
    /// Run context.
    run_id: RunId,
    /// Workflow context.
    workflow_id: WorkflowId,
    /// Workflow version context.
    workflow_version: WorkflowVersion,
    /// Correlation id from the host.
    correlation_id: Option<CorrelationId>,
    /// Originating node id.
    node_id: NodeId,
}

impl InferenceRequest {
    pub fn new(
        operation_id: InferenceOperationId,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            operation_id,
            inputs: HashMap::new(),
            params: HashMap::new(),
            models: Vec::new(),
            run_id,
            workflow_id,
            workflow_version,
            correlation_id: None,
            node_id,
        }
    }

    pub fn with_input(mut self, slot_id: impl Into<SlotId>, value: Arc<RuntimeValue>) -> Self {
        self.inputs.insert(slot_id.into(), value);
        self
    }

    pub fn with_inputs(mut self, inputs: HashMap<SlotId, Arc<RuntimeValue>>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn with_param(mut self, slot_id: impl Into<SlotId>, value: ParamValue) -> Self {
        self.params.insert(slot_id.into(), value);
        self
    }

    pub fn with_params(mut self, params: HashMap<SlotId, ParamValue>) -> Self {
        self.params = params;
        self
    }

    pub fn with_model(mut self, model: ResolvedInferenceModel) -> Self {
        self.models.push(model);
        self
    }

    pub fn with_models(mut self, models: Vec<ResolvedInferenceModel>) -> Self {
        self.models = models;
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn operation_id(&self) -> &InferenceOperationId {
        &self.operation_id
    }

    pub fn inputs(&self) -> &HashMap<SlotId, Arc<RuntimeValue>> {
        &self.inputs
    }

    pub fn params(&self) -> &HashMap<SlotId, ParamValue> {
        &self.params
    }

    pub fn models(&self) -> &[ResolvedInferenceModel] {
        &self.models
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
}
