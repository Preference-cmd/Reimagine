//! Per-node execution context passed to `NodeExecutor::execute`.

use std::collections::HashMap;
use std::sync::Arc;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    NodeId, NodeTypeId, ParamValue, RunId, SlotId, WorkflowId, WorkflowVersion,
};

use crate::artifacts::NodeArtifactCapability;
use crate::cancellation::CancellationToken;
use crate::value::RuntimeValue;

/// Resolved input values for a single node, keyed by input `SlotId`.
#[derive(Debug, Clone, Default)]
pub struct NodeInputs {
    values: HashMap<SlotId, Arc<RuntimeValue>>,
}

impl NodeInputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, slot_id: impl Into<SlotId>, value: Arc<RuntimeValue>) {
        self.values.insert(slot_id.into(), value);
    }

    pub fn get(&self, slot_id: &SlotId) -> Option<&Arc<RuntimeValue>> {
        self.values.get(slot_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SlotId, &Arc<RuntimeValue>)> {
        self.values.iter()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Resolved node parameters (literal/typed), keyed by input `SlotId`.
///
/// Parameters are values that came from the plan as `Param` or `Default`
/// bindings — the executor should not need to reach back into the workflow
/// for them.
#[derive(Debug, Clone, Default)]
pub struct NodeParams {
    values: HashMap<SlotId, ParamValue>,
}

impl NodeParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, slot_id: impl Into<SlotId>, value: ParamValue) {
        self.values.insert(slot_id.into(), value);
    }

    pub fn get(&self, slot_id: &SlotId) -> Option<&ParamValue> {
        self.values.get(slot_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SlotId, &ParamValue)> {
        self.values.iter()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Per-node execution context. Read-only from the executor's perspective.
pub struct NodeExecutionContext {
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
    type_id: NodeTypeId,
    inputs: NodeInputs,
    params: NodeParams,
    artifacts: NodeArtifactCapability,
    cancellation: CancellationToken,
    started_at: Timestamp,
}

impl NodeExecutionContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        correlation_id: Option<CorrelationId>,
        node_id: NodeId,
        type_id: NodeTypeId,
        inputs: NodeInputs,
        params: NodeParams,
        artifacts: NodeArtifactCapability,
        cancellation: CancellationToken,
        started_at: Timestamp,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            workflow_version,
            correlation_id,
            node_id,
            type_id,
            inputs,
            params,
            artifacts,
            cancellation,
            started_at,
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

    pub fn type_id(&self) -> &NodeTypeId {
        &self.type_id
    }

    pub fn inputs(&self) -> &NodeInputs {
        &self.inputs
    }

    pub fn params(&self) -> &NodeParams {
        &self.params
    }

    pub fn artifacts(&self) -> &NodeArtifactCapability {
        &self.artifacts
    }

    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    pub fn started_at(&self) -> &Timestamp {
        &self.started_at
    }
}
