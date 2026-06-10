//! Inputs supplied to a run that are not part of the prepared execution plan.

use std::collections::HashMap;

use reimagine_core::model::{NodeId, ParamValue, SlotId, WorkflowInputId};

use crate::value::RuntimeValue;

/// Map of node inputs that are not resolved from plan edges.
///
/// Keyed by `(node_id, slot_id)`. Values are `Arc<RuntimeValue>` so the
/// caller can share backend-owned payload handles cheaply.
#[derive(Debug, Default, Clone)]
pub struct RunInputs {
    values: HashMap<(NodeId, SlotId), std::sync::Arc<RuntimeValue>>,
    node_params: HashMap<(NodeId, SlotId), ParamValue>,
    workflow_inputs: HashMap<WorkflowInputId, std::sync::Arc<RuntimeValue>>,
}

impl RunInputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        node_id: impl Into<NodeId>,
        slot_id: impl Into<SlotId>,
        value: std::sync::Arc<RuntimeValue>,
    ) {
        self.values.insert((node_id.into(), slot_id.into()), value);
    }

    pub fn get(&self, node_id: &NodeId, slot_id: &SlotId) -> Option<&std::sync::Arc<RuntimeValue>> {
        self.values.get(&(node_id.clone(), slot_id.clone()))
    }

    pub fn insert_node_param(
        &mut self,
        node_id: impl Into<NodeId>,
        slot_id: impl Into<SlotId>,
        value: ParamValue,
    ) {
        self.node_params
            .insert((node_id.into(), slot_id.into()), value);
    }

    pub fn node_param(&self, node_id: &NodeId, slot_id: &SlotId) -> Option<&ParamValue> {
        self.node_params.get(&(node_id.clone(), slot_id.clone()))
    }

    pub fn insert_workflow_input(
        &mut self,
        workflow_input_id: impl Into<WorkflowInputId>,
        value: std::sync::Arc<RuntimeValue>,
    ) {
        self.workflow_inputs.insert(workflow_input_id.into(), value);
    }

    pub fn workflow_input(
        &self,
        workflow_input_id: &WorkflowInputId,
    ) -> Option<&std::sync::Arc<RuntimeValue>> {
        self.workflow_inputs.get(workflow_input_id)
    }
}
