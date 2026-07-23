use crate::model::{NodeId, SlotId, WorkflowInputId, WorkflowOutputId};

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Endpoint {
    NodeSlot { node: NodeId, slot: SlotId },
    WorkflowInput { workflow_input: WorkflowInputId },
    WorkflowOutput { workflow_output: WorkflowOutputId },
}

impl Endpoint {
    pub fn node_slot(node: NodeId, slot: SlotId) -> Self {
        Self::NodeSlot { node, slot }
    }

    pub fn workflow_input(id: WorkflowInputId) -> Self {
        Self::WorkflowInput { workflow_input: id }
    }

    pub fn workflow_output(id: WorkflowOutputId) -> Self {
        Self::WorkflowOutput {
            workflow_output: id,
        }
    }
}
