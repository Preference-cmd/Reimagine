use crate::model::{SlotId, WorkflowInputId, WorkflowOutputId};

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct WorkflowMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_by: Option<String>,
}

impl WorkflowMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn created_by(&self) -> Option<&str> {
        self.created_by.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowInputDef {
    id: WorkflowInputId,
    slot: SlotId,
}

impl WorkflowInputDef {
    pub fn new(id: WorkflowInputId, slot: SlotId) -> Self {
        Self { id, slot }
    }

    pub fn id(&self) -> &WorkflowInputId {
        &self.id
    }

    pub fn slot(&self) -> &SlotId {
        &self.slot
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowOutputDef {
    id: WorkflowOutputId,
    slot: SlotId,
}

impl WorkflowOutputDef {
    pub fn new(id: WorkflowOutputId, slot: SlotId) -> Self {
        Self { id, slot }
    }

    pub fn id(&self) -> &WorkflowOutputId {
        &self.id
    }

    pub fn slot(&self) -> &SlotId {
        &self.slot
    }
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct WorkflowInterface {
    inputs: Vec<WorkflowInputDef>,
    outputs: Vec<WorkflowOutputDef>,
}

impl WorkflowInterface {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inputs(&self) -> &[WorkflowInputDef] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[WorkflowOutputDef] {
        &self.outputs
    }
}
