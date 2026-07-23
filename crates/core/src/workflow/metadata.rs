use crate::model::{SlotId, SlotKind, WorkflowInputId, WorkflowOutputId};

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

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_created_by(mut self, created_by: impl Into<String>) -> Self {
        self.created_by = Some(created_by.into());
        self
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
    #[serde(default = "default_interface_slot_kind")]
    kind: SlotKind,
}

impl WorkflowInputDef {
    pub fn new(id: WorkflowInputId, slot: SlotId, kind: SlotKind) -> Self {
        Self { id, slot, kind }
    }

    pub fn id(&self) -> &WorkflowInputId {
        &self.id
    }

    pub fn slot(&self) -> &SlotId {
        &self.slot
    }

    pub fn kind(&self) -> SlotKind {
        self.kind
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowOutputDef {
    id: WorkflowOutputId,
    slot: SlotId,
    #[serde(default = "default_interface_slot_kind")]
    kind: SlotKind,
}

impl WorkflowOutputDef {
    pub fn new(id: WorkflowOutputId, slot: SlotId, kind: SlotKind) -> Self {
        Self { id, slot, kind }
    }

    pub fn id(&self) -> &WorkflowOutputId {
        &self.id
    }

    pub fn slot(&self) -> &SlotId {
        &self.slot
    }

    pub fn kind(&self) -> SlotKind {
        self.kind
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

    pub fn with_input(mut self, input: WorkflowInputDef) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn with_output(mut self, output: WorkflowOutputDef) -> Self {
        self.outputs.push(output);
        self
    }

    pub fn inputs(&self) -> &[WorkflowInputDef] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[WorkflowOutputDef] {
        &self.outputs
    }

    pub fn input(&self, id: &WorkflowInputId) -> Option<&WorkflowInputDef> {
        self.inputs.iter().find(|input| input.id() == id)
    }

    pub fn output(&self, id: &WorkflowOutputId) -> Option<&WorkflowOutputDef> {
        self.outputs.iter().find(|output| output.id() == id)
    }
}

fn default_interface_slot_kind() -> SlotKind {
    SlotKind::Null
}
