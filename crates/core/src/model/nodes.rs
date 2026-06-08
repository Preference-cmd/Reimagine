use super::ids::NodeTypeId;
use super::slots::{InputSlotDef, OutputSlotDef};
use super::SlotId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum NodeEffect {
    Pure,
    SideEffect,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeDef {
    type_id: NodeTypeId,
    display_name: String,
    category: String,
    effect: NodeEffect,
    input_slots: Vec<InputSlotDef>,
    output_slots: Vec<OutputSlotDef>,
}

impl NodeDef {
    pub fn new(
        type_id: impl Into<NodeTypeId>,
        display_name: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        Self {
            type_id: type_id.into(),
            display_name: display_name.into(),
            category: category.into(),
            effect: NodeEffect::Pure,
            input_slots: Vec::new(),
            output_slots: Vec::new(),
        }
    }

    pub fn with_effect(mut self, effect: NodeEffect) -> Self {
        self.effect = effect;
        self
    }

    pub fn with_input_slot(mut self, input: InputSlotDef) -> Self {
        self.input_slots.push(input);
        self
    }

    pub fn with_output_slot(mut self, output: OutputSlotDef) -> Self {
        self.output_slots.push(output);
        self
    }

    pub fn type_id(&self) -> &NodeTypeId {
        &self.type_id
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn category(&self) -> &str {
        &self.category
    }

    pub fn effect(&self) -> NodeEffect {
        self.effect
    }

    pub fn input_slots(&self) -> &[InputSlotDef] {
        &self.input_slots
    }

    pub fn output_slots(&self) -> &[OutputSlotDef] {
        &self.output_slots
    }

    pub fn input_slot(&self, id: &SlotId) -> Option<&InputSlotDef> {
        self.input_slots.iter().find(|slot| slot.id() == id)
    }

    pub fn output_slot(&self, id: &SlotId) -> Option<&OutputSlotDef> {
        self.output_slots.iter().find(|slot| slot.id() == id)
    }
}

pub trait NodeCatalog {
    fn get(&self, type_id: &NodeTypeId) -> Option<&NodeDef>;
}
