use super::SlotId;
use super::ids::NodeTypeId;
use super::slots::{InputSlotDef, OutputSlotDef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum NodeEffect {
    Pure,
    SideEffect,
}

/// Model family / architecture tag for node compatibility.
///
/// Used to declare which model architectures a node works with.
/// `None` in [`NodeResourceRequirements`] means the node is model-agnostic.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModelFamily(String);

impl ModelFamily {
    pub fn new(family: impl Into<String>) -> Self {
        Self(family.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn stable_diffusion() -> Self {
        Self("stable_diffusion".into())
    }

    pub fn flux() -> Self {
        Self("flux".into())
    }

    pub fn any() -> Self {
        Self("any".into())
    }
}

/// Backend capability required by a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BackendCapability {
    Gpu,
    Cpu,
    Metal,
    Cuda,
    WebGpu,
    ComputeShader,
}

/// Roles of model components a node needs loaded.
///
/// Used in [`NodeResourceRequirements`] to declare which pre-loaded
/// components a node requires from the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ComponentRole {
    CheckpointBundle,
    DiffusionModel,
    TextEncoder,
    SecondaryTextEncoder,
    VaeDecoder,
    VaeEncoder,
    Lora,
    ControlNet,
    Upscaler,
}

/// Resource requirements a node places on the backend runtime.
///
/// Attached to [`NodeDef`] to enable the runtime to know what to pre-load,
/// the backend to know what to keep in memory, and the scheduler to
/// prefetch for the next stage.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NodeResourceRequirements {
    model_family: Option<ModelFamily>,
    required_capabilities: Vec<BackendCapability>,
    estimated_vram_bytes: Option<u64>,
    required_components: Vec<ComponentRole>,
}

impl NodeResourceRequirements {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_family(mut self, family: ModelFamily) -> Self {
        self.model_family = Some(family);
        self
    }

    pub fn required_capabilities(mut self, caps: Vec<BackendCapability>) -> Self {
        self.required_capabilities = caps;
        self
    }

    pub fn estimated_vram_bytes(mut self, bytes: u64) -> Self {
        self.estimated_vram_bytes = Some(bytes);
        self
    }

    pub fn required_components(mut self, components: Vec<ComponentRole>) -> Self {
        self.required_components = components;
        self
    }

    pub fn get_model_family(&self) -> Option<&ModelFamily> {
        self.model_family.as_ref()
    }

    pub fn get_required_capabilities(&self) -> &[BackendCapability] {
        &self.required_capabilities
    }

    pub fn get_estimated_vram_bytes(&self) -> Option<u64> {
        self.estimated_vram_bytes
    }

    pub fn get_required_components(&self) -> &[ComponentRole] {
        &self.required_components
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeDef {
    type_id: NodeTypeId,
    display_name: String,
    category: String,
    effect: NodeEffect,
    input_slots: Vec<InputSlotDef>,
    output_slots: Vec<OutputSlotDef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resource_requirements: Option<NodeResourceRequirements>,
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
            resource_requirements: None,
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

    pub fn with_resource_requirements(mut self, requirements: NodeResourceRequirements) -> Self {
        self.resource_requirements = Some(requirements);
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

    pub fn resource_requirements(&self) -> Option<&NodeResourceRequirements> {
        self.resource_requirements.as_ref()
    }
}

pub trait NodeCatalog {
    fn get(&self, type_id: &NodeTypeId) -> Option<&NodeDef>;
}
