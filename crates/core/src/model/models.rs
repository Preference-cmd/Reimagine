use super::ids::ModelId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelFamily {
    Sdxl,
    Sd15,
    Flux,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelRole {
    CheckpointBundle,
    Denoiser,
    TextEncoder,
    Vae,
    Scheduler,
    Lora,
    ControlNet,
    Upscaler,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModelRef {
    id: ModelId,
    family: ModelFamily,
    role: ModelRole,
}

impl ModelRef {
    pub fn new(id: ModelId, family: ModelFamily, role: ModelRole) -> Self {
        Self { id, family, role }
    }

    pub fn id(&self) -> &ModelId {
        &self.id
    }

    pub fn family(&self) -> ModelFamily {
        self.family
    }

    pub fn role(&self) -> ModelRole {
        self.role
    }
}
