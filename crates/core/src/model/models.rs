use super::ids::ModelId;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModelSeries(String);

impl ModelSeries {
    pub fn new(series: impl Into<String>) -> Self {
        Self(series.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelSeries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ModelSeries {
    fn from(series: String) -> Self {
        Self(series)
    }
}

impl From<&str> for ModelSeries {
    fn from(series: &str) -> Self {
        Self(series.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModelVariant(String);

impl ModelVariant {
    pub fn new(variant: impl Into<String>) -> Self {
        Self(variant.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ModelVariant {
    fn from(variant: String) -> Self {
        Self(variant)
    }
}

impl From<&str> for ModelVariant {
    fn from(variant: &str) -> Self {
        Self(variant.to_owned())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelRole {
    CheckpointBundle,
    DiffusionModel,
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
    model_series: ModelSeries,
    variant: ModelVariant,
    role: ModelRole,
}

impl ModelRef {
    pub fn new(
        id: ModelId,
        model_series: ModelSeries,
        variant: ModelVariant,
        role: ModelRole,
    ) -> Self {
        Self {
            id,
            model_series,
            variant,
            role,
        }
    }

    pub fn id(&self) -> &ModelId {
        &self.id
    }

    pub fn model_series(&self) -> &ModelSeries {
        &self.model_series
    }

    pub fn variant(&self) -> &ModelVariant {
        &self.variant
    }

    pub fn role(&self) -> ModelRole {
        self.role
    }
}
