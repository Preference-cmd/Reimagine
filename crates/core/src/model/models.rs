use super::ids::ModelId;

/// Canonical, backend-neutral model file format.
///
/// This is the single source of truth for model format variants across
/// the workspace. Both `model-manager` and `inference` re-export this
/// type rather than defining their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModelFormat {
    #[serde(rename = "safetensors")]
    SafeTensors,
    #[serde(rename = "gguf")]
    Gguf,
    #[serde(rename = "pytorch")]
    PyTorch,
    #[serde(rename = "onnx")]
    Onnx,
    #[serde(rename = "unknown")]
    Unknown,
}

impl ModelFormat {
    /// Returns `true` when the format is one the runtime can load.
    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SafeTensors => f.write_str("SafeTensors"),
            Self::Gguf => f.write_str("Gguf"),
            Self::PyTorch => f.write_str("PyTorch"),
            Self::Onnx => f.write_str("Onnx"),
            Self::Unknown => f.write_str("Unknown"),
        }
    }
}

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
