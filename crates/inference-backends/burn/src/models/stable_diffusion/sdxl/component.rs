use std::fmt;

use serde::{Deserialize, Serialize};

use super::contract::BurnSdxlComponentContract;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BurnSdxlComponentRole {
    #[serde(rename = "diffusion")]
    Diffusion,
    #[serde(rename = "vae")]
    Vae,
    #[serde(rename = "text_encoder")]
    TextEncoder,
    #[serde(rename = "text_encoder_2")]
    TextEncoder2,
}

impl BurnSdxlComponentRole {
    pub const fn all() -> [Self; 4] {
        [
            Self::Diffusion,
            Self::Vae,
            Self::TextEncoder,
            Self::TextEncoder2,
        ]
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Diffusion => "diffusion",
            Self::Vae => "vae",
            Self::TextEncoder => "text_encoder",
            Self::TextEncoder2 => "text_encoder_2",
        }
    }

    pub fn contract(self) -> BurnSdxlComponentContract {
        BurnSdxlComponentContract::new(self)
    }
}

impl TryFrom<&str> for BurnSdxlComponentRole {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "diffusion" => Ok(Self::Diffusion),
            "vae" => Ok(Self::Vae),
            "text_encoder" => Ok(Self::TextEncoder),
            "text_encoder_2" => Ok(Self::TextEncoder2),
            other => Err(format!("unsupported Burn SDXL component role `{other}`")),
        }
    }
}

impl fmt::Display for BurnSdxlComponentRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnTensorDType {
    F32,
    F16,
    Bf16,
    Unsupported(String),
}

impl BurnTensorDType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::F32 => "fp32",
            Self::F16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Unsupported(value) => value,
        }
    }

    pub const fn is_supported(&self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnTensorShapeSpec {
    Rank(usize),
}

impl BurnTensorShapeSpec {
    pub const fn rank(self) -> usize {
        match self {
            Self::Rank(rank) => rank,
        }
    }

    pub fn matches(&self, shape: &[usize]) -> bool {
        shape.len() == self.rank()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurnTensorSpec {
    pub key: &'static str,
    pub shape: BurnTensorShapeSpec,
    pub required: bool,
    pub notes: &'static str,
}

impl BurnTensorSpec {
    pub const fn required_rank(key: &'static str, rank: usize, notes: &'static str) -> Self {
        Self {
            key,
            shape: BurnTensorShapeSpec::Rank(rank),
            required: true,
            notes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnTensorInventoryEntry {
    pub key: String,
    pub shape: Vec<usize>,
    pub dtype: BurnTensorDType,
}

impl BurnTensorInventoryEntry {
    pub fn new(key: impl Into<String>, shape: Vec<usize>, dtype: BurnTensorDType) -> Self {
        Self {
            key: key.into(),
            shape,
            dtype,
        }
    }
}
