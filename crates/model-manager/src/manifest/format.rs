use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFormat {
    #[serde(rename = "safetensors")]
    Safetensors,
    #[serde(rename = "gguf")]
    Gguf,
    #[serde(rename = "ckpt")]
    Ckpt,
    #[serde(rename = "unknown")]
    Unknown,
}

impl ModelFormat {
    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}
