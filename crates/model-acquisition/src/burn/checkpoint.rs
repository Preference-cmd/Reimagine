//! Burn checkpoint conversion types.
//!
//! Provides the trait and data types for converting checkpoint files
//! (`.ckpt`, `.safetensors`) into Burn-native split component layouts.

use std::path::{Path, PathBuf};

/// A single converted component produced by a Burn checkpoint conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnConversionComponent {
    pub role: BurnConversionComponentRole,
    pub path: PathBuf,
}

/// The role of a component within a Burn checkpoint conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnConversionComponentRole {
    Diffusion,
    Vae,
    TextEncoder,
    TextEncoder2,
}

impl BurnConversionComponentRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Diffusion => "diffusion",
            Self::Vae => "vae",
            Self::TextEncoder => "text_encoder",
            Self::TextEncoder2 => "text_encoder_2",
        }
    }
}

/// Summary report produced by a Burn checkpoint conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnConversionReport {
    pub output_components: Vec<BurnConversionComponent>,
    pub mapped_tensor_count: usize,
    pub source_layout: String,
}

/// Trait for converting a checkpoint file into Burn-native split components.
///
/// Implementations handle the format-specific logic of reading a checkpoint
/// (single-file `.safetensors`, `.ckpt`, etc.) and writing out the
/// per-component safetensors files in Burn's expected layout.
pub trait BurnCheckpointConverter: Send + Sync + 'static {
    fn convert(
        &self,
        source_path: &Path,
        model_id: &str,
        model_root: &Path,
    ) -> Result<BurnConversionReport, String>;
}
