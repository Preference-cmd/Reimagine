use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnConversionComponent {
    pub role: BurnConversionComponentRole,
    pub path: PathBuf,
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnConversionReport {
    pub output_components: Vec<BurnConversionComponent>,
    pub mapped_tensor_count: usize,
    pub source_layout: String,
}

pub trait BurnCheckpointConverter: Send + Sync + 'static {
    fn convert(
        &self,
        source_path: &Path,
        model_id: &str,
        model_root: &Path,
    ) -> Result<BurnConversionReport, String>;
}
