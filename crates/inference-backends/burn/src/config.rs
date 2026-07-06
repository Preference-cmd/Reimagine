use std::path::PathBuf;

use crate::device::BurnDevice;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnBackendConfig {
    models_dir: PathBuf,
    output_dir: PathBuf,
    device: BurnDevice,
    tokenizer_root: Option<PathBuf>,
}

impl BurnBackendConfig {
    pub fn new(models_dir: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            output_dir: output_dir.into(),
            device: BurnDevice::default_device(),
            tokenizer_root: None,
        }
    }

    pub fn with_device(mut self, device: BurnDevice) -> Self {
        self.device = device;
        self
    }

    /// Override the SDXL tokenizer asset root.
    ///
    /// Library code must resolve bundled tokenizer assets through this
    /// explicit seam instead of the process current working directory.
    /// When `None`, the bundled assets shipped under
    /// `assets/tokenizers/stable_diffusion/sdxl` (relative to the
    /// workspace) are used.
    pub fn with_tokenizer_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.tokenizer_root = Some(root.into());
        self
    }

    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }

    pub fn output_dir(&self) -> &PathBuf {
        &self.output_dir
    }

    pub fn device(&self) -> &BurnDevice {
        &self.device
    }

    pub fn device_label(&self) -> &str {
        self.device.label()
    }

    pub fn tokenizer_root(&self) -> Option<&PathBuf> {
        self.tokenizer_root.as_ref()
    }
}
