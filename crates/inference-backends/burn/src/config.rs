use std::path::PathBuf;

use crate::device::BurnDevice;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnBackendConfig {
    models_dir: PathBuf,
    output_dir: PathBuf,
    device: BurnDevice,
}

impl BurnBackendConfig {
    pub fn new(models_dir: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            output_dir: output_dir.into(),
            device: BurnDevice::new("cpu"),
        }
    }

    pub fn with_device(mut self, device: BurnDevice) -> Self {
        self.device = device;
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
}
