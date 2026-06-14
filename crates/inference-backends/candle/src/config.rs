use std::path::PathBuf;

use crate::device::CandleDevice;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleBackendConfig {
    models_dir: PathBuf,
    device: CandleDevice,
}

impl CandleBackendConfig {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            device: CandleDevice::new("cpu"),
        }
    }

    pub fn with_device(mut self, device: CandleDevice) -> Self {
        self.device = device;
        self
    }

    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }

    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    pub fn device_label(&self) -> &str {
        self.device.label()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_to_cpu_device() {
        let config = CandleBackendConfig::new("/models");
        assert_eq!(config.device().label(), "cpu");
        assert_eq!(config.device_label(), "cpu");
    }

    #[test]
    fn config_stores_models_dir() {
        let config = CandleBackendConfig::new("/models");
        assert_eq!(config.models_dir(), &PathBuf::from("/models"));
    }

    #[test]
    fn config_with_device_round_trips() {
        let config = CandleBackendConfig::new("/models").with_device(CandleDevice::new("mps"));
        assert_eq!(config.device().label(), "mps");
        assert_eq!(config.device_label(), "mps");
    }
}
