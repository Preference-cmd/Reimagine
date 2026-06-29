use burn_ndarray::NdArrayDevice;

use crate::error::BurnBackendError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnDevice {
    label: String,
}

impl BurnDevice {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn try_build_device(&self) -> Result<NdArrayDevice, BurnBackendError> {
        match self.label.as_str() {
            "cpu" => Ok(NdArrayDevice::Cpu),
            label => Err(BurnBackendError::DeviceUnavailable {
                requested: label.to_owned(),
                reason: "Burn skeleton only supports the burn-ndarray CPU device".to_owned(),
            }),
        }
    }
}
