use candle_core::Device;

use crate::error::CandleBackendError;

/// Backend-side device policy.
///
/// V1 supports CPU and Metal (Apple Silicon). The actual Candle
/// `Device` is constructed via [`CandleDevice::try_build_device`] so
/// the config layer can stay serializable and easy to test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleDevice {
    label: String,
}

impl CandleDevice {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// Build a Candle [`Device`] that matches this policy label.
    ///
    /// Returns `Err(CandleBackendError::DeviceUnavailable)` if the
    /// label is unknown or the device cannot be constructed on this
    /// host (e.g. Metal requested on a non-Apple system).
    pub fn try_build_device(&self) -> Result<Device, CandleBackendError> {
        match self.label.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "metal" | "mps" => {
                Device::new_metal(0).map_err(|err| CandleBackendError::DeviceUnavailable {
                    requested: self.label.clone(),
                    reason: err.to_string(),
                })
            }
            other => Err(CandleBackendError::DeviceUnavailable {
                requested: other.to_string(),
                reason: "unsupported device label; expected `cpu`, `metal`, or `mps`".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_stores_label() {
        let device = CandleDevice::new("cpu");
        assert_eq!(device.label(), "cpu");
    }

    #[test]
    fn device_equality() {
        assert_eq!(CandleDevice::new("cpu"), CandleDevice::new("cpu"));
        assert_ne!(CandleDevice::new("cpu"), CandleDevice::new("mps"));
    }

    #[test]
    fn try_build_device_returns_cpu_for_cpu_label() {
        let device = CandleDevice::new("cpu").try_build_device().unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn try_build_device_rejects_unknown_label() {
        let err = CandleDevice::new("tpu").try_build_device().unwrap_err();
        match err {
            CandleBackendError::DeviceUnavailable { requested, .. } => {
                assert_eq!(requested, "tpu");
            }
            other => panic!("expected DeviceUnavailable, got {other:?}"),
        }
    }
}
