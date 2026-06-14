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
}
