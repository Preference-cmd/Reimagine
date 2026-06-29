#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnBackendError {
    DeviceUnavailable { requested: String, reason: String },
}

impl std::fmt::Display for BurnBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceUnavailable { requested, reason } => {
                write!(f, "Burn device `{requested}` is unavailable: {reason}")
            }
        }
    }
}

impl std::error::Error for BurnBackendError {}
