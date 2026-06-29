use crate::models::stable_diffusion::sdxl::BurnSdxlContractError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnBackendError {
    DeviceUnavailable { requested: String, reason: String },
    SdxlContract(BurnSdxlContractError),
}

impl std::fmt::Display for BurnBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceUnavailable { requested, reason } => {
                write!(f, "Burn device `{requested}` is unavailable: {reason}")
            }
            Self::SdxlContract(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for BurnBackendError {}

impl From<BurnSdxlContractError> for BurnBackendError {
    fn from(value: BurnSdxlContractError) -> Self {
        Self::SdxlContract(value)
    }
}
