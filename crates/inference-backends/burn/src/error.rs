use std::path::PathBuf;

use crate::models::stable_diffusion::sdxl::BurnSdxlContractError;
use crate::models::stable_diffusion::sdxl::BurnTokenizerError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnBackendError {
    DeviceUnavailable {
        requested: String,
        reason: String,
    },
    InvalidRequest(String),
    UnsupportedSourceLayout(String),
    MissingComponent(String),
    DuplicateComponent(String),
    ComponentMetadataMismatch {
        path: PathBuf,
        expected: String,
        found: String,
    },
    ComponentValidation {
        path: PathBuf,
        source: BurnSdxlContractError,
    },
    ComponentRead {
        path: PathBuf,
        message: String,
    },
    CacheIncompatible(String),
    BackendNotImplemented(String),
    Tokenizer(BurnTokenizerError),
    SdxlContract(BurnSdxlContractError),
}

impl std::fmt::Display for BurnBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceUnavailable { requested, reason } => {
                write!(f, "Burn device `{requested}` is unavailable: {reason}")
            }
            Self::InvalidRequest(message) => write!(f, "invalid Burn request: {message}"),
            Self::UnsupportedSourceLayout(message) => {
                write!(f, "unsupported Burn source layout: {message}")
            }
            Self::MissingComponent(component) => {
                write!(f, "missing Burn SDXL component `{component}`")
            }
            Self::DuplicateComponent(component) => {
                write!(f, "duplicate Burn SDXL component `{component}`")
            }
            Self::ComponentMetadataMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "Burn component metadata mismatch for `{}`: expected {expected}, found {found}",
                path.display()
            ),
            Self::ComponentValidation { path, source } => write!(
                f,
                "Burn component validation failed for `{}`: {source}",
                path.display()
            ),
            Self::ComponentRead { path, message } => write!(
                f,
                "Burn component read failed for `{}`: {message}",
                path.display()
            ),
            Self::CacheIncompatible(message) => {
                write!(f, "Burn model cache incompatible: {message}")
            }
            Self::BackendNotImplemented(capability) => {
                write!(f, "Burn backend does not implement `{capability}`")
            }
            Self::Tokenizer(error) => write!(f, "{error}"),
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

impl From<BurnTokenizerError> for BurnBackendError {
    fn from(value: BurnTokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}
