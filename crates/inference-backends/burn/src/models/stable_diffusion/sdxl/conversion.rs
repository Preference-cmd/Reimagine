use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::component::{BurnSdxlComponentRole, BurnTensorDType, BurnTensorInventoryEntry};
use super::contract::{BURN_SDXL_COMPONENT_CONTRACT_VERSION, BurnDTypePolicy};
use super::metadata::metadata_keys;
use super::validation::{BurnSdxlComponentValidationReport, BurnSdxlContractError};

pub const BURN_SDXL_SYNTHETIC_SOURCE_LAYOUT: &str = "synthetic_burn_native";
pub const BURN_SDXL_CONVERSION_REPORT_FILE: &str = "conversion-report.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnTensorSource {
    Zeros,
    Data(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSyntheticTensor {
    pub key: String,
    pub shape: Vec<usize>,
    pub dtype: BurnTensorDType,
    pub source: BurnTensorSource,
}

impl BurnSyntheticTensor {
    pub fn zeros(key: impl Into<String>, shape: Vec<usize>, dtype: BurnTensorDType) -> Self {
        Self {
            key: key.into(),
            shape,
            dtype,
            source: BurnTensorSource::Zeros,
        }
    }
}

impl From<&BurnSyntheticTensor> for BurnTensorInventoryEntry {
    fn from(value: &BurnSyntheticTensor) -> Self {
        Self::new(value.key.clone(), value.shape.clone(), value.dtype.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlSyntheticComponent {
    pub role: BurnSdxlComponentRole,
    pub dtype_policy: BurnDTypePolicy,
    pub tensors: Vec<BurnSyntheticTensor>,
}

impl BurnSdxlSyntheticComponent {
    pub fn metadata(&self) -> BTreeMap<String, String> {
        let contract = self.role.contract();

        BTreeMap::from([
            (
                metadata_keys::CONTRACT.to_owned(),
                contract.contract_name().to_owned(),
            ),
            (
                metadata_keys::CONTRACT_VERSION.to_owned(),
                contract.contract_version.to_string(),
            ),
            (
                metadata_keys::BACKEND.to_owned(),
                contract.backend().to_owned(),
            ),
            (
                metadata_keys::MODEL_SERIES.to_owned(),
                contract.model_series().to_owned(),
            ),
            (
                metadata_keys::VARIANT.to_owned(),
                contract.variant().to_owned(),
            ),
            (
                metadata_keys::COMPONENT_ROLE.to_owned(),
                self.role.as_str().to_owned(),
            ),
            (
                metadata_keys::TENSOR_LAYOUT.to_owned(),
                contract.tensor_layout().to_owned(),
            ),
            (
                metadata_keys::DTYPE_POLICY.to_owned(),
                self.dtype_policy.as_str().to_owned(),
            ),
        ])
    }

    pub fn inventory(&self) -> Vec<BurnTensorInventoryEntry> {
        self.tensors
            .iter()
            .map(BurnTensorInventoryEntry::from)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticSdxlConversionPlan {
    pub source_identity: String,
    pub components: Vec<BurnSdxlSyntheticComponent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedBurnSdxlComponent {
    pub metadata: BTreeMap<String, String>,
    pub inventory: Vec<BurnTensorInventoryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurnSdxlOutputComponentReport {
    pub role: BurnSdxlComponentRole,
    pub path: String,
    pub tensor_count: usize,
    pub validated_required_tensor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurnSdxlPackageReport {
    pub schema_version: u32,
    pub layout: String,
    pub converter_version: String,
    pub package_root: String,
    pub created_at: Option<u64>,
    pub source: BurnSdxlPackageSourceReport,
    pub target: BurnSdxlPackageTargetReport,
    pub components: Vec<BurnSdxlPackageComponentReport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurnSdxlPackageSourceReport {
    pub source_model_id: String,
    pub source_layout: String,
    pub source_fingerprint: String,
    pub fingerprint_kind: String,
    pub source_files: Vec<BurnSdxlPackageSourceFileReport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurnSdxlPackageSourceFileReport {
    pub relative_path: String,
    pub size_bytes: u64,
    pub modified_at: Option<u64>,
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurnSdxlPackageTargetReport {
    pub backend: String,
    pub contract: String,
    pub contract_version: u32,
    pub model_series: String,
    pub variant: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurnSdxlPackageComponentReport {
    pub component_role: BurnSdxlComponentRole,
    pub model_role: String,
    pub relative_path: String,
    pub format: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurnSdxlConversionReport {
    pub source_identity: String,
    pub source_layout: String,
    pub target_contract_version: u32,
    pub output_components: Vec<BurnSdxlOutputComponentReport>,
    pub mapped_tensor_count: usize,
    pub ignored_tensor_families: Vec<String>,
    pub diagnostics: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package: Option<BurnSdxlPackageReport>,
}

impl BurnSdxlConversionReport {
    pub fn synthetic(source_identity: impl Into<String>) -> Self {
        Self {
            source_identity: source_identity.into(),
            source_layout: BURN_SDXL_SYNTHETIC_SOURCE_LAYOUT.to_owned(),
            target_contract_version: BURN_SDXL_COMPONENT_CONTRACT_VERSION,
            output_components: Vec::new(),
            mapped_tensor_count: 0,
            ignored_tensor_families: Vec::new(),
            diagnostics: Vec::new(),
            package: None,
        }
    }
}

#[derive(Debug)]
pub enum BurnSdxlConversionError {
    Validation {
        role: BurnSdxlComponentRole,
        source: BurnSdxlContractError,
    },
    InvalidComponentSet {
        reason: String,
    },
    InvalidTensorData {
        key: String,
        reason: String,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    SafetensorsWrite {
        path: PathBuf,
        source: safetensors::SafeTensorError,
    },
    SafetensorsReadBack {
        path: PathBuf,
        source: safetensors::SafeTensorError,
    },
}

impl std::fmt::Display for BurnSdxlConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation { role, source } => {
                write!(f, "invalid synthetic Burn SDXL {role} component: {source}")
            }
            Self::InvalidComponentSet { reason } => {
                write!(f, "invalid synthetic Burn SDXL component set: {reason}")
            }
            Self::InvalidTensorData { key, reason } => {
                write!(f, "invalid synthetic Burn SDXL tensor `{key}`: {reason}")
            }
            Self::Io { path, source } => {
                write!(
                    f,
                    "Burn SDXL conversion I/O failed at `{}`: {source}",
                    path.display()
                )
            }
            Self::Json { path, source } => {
                write!(
                    f,
                    "Burn SDXL conversion report JSON failed at `{}`: {source}",
                    path.display()
                )
            }
            Self::SafetensorsWrite { path, source } => {
                write!(
                    f,
                    "Burn SDXL safetensors write failed at `{}`: {source}",
                    path.display()
                )
            }
            Self::SafetensorsReadBack { path, source } => {
                write!(
                    f,
                    "Burn SDXL safetensors read-back failed at `{}`: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for BurnSdxlConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation { source, .. } => Some(source),
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::SafetensorsWrite { source, .. } => Some(source),
            Self::SafetensorsReadBack { source, .. } => Some(source),
            Self::InvalidComponentSet { .. } | Self::InvalidTensorData { .. } => None,
        }
    }
}

pub(crate) fn validation_error(
    role: BurnSdxlComponentRole,
    source: BurnSdxlContractError,
) -> BurnSdxlConversionError {
    BurnSdxlConversionError::Validation { role, source }
}

pub(crate) fn output_component_report(
    role: BurnSdxlComponentRole,
    path: String,
    tensor_count: usize,
    validation: &BurnSdxlComponentValidationReport,
) -> BurnSdxlOutputComponentReport {
    BurnSdxlOutputComponentReport {
        role,
        path,
        tensor_count,
        validated_required_tensor_count: validation.matched_required_tensors.len(),
    }
}
