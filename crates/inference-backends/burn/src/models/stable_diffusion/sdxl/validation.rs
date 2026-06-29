use std::collections::{BTreeMap, BTreeSet};

use super::component::{BurnSdxlComponentRole, BurnTensorInventoryEntry};
use super::contract::{BACKEND_NAME, CONTRACT_NAME, MODEL_SERIES, TENSOR_LAYOUT, VARIANT};
use super::metadata::{BurnComponentMetadata, metadata_keys};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnSdxlContractError {
    MissingMetadata {
        key: String,
    },
    InvalidMetadata {
        key: String,
        expected: String,
        found: String,
    },
    UnsupportedContractVersion {
        found: String,
    },
    MissingRequiredTensors {
        keys: Vec<String>,
    },
    TensorShapeMismatch {
        key: String,
        expected: String,
        found: Vec<usize>,
    },
    UnsupportedTensorDType {
        key: String,
        dtype: String,
    },
}

impl std::fmt::Display for BurnSdxlContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMetadata { key } => {
                write!(f, "missing Burn SDXL component metadata `{key}`")
            }
            Self::InvalidMetadata {
                key,
                expected,
                found,
            } => write!(
                f,
                "invalid Burn SDXL component metadata `{key}`: expected {expected}, found {found}"
            ),
            Self::UnsupportedContractVersion { found } => {
                write!(
                    f,
                    "unsupported Burn SDXL component contract version `{found}`"
                )
            }
            Self::MissingRequiredTensors { keys } => {
                write!(f, "missing required Burn SDXL tensors: {}", keys.join(", "))
            }
            Self::TensorShapeMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "Burn SDXL tensor `{key}` shape mismatch: expected {expected}, found {found:?}"
            ),
            Self::UnsupportedTensorDType { key, dtype } => {
                write!(
                    f,
                    "Burn SDXL tensor `{key}` has unsupported dtype `{dtype}`"
                )
            }
        }
    }
}

impl std::error::Error for BurnSdxlContractError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnSdxlValidationWarning {
    UnusedTensor { key: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlComponentValidationReport {
    pub component_role: BurnSdxlComponentRole,
    pub matched_required_tensors: Vec<String>,
    pub missing_required_tensors: Vec<String>,
    pub unused_tensors: Vec<String>,
    pub warnings: Vec<BurnSdxlValidationWarning>,
}

pub fn validate_component_inventory(
    metadata: &BTreeMap<String, String>,
    inventory: &[BurnTensorInventoryEntry],
) -> Result<BurnSdxlComponentValidationReport, BurnSdxlContractError> {
    let metadata = BurnComponentMetadata::parse(metadata)?;
    validate_metadata_values(&metadata)?;

    let contract = metadata.component_role.contract();
    let inventory_by_key = inventory
        .iter()
        .map(|entry| (entry.key.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let expected_keys = contract
        .expected_tensor_specs()
        .iter()
        .map(|spec| spec.key)
        .collect::<BTreeSet<_>>();

    let mut matched_required_tensors = Vec::new();
    let mut missing_required_tensors = Vec::new();

    for spec in contract.expected_tensor_specs() {
        let Some(entry) = inventory_by_key.get(spec.key) else {
            if spec.required {
                missing_required_tensors.push(spec.key.to_owned());
            }
            continue;
        };

        if !entry.dtype.is_supported() {
            return Err(BurnSdxlContractError::UnsupportedTensorDType {
                key: entry.key.clone(),
                dtype: entry.dtype.as_str().to_owned(),
            });
        }

        if !spec.shape.matches(&entry.shape) {
            return Err(BurnSdxlContractError::TensorShapeMismatch {
                key: entry.key.clone(),
                expected: format!("rank {}", spec.shape.rank()),
                found: entry.shape.clone(),
            });
        }

        if spec.required {
            matched_required_tensors.push(spec.key.to_owned());
        }
    }

    if !missing_required_tensors.is_empty() {
        return Err(BurnSdxlContractError::MissingRequiredTensors {
            keys: missing_required_tensors,
        });
    }

    let unused_tensors = inventory
        .iter()
        .filter(|entry| !expected_keys.contains(entry.key.as_str()))
        .map(|entry| entry.key.clone())
        .collect::<Vec<_>>();
    let warnings = unused_tensors
        .iter()
        .cloned()
        .map(|key| BurnSdxlValidationWarning::UnusedTensor { key })
        .collect();

    Ok(BurnSdxlComponentValidationReport {
        component_role: metadata.component_role,
        matched_required_tensors,
        missing_required_tensors: Vec::new(),
        unused_tensors,
        warnings,
    })
}

fn validate_metadata_values(metadata: &BurnComponentMetadata) -> Result<(), BurnSdxlContractError> {
    assert_metadata_value(metadata_keys::CONTRACT, CONTRACT_NAME, &metadata.contract)?;
    assert_metadata_value(metadata_keys::BACKEND, BACKEND_NAME, &metadata.backend)?;
    assert_metadata_value(
        metadata_keys::MODEL_SERIES,
        MODEL_SERIES,
        &metadata.model_series,
    )?;
    assert_metadata_value(metadata_keys::VARIANT, VARIANT, &metadata.variant)?;
    assert_metadata_value(
        metadata_keys::TENSOR_LAYOUT,
        TENSOR_LAYOUT,
        &metadata.tensor_layout,
    )?;

    Ok(())
}

fn assert_metadata_value(
    key: &'static str,
    expected: &'static str,
    found: &str,
) -> Result<(), BurnSdxlContractError> {
    if found != expected {
        return Err(BurnSdxlContractError::InvalidMetadata {
            key: key.to_owned(),
            expected: expected.to_owned(),
            found: found.to_owned(),
        });
    }

    Ok(())
}
