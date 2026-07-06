use std::collections::BTreeMap;

use super::component::BurnSdxlComponentRole;
use super::contract::{BURN_SDXL_COMPONENT_CONTRACT_VERSION, BurnDTypePolicy};
use super::validation::BurnSdxlContractError;

pub mod metadata_keys {
    pub const CONTRACT: &str = "reimagine.contract";
    pub const CONTRACT_VERSION: &str = "reimagine.contract_version";
    pub const BACKEND: &str = "reimagine.backend";
    pub const MODEL_SERIES: &str = "reimagine.model_series";
    pub const VARIANT: &str = "reimagine.variant";
    pub const COMPONENT_ROLE: &str = "reimagine.component_role";
    pub const TENSOR_LAYOUT: &str = "reimagine.tensor_layout";
    pub const DTYPE_POLICY: &str = "reimagine.dtype_policy";
    pub const FIXTURE_PROFILE: &str = "reimagine.fixture_profile";
    pub const TINY_SDXL_E2E_PROFILE: &str = "tiny_sdxl_e2e";
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnComponentMetadata {
    pub contract: String,
    pub contract_version: u32,
    pub backend: String,
    pub model_series: String,
    pub variant: String,
    pub component_role: BurnSdxlComponentRole,
    pub tensor_layout: String,
    pub dtype_policy: BurnDTypePolicy,
    pub fixture_profile: Option<String>,
}

impl BurnComponentMetadata {
    pub fn parse(raw: &BTreeMap<String, String>) -> Result<Self, BurnSdxlContractError> {
        let contract = required(raw, metadata_keys::CONTRACT)?;
        let contract_version =
            parse_contract_version(required(raw, metadata_keys::CONTRACT_VERSION)?)?;
        let backend = required(raw, metadata_keys::BACKEND)?;
        let model_series = required(raw, metadata_keys::MODEL_SERIES)?;
        let variant = required(raw, metadata_keys::VARIANT)?;
        let component_role = parse_component_role(required(raw, metadata_keys::COMPONENT_ROLE)?)?;
        let tensor_layout = required(raw, metadata_keys::TENSOR_LAYOUT)?;
        let dtype_policy = parse_dtype_policy(required(raw, metadata_keys::DTYPE_POLICY)?)?;
        let fixture_profile = raw.get(metadata_keys::FIXTURE_PROFILE).cloned();

        Ok(Self {
            contract: contract.to_owned(),
            contract_version,
            backend: backend.to_owned(),
            model_series: model_series.to_owned(),
            variant: variant.to_owned(),
            component_role,
            tensor_layout: tensor_layout.to_owned(),
            dtype_policy,
            fixture_profile,
        })
    }

    pub fn is_tiny_sdxl_e2e_fixture(&self) -> bool {
        self.fixture_profile.as_deref() == Some(metadata_keys::TINY_SDXL_E2E_PROFILE)
    }
}

fn required<'a>(
    raw: &'a BTreeMap<String, String>,
    key: &'static str,
) -> Result<&'a str, BurnSdxlContractError> {
    raw.get(key)
        .map(String::as_str)
        .ok_or_else(|| BurnSdxlContractError::MissingMetadata {
            key: key.to_owned(),
        })
}

fn parse_contract_version(value: &str) -> Result<u32, BurnSdxlContractError> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| BurnSdxlContractError::InvalidMetadata {
            key: metadata_keys::CONTRACT_VERSION.to_owned(),
            expected: BURN_SDXL_COMPONENT_CONTRACT_VERSION.to_string(),
            found: value.to_owned(),
        })?;

    if parsed != BURN_SDXL_COMPONENT_CONTRACT_VERSION {
        return Err(BurnSdxlContractError::UnsupportedContractVersion {
            found: value.to_owned(),
        });
    }

    Ok(parsed)
}

fn parse_component_role(value: &str) -> Result<BurnSdxlComponentRole, BurnSdxlContractError> {
    BurnSdxlComponentRole::try_from(value).map_err(|_| BurnSdxlContractError::InvalidMetadata {
        key: metadata_keys::COMPONENT_ROLE.to_owned(),
        expected: "diffusion|vae|text_encoder|text_encoder_2".to_owned(),
        found: value.to_owned(),
    })
}

fn parse_dtype_policy(value: &str) -> Result<BurnDTypePolicy, BurnSdxlContractError> {
    BurnDTypePolicy::try_from(value).map_err(|_| BurnSdxlContractError::InvalidMetadata {
        key: metadata_keys::DTYPE_POLICY.to_owned(),
        expected: "fp32|fp16|bf16|mixed".to_owned(),
        found: value.to_owned(),
    })
}
