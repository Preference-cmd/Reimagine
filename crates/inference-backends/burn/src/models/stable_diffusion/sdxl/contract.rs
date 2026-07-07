use super::component::{BurnSdxlComponentRole, BurnTensorSpec};
use crate::text_encoder::specs::OwnedTensorSpec;

pub const BURN_SDXL_COMPONENT_CONTRACT_VERSION: u32 = 1;
pub(crate) const CONTRACT_NAME: &str = "burn.component";
pub(crate) const BACKEND_NAME: &str = "burn";
pub(crate) const MODEL_SERIES: &str = "stable_diffusion";
pub(crate) const VARIANT: &str = "sdxl";
pub(crate) const TENSOR_LAYOUT: &str = "burn-module-snapshot";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnDTypePolicy {
    Fp32,
    Fp16,
    Bf16,
    Mixed,
}

impl BurnDTypePolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Mixed => "mixed",
        }
    }
}

impl TryFrom<&str> for BurnDTypePolicy {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "fp32" => Ok(Self::Fp32),
            "fp16" => Ok(Self::Fp16),
            "bf16" => Ok(Self::Bf16),
            "mixed" => Ok(Self::Mixed),
            other => Err(format!("unsupported Burn dtype policy `{other}`")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlComponentContract {
    pub contract_version: u32,
    pub component_role: BurnSdxlComponentRole,
    pub dtype_policy: BurnDTypePolicy,
}

impl BurnSdxlComponentContract {
    pub const fn new(component_role: BurnSdxlComponentRole) -> Self {
        Self {
            contract_version: BURN_SDXL_COMPONENT_CONTRACT_VERSION,
            component_role,
            dtype_policy: BurnDTypePolicy::Mixed,
        }
    }

    pub const fn contract_name(&self) -> &'static str {
        CONTRACT_NAME
    }

    pub const fn backend(&self) -> &'static str {
        BACKEND_NAME
    }

    pub const fn model_series(&self) -> &'static str {
        MODEL_SERIES
    }

    pub const fn variant(&self) -> &'static str {
        VARIANT
    }

    pub const fn tensor_layout(&self) -> &'static str {
        TENSOR_LAYOUT
    }

    pub fn expected_tensor_specs(&self) -> &'static [BurnTensorSpec] {
        match self.component_role {
            BurnSdxlComponentRole::Diffusion => DIFFUSION_SPECS,
            BurnSdxlComponentRole::Vae => VAE_SPECS,
            BurnSdxlComponentRole::TextEncoder => TEXT_ENCODER_SPECS,
            BurnSdxlComponentRole::TextEncoder2 => TEXT_ENCODER_2_SPECS,
        }
    }

    /// Return the complete set of required tensor specs for this
    /// component, including all generated transformer-block keys
    /// for text-encoder roles. For diffusion and VAE roles the
    /// return value is a copy of the static representative specs;
    /// for text-encoder roles the set covers every tensor the
    /// executable module will try to deserialize.
    pub fn all_expected_tensor_specs(&self) -> Vec<OwnedTensorSpec> {
        match self.component_role {
            BurnSdxlComponentRole::Diffusion | BurnSdxlComponentRole::Vae => self
                .expected_tensor_specs()
                .iter()
                .map(|s| OwnedTensorSpec {
                    key: s.key.to_owned(),
                    shape: s.shape,
                    required: s.required,
                    notes: s.notes.to_owned(),
                })
                .collect(),
            BurnSdxlComponentRole::TextEncoder => {
                crate::text_encoder::specs::TextEncoderSpecSetBuilder::sdxl_clip_l().specs
            }
            BurnSdxlComponentRole::TextEncoder2 => {
                crate::text_encoder::specs::TextEncoderSpecSetBuilder::sdxl_open_clip_g().specs
            }
        }
    }
}

const DIFFUSION_SPECS: &[BurnTensorSpec] = &[
    BurnTensorSpec::required_rank(
        "conv_in.weight",
        4,
        "representative first convolution weight",
    ),
    BurnTensorSpec::required_rank(
        "conv_out.weight",
        4,
        "representative output convolution weight",
    ),
];

const VAE_SPECS: &[BurnTensorSpec] = &[
    BurnTensorSpec::required_rank(
        "conv_out.weight",
        4,
        "representative decoder output convolution weight",
    ),
    BurnTensorSpec::required_rank("conv_out.bias", 1, "representative decoder output bias"),
];

const TEXT_ENCODER_SPECS: &[BurnTensorSpec] = &[
    BurnTensorSpec::required_rank(
        "model.text_encoder.token_embedding.weight",
        2,
        "representative CLIP-L token embedding weight",
    ),
    BurnTensorSpec::required_rank(
        "model.text_encoder.final_layer_norm.gamma",
        1,
        "representative CLIP-L normalization weight in Burn naming",
    ),
];

const TEXT_ENCODER_2_SPECS: &[BurnTensorSpec] = &[
    BurnTensorSpec::required_rank(
        "model.text_encoder_2.token_embedding.weight",
        2,
        "representative CLIP-G token embedding weight",
    ),
    BurnTensorSpec::required_rank(
        "model.text_encoder_2.final_layer_norm.gamma",
        1,
        "representative CLIP-G normalization weight in Burn naming",
    ),
];
