pub use component::{
    BurnSdxlComponentRole, BurnTensorDType, BurnTensorInventoryEntry, BurnTensorShapeSpec,
    BurnTensorSpec,
};
pub use contract::{
    BURN_SDXL_COMPONENT_CONTRACT_VERSION, BurnDTypePolicy, BurnSdxlComponentContract,
};
pub use conversion::{BurnSdxlConversionError, BurnSdxlConversionReport};
pub use loaded::{BurnLoadedModelBundle, BurnLoadedSdxlBundle, BurnSdxlSourceSignature};
pub use metadata::{BurnComponentMetadata, metadata_keys};
pub use text::{BurnSdxlTextEncoderResources, load_sdxl_tokenizer};
pub use text_conditioning::sdxl_text_encoder_spec_set;
pub use tokenizer::{
    BurnSdxlTokenizedPrompt, BurnSdxlTokenizedPromptPair, BurnSdxlTokenizer,
    BurnSdxlTokenizerContext, BurnSdxlTokenizerResources, BurnTokenizerError, BurnTokenizerRole,
    MAX_SEQUENCE_LENGTH, PRIMARY_TOKENIZER_ASSET, SECONDARY_TOKENIZER_ASSET, TOKEN_BOS, TOKEN_EOS,
    TOKEN_PAD,
};
pub use validation::{
    BurnSdxlComponentValidationReport, BurnSdxlContractError, BurnSdxlValidationWarning,
    validate_component_inventory, validate_component_inventory_full,
};

// New checkpoint import pipeline (model-pipeline/01)
pub mod checkpoint_import;
mod checkpoint_inventory;
mod checkpoint_projection;
mod checkpoint_writer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlDiffusersSplitPackageRequest {
    pub source_root: std::path::PathBuf,
    pub source_model_id: String,
    pub source_fingerprint: Option<String>,
    pub converted_models_root: std::path::PathBuf,
    pub overwrite: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnSdxlDiffusersSplitPackageResult {
    pub package_root: std::path::PathBuf,
    pub report_path: std::path::PathBuf,
    pub reused_existing: bool,
}

pub fn package_diffusers_style_split_sdxl_source(
    request: &BurnSdxlDiffusersSplitPackageRequest,
) -> Result<BurnSdxlDiffusersSplitPackageResult, BurnSdxlConversionError> {
    let result = package::package_diffusers_style_split_source(&package::BurnSdxlPackageRequest {
        source_set: source_layout::BurnSdxlSourceSet::diffusers_style_split_safetensors(
            request.source_root.clone(),
        ),
        source_model_id: request.source_model_id.clone(),
        source_fingerprint: request.source_fingerprint.clone(),
        converted_models_root: request.converted_models_root.clone(),
        overwrite: request.overwrite,
    })?;

    Ok(BurnSdxlDiffusersSplitPackageResult {
        package_root: result.package_root,
        report_path: result.report_path,
        reused_existing: result.reused_existing,
    })
}

mod component;
mod contract;
mod load_diagnostics;
mod loaded;
// Offline conversion scaffolding is exercised by module tests and consumed by later Burn slices.
#[allow(dead_code)]
mod conversion;
pub mod diffusion;
mod metadata;
#[allow(dead_code)]
mod package;
#[allow(dead_code)]
mod source_layout;
#[allow(dead_code)]
mod source_mapping;
mod text;
pub mod text_conditioning;
mod tokenizer;
pub mod vae;
mod validation;
#[allow(dead_code)]
mod writer;
