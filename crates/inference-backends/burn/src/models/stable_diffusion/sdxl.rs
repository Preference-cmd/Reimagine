pub use component::{
    BurnSdxlComponentRole, BurnTensorDType, BurnTensorInventoryEntry, BurnTensorShapeSpec,
    BurnTensorSpec,
};
pub use contract::{
    BURN_SDXL_COMPONENT_CONTRACT_VERSION, BurnDTypePolicy, BurnSdxlComponentContract,
};
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
