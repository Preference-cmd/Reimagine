pub use component::{
    BurnSdxlComponentRole, BurnTensorDType, BurnTensorInventoryEntry, BurnTensorShapeSpec,
    BurnTensorSpec,
};
pub use contract::{
    BURN_SDXL_COMPONENT_CONTRACT_VERSION, BurnDTypePolicy, BurnSdxlComponentContract,
};
pub use metadata::{BurnComponentMetadata, metadata_keys};
pub use validation::{
    BurnSdxlComponentValidationReport, BurnSdxlContractError, BurnSdxlValidationWarning,
    validate_component_inventory,
};

mod component;
mod contract;
// Offline conversion scaffolding is exercised by module tests and consumed by later Burn slices.
#[allow(dead_code)]
mod conversion;
mod metadata;
#[allow(dead_code)]
mod package;
#[allow(dead_code)]
mod source_layout;
#[allow(dead_code)]
mod source_mapping;
mod validation;
#[allow(dead_code)]
mod writer;
