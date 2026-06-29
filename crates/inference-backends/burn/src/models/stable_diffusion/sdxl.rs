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
mod metadata;
mod validation;
