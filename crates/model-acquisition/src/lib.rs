#![deny(unsafe_code)]

pub mod burn;
pub mod config;
pub mod error;
pub mod hf;
pub mod paths;
pub mod report;
pub mod request;
pub mod staging;
pub mod timestamp;

pub use config::ModelAcquisitionConfig;
pub use error::{ModelAcquisitionError, ModelAcquisitionResult};
pub use hf::catalog::{
    IMAGE_GENERATION_FILTERS, ModelCard, ModelCardData, ModelCatalog, ModelCatalogEntry,
    ModelSearchQuery, SortBy, parse_card_data,
};
pub use hf::client::build_hf_client;
pub use hf::component_resolution::{
    ComponentRole, ResolvedComponent, resolve_component_paths, resolve_component_paths_verified,
    resolve_from_model_index,
};
pub use hf::format::{ModelRepoFormat, detect_format, diffusers_download_patterns};
pub use hf::metadata::{HfLfsInfo, HfRepoMetadata, HfSibling};
pub use hf::model_index::{ComponentEntry, ComponentMapping, ModelIndex};
pub use hf::provider::{AcquisitionProgressSink, ProgressSinkBridge};
pub use hf::strategy::{ResolvedPatterns, resolve_download_patterns};
pub use report::{AcquisitionFileEntry, AcquisitionOutcome, AcquisitionReport};
pub use request::{
    AcquireProvider, AllowPatterns, ModelAcquisitionRequest, OverwritePolicy, RepoId, Revision,
    TargetRelativeDir,
};
pub use staging::{promote_staged, staging_dir};

pub use burn::{
    BurnCheckpointConverter, BurnConversionComponent, BurnConversionComponentRole,
    BurnConversionReport,
};
