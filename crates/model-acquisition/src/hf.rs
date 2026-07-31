//! HuggingFace provider and client helpers.

pub mod catalog;
pub mod client;
pub mod component_resolution;
pub mod format;
pub mod metadata;
pub mod model_index;
pub mod provider;
pub mod strategy;

pub use catalog::{
    IMAGE_GENERATION_FILTERS, ModelCard, ModelCardData, ModelCatalog, ModelCatalogEntry,
    ModelSearchQuery, SortBy, parse_card_data,
};
pub use client::build_hf_client;
pub use component_resolution::{
    ComponentRole, ResolvedComponent, resolve_component_paths, resolve_component_paths_verified,
    resolve_from_model_index,
};
pub use format::{ModelRepoFormat, detect_format, diffusers_download_patterns};
pub use metadata::{HfLfsInfo, HfRepoMetadata, HfSibling};
pub use model_index::{ComponentEntry, ComponentMapping, ModelIndex};
pub use provider::{AcquisitionProgressSink, ProgressSinkBridge};
pub use strategy::{ResolvedPatterns, resolve_download_patterns};
