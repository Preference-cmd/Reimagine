//! Local model manifest and model identity infrastructure.

#![deny(unsafe_code)]

mod classify;
mod error;
mod identity;
mod manifest;
mod resolve;
mod scan;
mod store;
mod verify;

pub use classify::{ModelSeriesConfig, ModelSeriesRule};
pub use error::{ModelManagerError, ModelManagerResult};
pub use manifest::{
    validate_manifest, Fingerprint, ManifestValidationReport, ModelDescriptor, ModelFormat,
    ModelManifest, ModelRoot, ModelRootId, ModelRootKind, ModelSource, ModelSourceStatus,
};
pub use scan::ScanConfig;
pub use store::{load_model_manifest, ModelManifestStore};
