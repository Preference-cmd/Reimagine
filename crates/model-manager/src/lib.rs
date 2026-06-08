//! Local model manifest and model identity infrastructure.

#![deny(unsafe_code)]

#[path = "classify.rs"]
mod classify;
#[path = "error.rs"]
mod error;
#[path = "identity.rs"]
mod identity;
#[path = "manifest.rs"]
mod manifest;
#[path = "resolve.rs"]
mod resolve;
#[path = "scan.rs"]
mod scan;
#[path = "verify.rs"]
mod verify;

pub use classify::{ModelSeriesConfig, ModelSeriesRule};
pub use error::{ModelManagerError, ModelManagerResult};
pub use manifest::{
    Fingerprint, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelRootId,
    ModelRootKind, ModelSource, ModelSourceStatus,
};
pub use scan::ScanConfig;
