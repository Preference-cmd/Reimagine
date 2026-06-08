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

pub use classify::{
    ClassificationCandidate, ClassificationResult, Classifier, MODEL_SERIES_SCHEMA_VERSION,
    ModelSeriesConfig, ModelSeriesRule,
};
pub use error::{ModelManagerError, ModelManagerResult};
pub use identity::{AutoIdResult, IdPolicy, IdResolution};
pub use manifest::{
    Fingerprint, ManifestValidationReport, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot,
    ModelRootId, ModelRootKind, ModelSource, ModelSourceStatus, validate_manifest,
    validate_manifest_with_series_config,
};
pub use scan::{ManifestUpdate, ManifestUpdatePolicy, ModelScanner, ScanConfig, ScanObservation};
pub use store::{ModelManifestStore, load_model_manifest};
