#![deny(unsafe_code)]

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
pub use hf::client::build_hf_client;
pub use hf::provider::{AcquisitionProgressSink, ProgressSinkBridge};
pub use report::{AcquisitionFileEntry, AcquisitionOutcome, AcquisitionReport};
pub use request::{
    AcquireProvider, AllowPatterns, ModelAcquisitionRequest, OverwritePolicy, RepoId, Revision,
    TargetRelativeDir,
};
pub use staging::{promote_staged, staging_dir};
