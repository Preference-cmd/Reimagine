//! Workspace-scoped configuration infrastructure.

#![deny(unsafe_code)]

mod app_config;
mod atomic_write;
mod document;
mod error;
mod handle;
mod inference_backend;
mod key;
mod paths;
mod report;
mod store;

pub use app_config::AppConfig;
pub use atomic_write::atomic_write;
pub use document::{ConfigDocument, ConfigValidationContext};
pub use error::{ConfigError, ConfigResult};
pub use handle::ConfigHandle;
pub use inference_backend::{InferenceBackendConfig, InferenceBackendKind};
pub use key::ConfigKey;
pub use paths::AppPaths;
pub use report::ConfigReport;
pub use store::ConfigStore;
