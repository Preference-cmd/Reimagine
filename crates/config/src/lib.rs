//! Workspace-scoped configuration infrastructure.

#![deny(unsafe_code)]

#[path = "app_config.rs"]
mod app_config;
#[path = "atomic_write.rs"]
mod atomic_write;
#[path = "document.rs"]
mod document;
#[path = "error.rs"]
mod error;
#[path = "handle.rs"]
mod handle;
#[path = "key.rs"]
mod key;
#[path = "paths.rs"]
mod paths;
#[path = "report.rs"]
mod report;
#[path = "store.rs"]
mod store;

pub use app_config::AppConfig;
pub use atomic_write::atomic_write;
pub use document::{ConfigDocument, ConfigValidationContext};
pub use error::{ConfigError, ConfigResult};
pub use handle::ConfigHandle;
pub use key::ConfigKey;
pub use paths::AppPaths;
pub use report::ConfigReport;
pub use store::ConfigStore;
