//! Host-neutral application service shell.
//!
//! This crate owns the V1 workspace host boundary that future Tauri and Axum
//! adapters call into. It deliberately keeps concrete UI/server types out of
//! the public API and delegates domain semantics to the lower-level crates.

#![deny(unsafe_code)]

mod agent_service;
mod app_host;
mod error;
mod model_service;
mod workflow_service;
mod workspace;

pub use agent_service::AgentService;
pub use app_host::AppHost;
pub use error::{AppHostError, AppHostResult};
pub use model_service::ModelService;
pub use workflow_service::WorkflowService;
pub use workspace::WorkspaceHost;
