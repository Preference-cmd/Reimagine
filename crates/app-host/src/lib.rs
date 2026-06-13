//! Host-neutral application service shell.
//!
//! This crate owns the V1 workspace host boundary that future Tauri and Axum
//! adapters call into. It deliberately keeps concrete UI/server types out of
//! the public API and delegates domain semantics to the lower-level crates.

#![deny(unsafe_code)]

mod agent_provider;
mod agent_service;
mod app_host;
mod error;
mod model_service;
mod policy;
mod proposal;
mod readiness;
mod run_observation;
mod run_workflow;
mod services;
mod tools;
mod workflow_service;
mod workspace;

pub use agent_provider::AgentProviderCatalog;
pub use agent_service::{AgentService, AgentServiceTurnRequest};
pub use app_host::AppHost;
pub use error::{AppHostError, AppHostResult};
pub use model_service::ModelService;
pub use policy::WorkflowCommandPolicy;
pub use proposal::{ProposalReceipt, ProposalStatus, WorkflowProposal};
pub use readiness::SnapshotExternalReadinessProvider;
pub use run_workflow::{RunWorkflowRequest, RunWorkflowResult, run_id_of};
pub use services::WorkspaceServices;
pub use tools::register_app_tools;
pub use workflow_service::WorkflowService;
pub use workspace::WorkspaceHost;
