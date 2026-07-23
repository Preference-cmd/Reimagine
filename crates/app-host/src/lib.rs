//! Host-neutral application service shell.
//!
//! This crate owns the V1 workspace host boundary that future Tauri and Axum
//! adapters call into. It deliberately keeps concrete UI/server types out of
//! the public API and delegates domain semantics to the lower-level crates.

#![deny(unsafe_code)]

mod agent_provider;
mod agent_service;
mod app_host;
pub mod artifact_access;
pub mod dto;
mod error;
mod inference;
mod inference_backend;
mod model_acquisition_service;
mod model_conversion;
mod model_service;
mod node_catalog;
mod policy;
mod proposal;
mod readiness;
mod run_observation;
mod run_workflow;
mod services;
mod tools;
mod worker_management;
mod workflow_service;
mod workspace;

pub use agent_provider::AgentProviderCatalog;
pub use agent_service::{AgentService, AgentServiceTurnRequest};
pub use app_host::AppHost;
pub use artifact_access::{
    ArtifactAccess, ArtifactAccessError, media_type_for_reference, resolve_artifact_path,
};
pub use error::{AppHostError, AppHostResult};
pub use inference::switch::{
    ProcessSwitchableWorker, RunCancellation, SwitchableWorker, SwitchingInferenceRuntime,
    WorkerSelectionHandle, WorkerSwitchError, WorkerSwitchService, WorkerSwitchTarget,
};
pub use inference::worker::{
    EmptyWorkerInventoryProvider, StaticWorkerInventoryProvider, WorkerActivationError,
    WorkerBackendCandidate, WorkerInventoryProvider, WorkerInventorySnapshot,
};
pub use inference_backend::BackendSelection;
pub use model_acquisition_service::ModelAcquisitionService;
pub use model_conversion::{
    BurnCheckpointConverter, BurnConversionComponent, BurnConversionComponentRole,
    BurnConversionReport,
};
pub use model_service::{AcquireAndConvertReport, AcquireAndConvertRequest, ModelService};
pub use node_catalog::{NodeCatalogAlignment, NodeCatalogService};
pub use policy::WorkflowCommandPolicy;
pub use proposal::{ProposalReceipt, ProposalStatus, WorkflowProposal};
pub use readiness::SnapshotExternalReadinessProvider;
pub use reimagine_inference::{BackendInstance, WorkspaceComputeProfile};
pub use run_workflow::{RunWorkflowRequest, RunWorkflowResult, run_id_of};
pub use services::WorkspaceServices;
pub use tools::register_app_tools;
pub use worker_management::{
    WorkerCatalogItemDto, WorkerInstallationDto, WorkerManagementError, WorkerManagementService,
};
pub use workflow_service::WorkflowService;
pub use workspace::WorkspaceHost;
