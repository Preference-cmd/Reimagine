//! Host-neutral application service shell.
//!
//! This crate owns the V1 workspace host boundary that future Tauri and Axum
//! adapters call into. It deliberately keeps concrete UI/server types out of
//! the public API and delegates domain semantics to the lower-level crates.

#![deny(unsafe_code)]

mod app_host;
pub mod artifact_access;
mod board_service;
mod document_events;
mod document_store;
pub mod dto;
mod error;
mod error_code;
mod node_catalog;
mod project_service;
mod proposal;
mod run_observation;
mod run_workflow;
mod services;
mod workflow_service;
mod workspace;
mod workspace_scope;

#[cfg(feature = "aigc")]
pub use aigc::BackendSelection;
#[cfg(feature = "aigc")]
pub use aigc::ModelAcquisitionService;
#[cfg(feature = "aigc")]
pub use aigc::SnapshotExternalReadinessProvider;
#[cfg(feature = "aigc")]
pub use aigc::inference::grpc_worker::{
    GrpcSwitchableWorker, GrpcWorkerCandidate, GrpcWorkerCandidateConfig,
};
#[cfg(feature = "aigc")]
pub use aigc::inference::quic_worker::{
    QuicSwitchableWorker, QuicWorkerCandidate, QuicWorkerCandidateConfig,
};
#[cfg(feature = "aigc")]
pub use aigc::inference::switch::{
    ProcessSwitchableWorker, RunCancellation, SwitchableWorker, WorkerSelectionHandle,
    WorkerSwitchError, WorkerSwitchService, WorkerSwitchTarget,
};
#[cfg(feature = "aigc")]
pub use aigc::inference::worker::{
    EmptyWorkerInventoryProvider, InstalledWorkerInventoryProvider, StaticWorkerInventoryProvider,
    WorkerActivationError, WorkerBackendCandidate, WorkerInventoryProvider,
    WorkerInventorySnapshot,
};
#[cfg(feature = "aigc")]
pub use aigc::{
    AcquireAndConvertReport, AcquireAndConvertRequest, AigcError, AigcResult,
    BurnCheckpointConverter, BurnConversionComponent, BurnConversionComponentRole,
    BurnConversionReport, GrpcAuth, GrpcTls, ModelService, WorkerCatalogItemDto,
    WorkerInstallationDto, WorkerManagementError, WorkerManagementService,
};
pub use app_host::AppHost;
pub use artifact_access::{
    ArtifactAccess, ArtifactAccessError, media_type_for_reference, resolve_artifact_path,
};
pub use board_service::BoardService;
pub use document_events::{DocumentChangedEvent, DocumentEventBus};
pub use error::{AppHostError, AppHostResult};
pub use error_code::AppHostErrorCode;
#[cfg(feature = "aigc")]
pub use error_code::{worker_switch_error_code, worker_switch_error_details};
#[cfg(feature = "aigc")]
pub use node_catalog::NodeCatalogAlignment;
pub use node_catalog::NodeCatalogService;
pub use project_service::ProjectService;
pub use proposal::{ProposalReceipt, ProposalStatus, WorkflowProposal};
pub use reimagine_inference::{BackendInstance, WorkspaceComputeProfile};
pub use run_workflow::{RunWorkflowRequest, RunWorkflowResult, run_id_of};
pub use services::WorkspaceServices;
pub use workflow_service::WorkflowService;
pub use workspace::{WorkspaceHost, WorkspaceHostBuilder};
pub use workspace_scope::WorkspaceScope;
