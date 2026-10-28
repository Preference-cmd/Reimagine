//! Host-neutral application service shell.
//!
//! This crate owns the V1 workspace host boundary that host adapters (such as
//! the Tauri desktop app) call into. It deliberately keeps concrete UI types out of
//! the public API and delegates domain semantics to the lower-level crates.

#![deny(unsafe_code)]

mod app_host;
pub mod artifact_access;
pub mod asset_store;
mod board_service;
pub mod collection_ops;
pub mod collection_service;
mod document_store;
pub mod dto;
pub mod dynamic_workflow_ops;
pub mod dynamic_workflow_service;
mod error;
mod error_code;
mod node_catalog;
pub mod operation_service;
mod project_service;
mod proposal;
mod run_observation;
mod run_workflow;
mod services;
pub mod step_log;
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
pub use aigc::inference::remote_worker::{
    RemoteSwitchableWorker, RemoteWorkerCandidate, RemoteWorkerCandidateConfig,
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
pub use asset_store::{AssetStore, resolve_media_type};
pub use board_service::BoardService;
pub use collection_ops::{
    CollectionCreateOp, CollectionGetOp, CollectionRedoOp, CollectionSetSlotOp, CollectionUndoOp,
    register_collection_ops,
};
pub use collection_service::CollectionService;
pub use dynamic_workflow_ops::{
    DynamicWorkflowCancelOp, DynamicWorkflowCreateOp, DynamicWorkflowGetOp,
    DynamicWorkflowGetRunOp, DynamicWorkflowListRunsOp, DynamicWorkflowResumeOp,
    DynamicWorkflowRunOp, register_dynamic_workflow_ops,
};
pub use dynamic_workflow_service::{
    DYNAMIC_WORKFLOW_SLOT_CONCURRENCY, DynamicWorkflowResult, DynamicWorkflowService,
};
pub use error::{AppHostError, AppHostResult};
pub use error_code::AppHostErrorCode;
#[cfg(feature = "aigc")]
pub use error_code::{worker_switch_error_code, worker_switch_error_details};
#[cfg(feature = "aigc")]
pub use node_catalog::NodeCatalogAlignment;
pub use node_catalog::NodeCatalogService;
pub use operation_service::{
    AssetImportOp, InvokeRequest, InvokeResult, KernelCall, KernelOp, OperationInfo,
    OperationProvider, OperationService, is_valid_preview_key, validate_preview_key,
};
pub use project_service::ProjectService;
pub use proposal::{ProposalReceipt, ProposalStatus, WorkflowProposal};
pub use reimagine_inference::{BackendInstance, WorkspaceComputeProfile};
pub use run_workflow::{RunWorkflowRequest, RunWorkflowResult, run_id_of};
pub use services::WorkspaceServices;
pub use step_log::StepLog;
pub use workflow_service::WorkflowService;
pub use workspace::{WorkspaceHost, WorkspaceHostBuilder};
pub use workspace_scope::WorkspaceScope;
