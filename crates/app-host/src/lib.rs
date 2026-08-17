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
mod board_service;
pub mod dto;
mod error;
mod error_code;
mod inference;
mod inference_backend;
mod model_acquisition_service;
mod model_conversion;
mod model_service;
mod node_catalog;
mod policy;
mod project_service;
mod proposal;
mod protocol;
mod provider_config;
mod readiness;
mod run_observation;
mod run_workflow;
mod services;
mod tools;
mod worker_management;
mod workflow_service;
mod workspace;

pub use agent_provider::{AgentProviderCatalog, build_provider, register_providers_from_document};
pub use agent_service::{AgentService, AgentServiceTurnRequest};
pub use app_host::AppHost;
pub use artifact_access::{
    ArtifactAccess, ArtifactAccessError, media_type_for_reference, resolve_artifact_path,
};
pub use board_service::{BoardChangedEvent, BoardService};
pub use error::{AppHostError, AppHostResult};
pub use error_code::{AppHostErrorCode, worker_switch_error_code, worker_switch_error_details};
pub use inference::grpc_worker::{
    GrpcSwitchableWorker, GrpcWorkerCandidate, GrpcWorkerCandidateConfig,
};
pub use inference::quic_worker::{
    QuicSwitchableWorker, QuicWorkerCandidate, QuicWorkerCandidateConfig,
};
pub use inference::switch::{
    ProcessSwitchableWorker, RunCancellation, SwitchableWorker, WorkerSelectionHandle,
    WorkerSwitchError, WorkerSwitchService, WorkerSwitchTarget,
};
pub use inference::worker::{
    EmptyWorkerInventoryProvider, InstalledWorkerInventoryProvider, StaticWorkerInventoryProvider,
    WorkerActivationError, WorkerBackendCandidate, WorkerInventoryProvider,
    WorkerInventorySnapshot,
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
pub use project_service::ProjectService;
pub use proposal::{ProposalReceipt, ProposalStatus, WorkflowProposal};
pub use protocol::{TurnRunParams, TurnRunResult, TurnRunStatus};
pub use provider_config::{
    AgentProviderConfigDocument, AnthropicMessagesConfig, OpenAiChatCompletionsConfig,
    OpenAiResponsesConfig, Protocol, ProviderConfig,
};
pub use readiness::SnapshotExternalReadinessProvider;
pub use reimagine_backend_worker_transport_grpc::{GrpcAuth, GrpcTls};
pub use reimagine_inference::{BackendInstance, WorkspaceComputeProfile};
pub use run_workflow::{RunWorkflowRequest, RunWorkflowResult, run_id_of};
pub use services::WorkspaceServices;
pub use tools::register_app_tools;
pub use worker_management::{
    WorkerCatalogItemDto, WorkerInstallationDto, WorkerManagementError, WorkerManagementService,
};
pub use workflow_service::WorkflowService;
pub use workspace::WorkspaceHost;
pub use workspace::WorkspaceHostBuilder;
