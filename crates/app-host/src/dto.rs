//! Cross-host shared data transfer objects.
//!
//! These types are the V1 wire contract shared between host adapters
//! (Axum, Tauri). The shapes are intentionally minimal — they wrap
//! host-neutral types from `reimagine-core` and `reimagine-runtime`
//! so that the JSON surface stays stable even as the underlying types
//! evolve.
//!
//! Two principles govern this module:
//!
//! 1. DTOs do not contain runtime value stores or backend tensor
//!    handles. The host-neutral [`reimagine_runtime::RunSnapshot`] and
//!    [`reimagine_runtime::RunSummary`] already enforce that boundary;
//!    we just pass them through as JSON.
//! 2. The shapes are flat enough to drive curl-based smoke tests and
//!    rich enough to drive end-to-end automation. We deliberately avoid
//!    re-serializing the underlying structs: any future host-side
//!    transformation belongs here, with a test.

mod agent;
mod artifacts;
mod compute_profile;
mod health;
pub mod model_acquisition;
mod models;
mod nodes;
mod runs;
mod workflows;

pub use agent::{
    AgentEventPayload, AgentMessageDto, AgentSessionInfo, AgentToolCallDto, AgentTurnResponse,
    AgentUsageDto,
};
pub use artifacts::{ArtifactDto, ArtifactMetadataDto};
pub use compute_profile::{
    BackendInstanceProfileDto, BackendProfileDto, ComputeProfileDto, DTypeProfileDto,
    DeviceProfileDto, MemoryProfileDto,
};
pub use health::HealthResponse;
pub use model_acquisition::{FileEntryDto, ModelDownloadInput, ModelDownloadOutput};
pub use models::ModelInfoDto;
pub use nodes::{NodeCatalogResponse, NodeDefDto, ParamSpecDto, SocketSpecDto};
pub use reimagine_core::command::{CommandResult, CommandResultStatus};
pub use runs::{
    DiagnosticDto, NodeStateDto, RunDto, RunEventDto, RunEventsResponse, RunSnapshotDto,
    RunSummaryDto,
};
pub use workflows::{
    OpenWorkflowRequest, OpenWorkflowResponse, RunTargetDto, RunWorkflowRequestDto,
    RunWorkflowResponse, TargetSelectionDto, WorkflowSource,
};
