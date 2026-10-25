//! Cross-host shared data transfer objects.
//!
//! These types are the V1 wire contract exposed to host adapters
//! (such as Tauri). The shapes are intentionally minimal — they wrap
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

mod artifacts;
mod boards;
mod health;
mod nodes;
mod projects;
mod runs;
mod workflows;

#[cfg(feature = "aigc")]
pub use aigc::dto::compute_profile;
#[cfg(feature = "aigc")]
pub use aigc::dto::model_acquisition;
#[cfg(feature = "aigc")]
pub use aigc::dto::models;

#[cfg(feature = "aigc")]
pub use aigc::dto::{
    BackendInstanceProfileDto, BackendProfileDto, ComputeProfileDto, DTypeProfileDto,
    DeviceProfileDto, DownloadEventPayload, FileEntryDto, MemoryProfileDto,
    ModelAcquireConversionReport, ModelAcquireDownloadReport, ModelAcquireInput,
    ModelAcquireOutput, ModelCardDto, ModelCatalogEntryDto, ModelComponentDto,
    ModelConvertConversionReport, ModelConvertInput, ModelConvertOutput, ModelDetailDto,
    ModelDownloadInput, ModelDownloadOutput, ModelFilters, ModelInfoDto, ModelListEntry,
    ModelListOutput, ModelRemoveOutput, TopologyWorkerDto, format_status,
    topology_workers_from_pool,
};
pub use artifacts::{ArtifactDto, ArtifactMetadataDto};
pub use boards::{BoardCommandResultDto, BoardSnapshotDto};
pub use health::{HealthDto, HealthResponse};
pub use nodes::{NodeCatalogResponse, NodeDefDto, ParamSpecDto, SocketSpecDto};
pub use projects::{ProjectDto, ProjectMetadataInputDto};
pub use reimagine_core::command::{CommandBatch, CommandResult, CommandStatus};
pub use runs::{
    DiagnosticDto, NodeStateDto, RunDto, RunEventDto, RunEventsResponse, RunSnapshotDto,
    RunSummaryDto,
};
pub use workflows::{
    OpenWorkflowRequest, OpenWorkflowResponse, RunTargetDto, RunWorkflowRequestDto,
    RunWorkflowResponse, TargetSelectionDto, WorkflowSource,
};
