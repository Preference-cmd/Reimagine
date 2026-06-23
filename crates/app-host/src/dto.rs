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

mod artifacts;
mod health;
mod nodes;
mod runs;
mod workflows;

pub use artifacts::ArtifactDto;
pub use health::HealthResponse;
pub use nodes::{NodeCatalogResponse, NodeDefDto, ParamSpecDto, SocketSpecDto};
pub use runs::{DiagnosticDto, NodeStateDto, RunDto, RunEventDto, RunEventsResponse, RunSnapshotDto, RunSummaryDto};
pub use workflows::{
    OpenWorkflowRequest, OpenWorkflowResponse, RunTargetDto, RunWorkflowRequestDto,
    RunWorkflowResponse, TargetSelectionDto, WorkflowSource,
};

/// Marker that the type is host-safe: it never carries
/// [`ExecutionValue`]-shaped payloads.
#[allow(dead_code)]
const fn _assert_no_runtime_values() {
    // The DTOs above are JSON-only; if a future change accidentally
    // re-introduces a runtime value handle, the API surface breaks.
    // The constant below documents the invariant and gives the next
    // reviewer a hint where to look.
    let _ = std::mem::size_of::<reimagine_runtime::value::ExecutionValue>();
}
