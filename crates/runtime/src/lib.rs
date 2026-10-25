//! Host-independent workflow runtime.
//!
//! This crate executes prepared workflow execution plans through
//! [`RuntimeService`], managing run state, stage scheduling, cancellation, and
//! artifact collection. It does not depend on a host adapter (such as Tauri),
//! `reimagine-model-manager` or a concrete inference backend.
//!
//! The executor contract (`NodeExecutor`, `NodeExecutionContext`,
//! `NodeInputs`/`NodeParams`, `NodeExecutorRegistry`, `ArtifactPublisher`,
//! `ArtifactEventKind`, `NodeExecutorError`) is
//! owned by [`reimagine_inference`] and re-exported here for backward
//! compatibility. New runtime-side code should prefer importing the
//! types from `reimagine_inference` directly.

#![deny(unsafe_code)]
// BE-39: no panicking `unwrap()` in non-test code. Test modules are
// exempt because the lint fires on `#[cfg(test)]` code too.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]

mod artifacts;
mod clock;
mod consumer_index;
mod error;
mod events;
mod handle;
mod progress;
mod resources;
mod run_inputs;
mod run_session;
mod runner;
mod scheduler;
mod snapshot;
mod stage_runner;
mod store;
mod value_store;

pub use artifacts::{ArtifactRecord, ArtifactStore, RuntimeNodeArtifactCapability};
pub use clock::{Clock, SystemClock};
pub use consumer_index::{ConsumerBinding, PlanConsumerIndex};
pub use error::RuntimeError;
pub use events::{BoxedRunEventSink, RunEventSink, VecRunEventSink};
pub use handle::RunHandle;
pub use reimagine_core::task::{CancelToken, TaskState};
pub use resources::NoopBackendInstanceRuntimeHooks;
pub use run_inputs::RunInputs;
pub use run_session::RunSession;
pub use runner::{NodeDeadlines, RuntimeOptions, RuntimeService, RuntimeServiceError};
pub use scheduler::{
    AdmissionDecision, DEFAULT_NODE_ESTIMATED_VRAM_BYTES, MemoryBudget, NodeState,
    ReadySetScheduler,
};
pub use snapshot::{RunArtifactRef, RunSnapshot, RunSnapshotUpdate, RunSummary};
pub use store::{RunStore, RunStoreInner};
pub use value_store::{OutputKey, RunValueStore, RuntimeValueRecord};

pub use value::RuntimeValue;
pub use value::{
    Backend, BackendPayloadKey, BackendTensorHandle, BackendTensorMetadata, ConditioningMetadata,
    ExecutionConditioning, ExecutionOutput, ExecutionValue, ExecutionValueKind,
    ExecutionValueRetention, RuntimeClipHandle, RuntimeConditioning, RuntimeImage, RuntimeLatent,
    RuntimeModelHandle, RuntimeVaeHandle, StageId,
};

// Executor contract re-exports from `reimagine_inference`. Existing
// call sites that previously imported these from `reimagine_runtime`
// continue to compile unchanged.
pub use reimagine_inference::{
    ArtifactEventKind, ArtifactPublisher, BoxedNodeExecutor, NodeExecutionContext,
    NodeExecutionOutputs, NodeExecutor, NodeExecutorError, NodeExecutorRegistry,
    NodeExecutorRegistryError, NodeInputs, NodeParams,
};

pub mod value;
