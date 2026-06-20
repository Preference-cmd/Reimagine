//! Host-independent workflow runtime.
//!
//! See `docs/architecture/modules/runtime.md` for the architecture source of
//! truth. This crate owns the [`RuntimeService`] and the run/snapshot/summary
//! state machine, but does not depend on Tauri, Axum, model-manager, or
//! Candle integration.
//!
//! The executor contract (`NodeExecutor`, `NodeExecutionContext`,
//! `NodeInputs`/`NodeParams`, `NodeExecutorRegistry`, `ArtifactPublisher`,
//! `NodeCancellation`, `ArtifactEventKind`, `NodeExecutorError`) is
//! owned by [`reimagine_inference`] and re-exported here for backward
//! compatibility. New runtime-side code should prefer importing the
//! types from `reimagine_inference` directly.

#![deny(unsafe_code)]

mod artifacts;
mod cancellation;
mod clock;
mod error;
mod events;
mod handle;
mod resources;
mod run_inputs;
mod run_session;
mod runner;
mod scheduler;
mod snapshot;
mod store;
mod value_store;

pub use artifacts::{ArtifactRecord, ArtifactStore, RuntimeNodeArtifactCapability};
pub use cancellation::CancellationToken;
pub use clock::{Clock, SystemClock};
pub use error::RuntimeError;
pub use events::{BoxedRunEventSink, RunEventSink, VecRunEventSink};
pub use handle::{RunHandle, RunState};
pub use resources::NoopRunResourceBackend;
pub use run_inputs::RunInputs;
pub use run_session::RunSession;
pub use runner::{RuntimeOptions, RuntimeService, RuntimeServiceError};
pub use scheduler::NodeState;
pub use snapshot::{RunArtifactRef, RunSnapshot, RunSummary};
pub use store::{RunStore, RunStoreInner};
pub use value_store::{OutputKey, RunValueStore};

pub use value::RuntimeValue;
pub use value::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, BackendTensorMetadata,
    ConditioningMetadata, ExecutionConditioning, ExecutionOutput, ExecutionValue,
    ExecutionValueKind, ExecutionValueRetention, RuntimeClipHandle, RuntimeConditioning,
    RuntimeImage, RuntimeLatent, RuntimeModelHandle, RuntimeVaeHandle,
};

// Executor contract re-exports from `reimagine_inference`. Existing
// call sites that previously imported these from `reimagine_runtime`
// continue to compile unchanged.
pub use reimagine_inference::{
    ArtifactEventKind, ArtifactPublisher, BoxedNodeExecutor, NodeCancellation,
    NodeExecutionContext, NodeExecutionOutputs, NodeExecutor, NodeExecutorError,
    NodeExecutorRegistry, NodeExecutorRegistryError, NodeInputs, NodeParams,
};

pub mod value;
