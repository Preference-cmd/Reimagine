//! Host-independent workflow runtime.
//!
//! See `docs/architecture/modules/runtime.md` for the architecture source of
//! truth. This crate owns the [`RuntimeService`] and the run/snapshot/summary
//! state machine, but does not depend on Tauri, Axum, model-manager, or
//! Candle integration.

#![deny(unsafe_code)]

mod artifacts;
mod cancellation;
mod clock;
mod error;
mod events;
mod executor;
mod handle;
mod node_context;
mod resources;
mod run_inputs;
mod run_session;
mod runner;
mod scheduler;
mod snapshot;
mod store;
mod value_store;

pub use artifacts::{ArtifactRecord, ArtifactStore, NodeArtifactCapability};
pub use cancellation::CancellationToken;
pub use clock::{Clock, SystemClock};
pub use error::RuntimeError;
pub use events::{BoxedRunEventSink, RunEventSink, VecRunEventSink};
pub use executor::{
    BoxedNodeExecutor, NodeExecutor, NodeExecutorError, NodeExecutorRegistry,
    NodeExecutorRegistryError,
};
pub use handle::{RunHandle, RunState};
pub use node_context::NodeExecutionContext;
pub use resources::{MemorySnapshot, NoopRunResourceBackend, RunResourceBackend};
pub use run_inputs::RunInputs;
pub use run_session::RunSession;
pub use runner::{RuntimeOptions, RuntimeService, RuntimeServiceError};
pub use scheduler::NodeState;
pub use snapshot::{RunArtifactRef, RunSnapshot, RunSummary};
pub use store::{RunStore, RunStoreInner};
pub use value_store::{OutputKey, RunValueStore};

pub use value::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, ConditioningMetadata, RuntimeClipHandle,
    RuntimeConditioning, RuntimeImage, RuntimeLatent, RuntimeModelHandle, RuntimeValue,
    RuntimeVaeHandle,
};

pub mod value;
