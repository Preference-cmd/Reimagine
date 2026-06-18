//! Reimagine core: workflow engine + backend-agnostic inference types.

#![deny(unsafe_code)]

pub mod command;
pub mod diagnostic;
pub mod event;
pub mod execution_plan;
pub mod execution_value;
pub mod history;
pub mod model;
pub mod readiness;
pub mod session;
pub mod validation;
pub mod workflow;

// Stable top-level re-exports for the public execution value model.
// `runtime`, `inference`, `inference-core`, and concrete backend
// crates should prefer these paths over `core::execution_value::...`.
pub use execution_value::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, BackendTensorMetadata,
    ConditioningMetadata, ExecutionConditioning, ExecutionValue, ExecutionValueKind,
    RuntimeClipHandle, RuntimeImage, RuntimeLatent, RuntimeModelHandle, RuntimeVaeHandle,
};
