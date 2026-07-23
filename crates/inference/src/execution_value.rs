//! Canonical execution value model for Reimagine.
//!
//! This module owns the canonical [`ExecutionValue`] envelope and the
//! backend-affine handle types that ride on it. It is the source of
//! truth for run-time value semantics and is shared by `runtime`,
//! `inference`, concrete backend crates, and any caller that needs
//! to pass typed values between inference executors and backends.
//!
//! The submodules stay private; downstream code should use the
//! top-level re-exports from `crate::*` (or the
//! runtime-facing facade in `reimagine_inference::*`) and must not
//! depend on paths such as
//! `inference_core::execution_value::handles::RuntimeLatent`.
//!
//! This envelope is **not** the workflow JSON value, **not** a host
//! DTO, **not** a snapshot / event payload, and **not** an Agent tool
//! result. It is the internal execution value used by runtime,
//! inference node executors, inference requests, and backend-produced
//! handles. Workflow JSON, snapshots, summaries, run events, Axum /
//! Tauri DTOs, and Agent tool results must not expose `ExecutionValue`,
//! backend tensor handles, or loaded model handles.

mod backend;
mod conditioning;
mod handles;
mod output;
mod tensor;
mod value;

pub use backend::BackendPayloadKey;
pub use conditioning::{ConditioningMetadata, ExecutionConditioning};
pub use handles::{
    BackendTensorHandle, RuntimeClipHandle, RuntimeImage, RuntimeLatent, RuntimeModelHandle,
    RuntimeVaeHandle,
};
pub use output::{ExecutionOutput, ExecutionValueRetention};
pub use tensor::BackendTensorMetadata;
pub use value::{ExecutionValue, ExecutionValueKind};
