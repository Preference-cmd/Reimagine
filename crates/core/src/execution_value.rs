//! Public execution value model for Reimagine.
//!
//! This module owns the canonical [`ExecutionValue`] envelope and the
//! backend-affine handle types that ride on it. It is the source of
//! truth for run-time value semantics and is shared by `runtime`,
//! `inference`, `inference-core`, and concrete backend crates.
//!
//! The submodules stay private; downstream code should use the
//! top-level re-exports from `reimagine_core::*` (or the
//! compatibility aliases in `reimagine_runtime::*` during the
//! migration window) and must not depend on paths such as
//! `core::execution_value::handles::RuntimeLatent`.
//!
//! This envelope is **not** the workflow JSON value and is **not**
//! [`core::model::NodeValue`], which stays as the saved/editor
//! semantic value model.

mod backend;
mod conditioning;
mod handles;
mod tensor;
mod value;

pub use backend::{BackendKind, BackendPayloadKey};
pub use conditioning::{ConditioningMetadata, ExecutionConditioning};
pub use handles::{
    BackendTensorHandle, RuntimeClipHandle, RuntimeImage, RuntimeLatent, RuntimeModelHandle,
    RuntimeVaeHandle,
};
pub use tensor::BackendTensorMetadata;
pub use value::{ExecutionValue, ExecutionValueKind};
