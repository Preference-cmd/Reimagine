//! Runtime values passed between node executors during a single run.
//!
//! The canonical type now lives in
//! [`reimagine_core::execution_value`](reimagine_core::execution_value)
//! under the name [`ExecutionValue`](reimagine_core::ExecutionValue).
//! This module exists for two reasons:
//!
//! 1. Path stability: callers can continue to use
//!    `use reimagine_runtime::value::RuntimeValue;` without churn.
//! 2. Migration window: the historical `RuntimeValue` /
//!    `RuntimeConditioning` names are preserved as compatibility
//!    aliases until downstream crates fully migrate to the canonical
//!    `ExecutionValue` / `ExecutionConditioning` names.
//!
//! Backend-affine handle types keep their existing names
//! (`RuntimeModelHandle`, `RuntimeClipHandle`, `RuntimeVaeHandle`,
//! `RuntimeLatent`, `RuntimeImage`, `BackendTensorHandle`,
//! `ConditioningMetadata`, `BackendKind`, `BackendPayloadKey`) because
//! they are part of the public backend contract and have always been
//! defined next to the runtime value envelope.

pub use reimagine_core::execution_value::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, BackendTensorMetadata,
    ConditioningMetadata, ExecutionConditioning, ExecutionValue, ExecutionValueKind,
    RuntimeClipHandle, RuntimeImage, RuntimeLatent, RuntimeModelHandle, RuntimeVaeHandle,
};

/// Compatibility alias for [`ExecutionValue`].
///
/// Prefer [`ExecutionValue`](reimagine_core::ExecutionValue) for new
/// code; the alias exists so existing call sites that import
/// `RuntimeValue` from `reimagine_runtime` continue to compile.
pub type RuntimeValue = ExecutionValue;

/// Compatibility alias for [`ExecutionConditioning`].
///
/// Prefer [`ExecutionConditioning`](reimagine_core::ExecutionConditioning)
/// for new code.
pub type RuntimeConditioning = ExecutionConditioning;
