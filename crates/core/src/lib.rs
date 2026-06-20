//! Reimagine core: workflow engine + backend-neutral domain types.
//!
//! `core` is the pure domain kernel. It owns the canonical workflow
//! data model, node definition schema, command application semantics,
//! validation and diagnostics, history, execution planning schema,
//! run event schema, and backend-neutral saved/editor values.
//!
//! Internal execution values, backend-affine tensor / model / image
//! handles, and `ExecutionConditioning` live in
//! `reimagine-inference-core` and are re-exported by
//! `reimagine-inference` for runtime-facing code. `core` deliberately
//! does not own runtime execution handles or any value that wraps
//! backend-owned payload stores.

#![deny(unsafe_code)]

pub mod command;
pub mod diagnostic;
pub mod event;
pub mod execution_plan;
pub mod history;
pub mod model;
pub mod readiness;
pub mod session;
pub mod validation;
pub mod workflow;
