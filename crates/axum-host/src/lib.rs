//! Axum HTTP host adapter for the Reimagine workspace.
//!
//! This crate is a peer to `src-tauri`: a host adapter that exposes the
//! host-neutral `app-host` facade over HTTP. It is the canonical
//! end-to-end test harness for workflow execution and is intentionally
//! thin — it owns Axum routing, request/response serialization, and
//! shared state. It must not reimplement workflow, model, runtime, or
//! Agent semantics; those live in the lower-level crates.
//!
//! See `docs/architecture/modules/axum-host.md` for the architecture
//! source of truth and `.scratch/axum-host/issues/01-...` for the V1
//! scope.

#![deny(unsafe_code)]

pub mod bootstrap;
pub mod cli;
mod dto;
mod error;
mod recorder;
mod router;
mod routes;
mod server;
mod state;

pub use error::{AxumHostError, AxumHostResult};
pub use recorder::RunEventRecorder;
pub use router::build_router;
pub use server::{AxumServerHandle, run_server, run_server_with_listener};
pub use state::AxumHostState;
