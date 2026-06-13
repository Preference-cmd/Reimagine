//! Backend-neutral inference layer for Reimagine.
//!
//! This crate defines the operation-based backend protocol, executor
//! factory shape, and backend-neutral errors for running built-in
//! generation nodes without making `runtime`, `app-host`, or host
//! adapters depend on a concrete inference backend.
//!
//! See `docs/architecture/modules/inference.md` for the architecture
//! source of truth.

#![deny(unsafe_code)]

mod backend;
pub mod capability;
pub mod error;
mod executors;
pub mod operation;
pub mod registry;
pub mod request;
pub mod resolver;
pub mod response;
pub mod testing;

pub use backend::InferenceBackend;
pub use capability::{InferenceBackendCapabilities, InferenceOperationSupport};
pub use error::InferenceError;
pub use operation::InferenceOperationId;
pub use registry::register_builtin_inference_executors;
pub use request::InferenceRequest;
pub use resolver::{ModelFormat, ModelResolver, ResolvedInferenceModel};
pub use response::{InferenceOutput, InferenceResponse};
pub use testing::FakeBackend;
