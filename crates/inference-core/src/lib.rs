//! Backend contract crate for Reimagine.
//!
//! This crate defines the typed capability protocol between inference
//! executors, the executor-facing router, and concrete inference
//! backends. It also owns bridge policy, the backend registry, and
//! the diagnostic / error mapping shared by every backend.
//!
//! The crate must not own workflow scheduling, node executors,
//! concrete backend payloads, or public execution values. Public
//! execution values and backend-affine handles belong to
//! `reimagine-core`; built-in node executors and the executor
//! registration helper belong to `reimagine-inference`.

#![deny(unsafe_code)]

mod backend;
mod bridge;
mod capability;
mod diagnostic;
mod error;
mod registry;
mod request;
mod resolver;
mod response;
mod runtime;

pub use backend::InferenceBackend;
pub use bridge::{
    BackendBridge, BackendBridgePolicy, BridgePlan, BridgeSupport, RejectAllBridgePolicy,
};
pub use capability::{InferenceBackendCapabilities, InferenceOperationSupport};
pub use diagnostic::{
    backend_bridge_required, backend_bridge_unsupported, backend_capability_unsupported,
    backend_not_registered, incompatible_handle_affinity,
};
pub use error::InferenceError;
pub use registry::{InferenceBackendRegistry, MergedInferenceBackendCapabilities};
pub use request::{
    ALL_V1_OPERATIONS, InferenceOperationId, InferenceRequest, OP_DIFFUSION_SAMPLE,
    OP_IMAGE_PREVIEW, OP_IMAGE_SAVE, OP_LATENT_CREATE_EMPTY, OP_LATENT_DECODE,
    OP_MODEL_LOAD_BUNDLE, OP_TEXT_ENCODE,
};
pub use resolver::{ModelFormat, ModelResolver, ResolvedInferenceModel};
pub use response::{InferenceOutput, InferenceResponse};
pub use runtime::{DefaultInferenceRuntime, InferenceRuntime};
