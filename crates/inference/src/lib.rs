//! Backend-neutral node orchestration layer for built-in V1
//! inference-backed node executors.
//!
//! The backend contract (trait, request/response DTOs, capability
//! report, model resolver, runtime/router, bridge policy,
//! registry) lives in `reimagine-inference-core`. This crate
//! re-exports those public types so existing
//! `use reimagine_inference::*` call sites keep working, and owns
//! the built-in `NodeExecutor` implementations plus the executor
//! registration helper.
//!
//! See `docs/architecture/modules/inference.md` for the
//! architecture source of truth.

#![deny(unsafe_code)]

mod error;
mod executors;
pub mod operation;
pub mod registry;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use reimagine_inference_core::{
    ALL_V1_OPERATIONS, BackendBridge, BackendBridgePolicy, BridgePlan, BridgeSupport,
    DefaultInferenceRuntime, InferenceBackend, InferenceBackendCapabilities,
    InferenceBackendRegistry, InferenceError, InferenceOperationId, InferenceOperationSupport,
    InferenceOutput, InferenceRequest, InferenceResponse, InferenceRuntime,
    MergedInferenceBackendCapabilities, ModelFormat, ModelResolver, OP_DIFFUSION_SAMPLE,
    OP_IMAGE_PREVIEW, OP_IMAGE_SAVE, OP_LATENT_CREATE_EMPTY, OP_LATENT_DECODE,
    OP_MODEL_LOAD_BUNDLE, OP_TEXT_ENCODE, RejectAllBridgePolicy, ResolvedInferenceModel,
};

pub use error::{IntoNodeExecutorError, into_executor_error};
pub use registry::register_builtin_inference_executors;
#[cfg(any(test, feature = "testing"))]
pub use testing::FakeBackend;
