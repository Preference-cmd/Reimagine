//! Backend-neutral node orchestration layer for built-in V1
//! inference-backed node executors.
//!
//! The backend contract (trait, typed request/response DTOs,
//! capability report, model resolver, runtime/router, bridge policy,
//! registry) lives in `reimagine-inference-core`. This crate
//! re-exports those public types so existing
//! `use reimagine_inference::*` call sites keep working, and owns
//! the built-in `NodeExecutor` implementations plus the executor
//! registration helper.
//!
//! The canonical execution value envelope and backend-affine handle
//! types (`ExecutionValue`, `ExecutionValueKind`, `BackendKind`,
//! `BackendPayloadKey`, `BackendTensorHandle`,
//! `BackendTensorMetadata`, `RuntimeModelHandle`, `RuntimeClipHandle`,
//! `RuntimeVaeHandle`, `RuntimeLatent`, `RuntimeImage`,
//! `ExecutionConditioning`, `ConditioningMetadata`) plus the
//! producer-declared output contract (`ExecutionOutput`,
//! `ExecutionValueRetention`) are owned by
//! `reimagine-inference-core` and re-exported here as the
//! runtime-facing facade. New runtime-facing code should import
//! execution values from `reimagine_inference::*`.
//!
//! See `docs/architecture/modules/inference.md` for the
//! architecture source of truth.

#![deny(unsafe_code)]

mod error;
mod executors;
pub mod operation;
pub mod registry;
/// Test-only fake backend and canned-response helpers.
///
/// Compiled for both unit tests and integration tests so downstream
/// `tests/` files can register executors with a `FakeBackend`
/// without enabling a feature flag. Production code must not depend
/// on this module.
#[doc(hidden)]
pub mod testing;

pub use reimagine_inference_core::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, BackendTensorMetadata,
    ConditioningMetadata, ExecutionConditioning, ExecutionOutput, ExecutionValue,
    ExecutionValueKind, ExecutionValueRetention, RuntimeClipHandle, RuntimeImage, RuntimeLatent,
    RuntimeModelHandle, RuntimeVaeHandle,
};

pub use reimagine_inference_core::{
    BackendBridge, BackendBridgePolicy, BridgePlan, BridgeSupport, CreateEmptyLatentRequest,
    CreateEmptyLatentResponse, DefaultInferenceRuntime, DiffusionSampleRequest,
    DiffusionSampleResponse, FilenamePrefix, ImagePreviewRequest, ImagePreviewResponse,
    ImageSaveRequest, ImageSaveResponse, InferenceBackend, InferenceBackendCapabilities,
    InferenceBackendRegistry, InferenceCapability, InferenceCapabilitySupport, InferenceError,
    InferenceRuntime, LatentDecodeRequest, LatentDecodeResponse, LoadBundleRequest,
    LoadBundleResponse, MergedInferenceBackendCapabilities, ModelFormat, ModelResolver,
    RejectAllBridgePolicy, ResolvedInferenceModel, SamplerName, SchedulerName, TextEncodeRequest,
    TextEncodeResponse,
};

pub use error::{IntoNodeExecutorError, into_executor_error};
pub use registry::register_builtin_inference_executors;
#[doc(hidden)]
pub use testing::{CannedCapabilityResponse, FakeBackend};
