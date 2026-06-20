//! Backend-neutral node orchestration layer for built-in V1
//! inference-backed node executors.
//!
//! The backend contract (trait, typed request/response DTOs,
//! capability report, model resolver, runtime/router, bridge policy,
//! registry, resource lifecycle) lives in
//! `reimagine-inference-core`. This crate owns the **node executor
//! contract** (`NodeExecutor` trait, `NodeExecutionContext`,
//! `NodeInputs`/`NodeParams`, `NodeExecutorRegistry`,
//! `ArtifactPublisher`, `NodeCancellation`, `ArtifactEventKind`,
//! `NodeExecutorError`) plus the built-in V1 executors and the
//! executor registration helper.
//!
//! This crate deliberately does **not** depend on `reimagine-runtime`.
//! The runtime depends on this crate as its executor/value facade.
//! Concrete artifact and cancellation impls live in the runtime and
//! are wrapped in trait objects at context construction time, so the
//! executor contract stays backend- and runtime-neutral.
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

mod artifact_publisher;
mod cancellation;
mod error;
mod executor;
mod executors;
pub mod node_context;
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

pub use artifact_publisher::{ArtifactEventKind, ArtifactPublisher};
pub use cancellation::NodeCancellation;
pub use error::{IntoNodeExecutorError, into_executor_error};
pub use executor::{
    BoxedNodeExecutor, NodeExecutionOutputs, NodeExecutor, NodeExecutorError, NodeExecutorRegistry,
    NodeExecutorRegistryError,
};
pub use node_context::{NodeExecutionContext, NodeInputs, NodeParams};
pub use registry::register_builtin_inference_executors;
#[doc(hidden)]
pub use testing::{
    CannedCapabilityResponse, FakeBackend, NoopArtifactPublisher, NoopNodeCancellation,
    RecordedArtifact, RecordingArtifactPublisher,
};
