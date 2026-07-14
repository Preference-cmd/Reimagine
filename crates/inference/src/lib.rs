//! Backend-neutral inference facade for built-in V1
//! inference-backed node executors and concrete inference backends.
//!
//! This crate owns the backend contract (trait, typed
//! request/response DTOs, capability report, model resolver,
//! runtime/router, bridge policy, backend registry,
//! backend-instance runtime hooks), canonical execution values, the **node executor
//! contract** (`NodeExecutor` trait, `NodeExecutionContext`,
//! `NodeInputs`/`NodeParams`, `NodeExecutorRegistry`,
//! `ArtifactPublisher`, `NodeCancellation`, `ArtifactEventKind`,
//! `NodeExecutorError`), built-in V1 executors, and the executor
//! registration helper.
//!
//! This crate deliberately does **not** depend on `reimagine-runtime`.
//! The runtime depends on this crate as its executor/value facade.
//! Concrete artifact and cancellation impls live in the runtime and
//! are wrapped in trait objects at context construction time, so the
//! executor contract stays backend- and runtime-neutral.
//!
//! Runtime-facing and backend-facing code should import execution
//! values, backend contracts, router contracts, backend-instance hook
//! contracts, and executor contracts from `reimagine_inference::*`.
//!
//! See `docs/architecture/modules/inference.md` for the
//! architecture source of truth.

#![deny(unsafe_code)]

mod artifact_publisher;
mod backend;
mod backend_registry;
mod backend_selection;
mod bridge;
mod cancellation;
mod capability;
mod diagnostic;
mod error;
mod execution_value;
mod executor;
mod executors;
mod inference_error;
mod invocation;
pub mod latent_content;
pub mod latent_space;
pub mod node_context;
pub mod operation;
mod profile;
pub mod registry;
mod request;
mod resolver;
mod resources;
mod response;
mod router;
mod routing_request;
/// Test-only fake backend and canned-response helpers.
///
/// Compiled for both unit tests and integration tests so downstream
/// `tests/` files can register executors with a `FakeBackend`
/// without enabling a feature flag. Production code must not depend
/// on this module.
#[doc(hidden)]
pub mod testing;

pub use execution_value::{
    BackendPayloadKey, BackendTensorHandle, BackendTensorMetadata, ConditioningMetadata,
    ExecutionConditioning, ExecutionOutput, ExecutionValue, ExecutionValueKind,
    ExecutionValueRetention, RuntimeClipHandle, RuntimeImage, RuntimeLatent, RuntimeModelHandle,
    RuntimeVaeHandle,
};

pub use latent_content::{LatentContent, LatentContentError};
pub use latent_space::{
    LatentSpaceError, LatentSpaceId, LatentSpaceMetadata, TensorLayout, stable_diffusion_sdxl_base,
};

pub use backend::InferenceBackend;
pub use backend_registry::{InferenceBackendRegistry, MergedInferenceBackendCapabilities};
pub use backend_selection::{
    ArcBackendSelectionPolicy, Backend, BackendInstance, BackendInstanceDescriptor,
    BackendOverrides, BackendSelectionOverlay, BackendSelectionPolicy, BackendSelectionRequest,
    DeviceProfile, StaticBackendSelectionPolicy,
};
pub use bridge::{
    BackendBridge, BackendBridgePolicy, BridgePlan, BridgeSupport, RejectAllBridgePolicy,
};
pub use capability::{
    InferenceBackendCapabilities, InferenceCapability, InferenceCapabilitySupport,
};
pub use diagnostic::{
    backend_bridge_required, backend_bridge_unsupported, backend_capability_unsupported,
    backend_not_registered, incompatible_handle_affinity,
};
pub use inference_error::InferenceError;
pub use invocation::{
    InferenceInvocation, InferenceProgress, InferenceProgressSink, NoopInferenceProgressSink,
};
pub use profile::{
    BackendInstanceProfile, BackendInstanceStatus, BackendProfile, BackendProfileProvider,
    DTypeProfile, DeviceKind, MemoryProfile, OperationOptionsProfile, OperationOptionsProfileKind,
    SamplerOptionProfile, SamplerSchedulerPairProfile, SchedulerOptionProfile,
    WorkspaceComputeProfile, diagnostics, kind_from_label,
};
pub use request::diffusion::{
    DiffusionDenoiseError, DiffusionDenoiseMode, DiffusionSampleRequest, SamplerName, SchedulerName,
};
pub use request::image::{FilenamePrefix, ImagePreviewRequest, ImageSaveRequest};
pub use request::image_import::{ImageImportRequest, ResolvedImageSource};
pub use request::latent::{CreateEmptyLatentRequest, LatentDecodeRequest};
pub use request::latent_encode::LatentEncodeRequest;
pub use request::model::LoadBundleRequest;
pub use request::text::TextEncodeRequest;
pub use resolver::{
    ModelFormat, ModelResolver, ModelSourceKind, ResolvedInferenceModel,
    ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};
pub use resources::{
    BackendInstanceObservation, BackendInstanceRuntimeHooks, BackendInstanceSnapshot,
    BackendResourceMechanism, BackendResourceObservation, BackendResourceSnapshot,
    BackendRunLifecycle, BackendRunLifecycleReport, BackendRunLifecycleRequest,
    CompositeBackendInstanceRuntimeHooks,
};
pub use response::diffusion::DiffusionSampleResponse;
pub use response::image::{ImagePreviewResponse, ImageSaveResponse};
pub use response::image_import::ImageImportResponse;
pub use response::latent::{CreateEmptyLatentResponse, LatentDecodeResponse};
pub use response::latent_encode::LatentEncodeResponse;
pub use response::model::LoadBundleResponse;
pub use response::text::TextEncodeResponse;
pub use router::{DefaultInferenceRuntime, InferenceRuntime};

pub use artifact_publisher::{ArtifactEventKind, ArtifactPublisher};
pub use cancellation::NodeCancellation;
pub use error::{IntoNodeExecutorError, into_executor_error};
pub use executor::{
    BoxedNodeExecutor, NodeExecutionOutputs, NodeExecutor, NodeExecutorError, NodeExecutorRegistry,
    NodeExecutorRegistryError,
};
pub use executors::image_import::{ImageSourceResolver, LoadImageExecutor};
pub use node_context::{NodeExecutionContext, NodeInputs, NodeParams};
pub use registry::register_builtin_inference_executors;
#[doc(hidden)]
pub use testing::{
    CannedCapabilityResponse, FakeBackend, NoopArtifactPublisher, NoopNodeCancellation,
    RecordedArtifact, RecordingArtifactPublisher,
};
