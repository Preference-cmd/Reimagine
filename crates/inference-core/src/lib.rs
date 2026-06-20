//! Execution value and backend contract crate for Reimagine.
//!
//! This crate owns:
//!
//! - The canonical internal [`ExecutionValue`](execution_value::ExecutionValue)
//!   envelope plus the backend-affine handle types that ride on it
//!   (model, clip, vae, latent, image, conditioning, tensor handles,
//!   and `BackendKind` / `BackendPayloadKey` identifiers). Runtime,
//!   inference executors, inference requests, and concrete backend
//!   adapters all pass these values through this envelope.
//! - The typed capability protocol between inference executors, the
//!   executor-facing router, and concrete inference backends.
//! - The backend registry, bridge policy, model resolver handoff,
//!   and the diagnostic / error mapping shared by every backend.
//!
//! `inference-core` deliberately does not own workflow scheduling,
//! node executors, concrete backend payloads, host DTOs, workflow
//! JSON, or UI / Agent observation shapes.
//!
//! The V1 typed surface lives under [`request`] and [`response`],
//! grouped by capability. [`capability::InferenceCapability`] is the
//! closed capability identity used for diagnostics, capability
//! reports, tracing, and bridge policy context. It is **not** the
//! runtime / backend dispatch key — the typed method call is the
//! dispatch.
//!
//! The runtime-facing facade re-exports the execution value types from
//! [`reimagine_inference`](../inference/index.html). New runtime code
//! should import execution values from `reimagine_inference`; backend
//! code may use `reimagine_inference_core` directly.

#![deny(unsafe_code)]

mod backend;
mod bridge;
mod capability;
mod diagnostic;
mod error;
mod execution_value;
mod registry;
mod request;
mod resolver;
mod response;
mod runtime;

pub use backend::InferenceBackend;
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
pub use error::InferenceError;
pub use execution_value::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, BackendTensorMetadata,
    ConditioningMetadata, ExecutionConditioning, ExecutionOutput, ExecutionValue,
    ExecutionValueKind, ExecutionValueRetention, RuntimeClipHandle, RuntimeImage, RuntimeLatent,
    RuntimeModelHandle, RuntimeVaeHandle,
};
pub use registry::{InferenceBackendRegistry, MergedInferenceBackendCapabilities};
pub use request::diffusion::{DiffusionSampleRequest, SamplerName, SchedulerName};
pub use request::image::{FilenamePrefix, ImagePreviewRequest, ImageSaveRequest};
pub use request::latent::{CreateEmptyLatentRequest, LatentDecodeRequest};
pub use request::model::LoadBundleRequest;
pub use request::text::TextEncodeRequest;
pub use resolver::{ModelFormat, ModelResolver, ResolvedInferenceModel};
pub use response::diffusion::DiffusionSampleResponse;
pub use response::image::{ImagePreviewResponse, ImageSaveResponse};
pub use response::latent::{CreateEmptyLatentResponse, LatentDecodeResponse};
pub use response::model::LoadBundleResponse;
pub use response::text::TextEncodeResponse;
pub use runtime::{DefaultInferenceRuntime, InferenceRuntime};
