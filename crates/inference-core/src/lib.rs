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
//!
//! The V1 typed surface lives under [`request`] and [`response`],
//! grouped by capability. [`capability::InferenceCapability`] is the
//! closed capability identity used for diagnostics, capability
//! reports, tracing, and bridge policy context. It is **not** the
//! runtime/backend dispatch key — the typed method call is the
//! dispatch.

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
pub use capability::{
    InferenceBackendCapabilities, InferenceCapability, InferenceCapabilitySupport,
};
pub use diagnostic::{
    backend_bridge_required, backend_bridge_unsupported, backend_capability_unsupported,
    backend_not_registered, incompatible_handle_affinity,
};
pub use error::InferenceError;
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
