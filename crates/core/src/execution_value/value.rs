//! Public execution value envelope used by runtime, inference,
//! inference-core, backend crates, and host observations.
//!
//! The canonical name is `ExecutionValue`. V1 contains the variants
//! that the workflow executor, the inference executors, and the
//! backend adapter crates already need:
//!
//! - [`Param`](ExecutionValue::Param) for typed parameters that pass
//!   through the executor (strings, integers, model refs, etc.)
//! - [`Model`](ExecutionValue::Model) / [`Clip`](ExecutionValue::Clip) /
//!   [`Vae`](ExecutionValue::Vae) for loaded-model handles
//! - [`Latent`](ExecutionValue::Latent) for sampled / empty latent
//!   tensors
//! - [`Conditioning`](ExecutionValue::Conditioning) for text-encoded
//!   conditioning (carries [`ExecutionConditioning`](super::conditioning::ExecutionConditioning)
//!   plus its public metadata)
//! - [`Image`](ExecutionValue::Image) for decoded image tensors
//! - [`Artifact`](ExecutionValue::Artifact) for output artifact
//!   references
//! - [`Null`](ExecutionValue::Null) as the explicit no-value variant
//!
//! `ExecutionValue` is the run-time envelope. It is **not** the
//! workflow JSON value and it is **not** [`core::model::NodeValue`],
//! which stays as the saved/editor semantic value model.

use crate::model::{ArtifactRef, ParamValue};

use super::conditioning::ExecutionConditioning;
use super::handles::{
    BackendTensorHandle, RuntimeClipHandle, RuntimeImage, RuntimeLatent, RuntimeModelHandle,
    RuntimeVaeHandle,
};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionValue {
    Param(ParamValue),
    Model(RuntimeModelHandle),
    Clip(RuntimeClipHandle),
    Vae(RuntimeVaeHandle),
    Latent(RuntimeLatent),
    Conditioning(ExecutionConditioning),
    Image(RuntimeImage),
    Artifact(ArtifactRef),
    Null,
}

impl ExecutionValue {
    /// Returns the [`ParamValue`] when this value is
    /// [`ExecutionValue::Param`], otherwise `None`.
    pub fn as_param(&self) -> Option<&ParamValue> {
        match self {
            Self::Param(param) => Some(param),
            _ => None,
        }
    }

    /// Returns `Some(&t)` if the value is an image tensor, otherwise `None`.
    pub fn as_image(&self) -> Option<&RuntimeImage> {
        match self {
            Self::Image(image) => Some(image),
            _ => None,
        }
    }

    /// Returns `Some(&t)` if the value is a latent tensor, otherwise `None`.
    pub fn as_latent(&self) -> Option<&RuntimeLatent> {
        match self {
            Self::Latent(latent) => Some(latent),
            _ => None,
        }
    }

    /// Returns `Some(&t)` if the value is a conditioning, otherwise `None`.
    pub fn as_conditioning(&self) -> Option<&ExecutionConditioning> {
        match self {
            Self::Conditioning(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `Some(&t)` if the value is a model handle, otherwise `None`.
    pub fn as_model(&self) -> Option<&RuntimeModelHandle> {
        match self {
            Self::Model(m) => Some(m),
            _ => None,
        }
    }

    /// Returns `Some(&t)` if the value is a clip handle, otherwise `None`.
    pub fn as_clip(&self) -> Option<&RuntimeClipHandle> {
        match self {
            Self::Clip(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `Some(&t)` if the value is a vae handle, otherwise `None`.
    pub fn as_vae(&self) -> Option<&RuntimeVaeHandle> {
        match self {
            Self::Vae(v) => Some(v),
            _ => None,
        }
    }

    /// Returns `Some(&t)` if the value is an artifact reference, otherwise `None`.
    pub fn as_artifact(&self) -> Option<&ArtifactRef> {
        match self {
            Self::Artifact(a) => Some(a),
            _ => None,
        }
    }

    /// Returns `true` when the value is [`ExecutionValue::Null`].
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Discriminant kind. Useful for diagnostics, logging, and slot
    /// validation that does not need to consume the inner handle.
    pub fn kind(&self) -> ExecutionValueKind {
        match self {
            Self::Param(_) => ExecutionValueKind::Param,
            Self::Model(_) => ExecutionValueKind::Model,
            Self::Clip(_) => ExecutionValueKind::Clip,
            Self::Vae(_) => ExecutionValueKind::Vae,
            Self::Latent(_) => ExecutionValueKind::Latent,
            Self::Conditioning(_) => ExecutionValueKind::Conditioning,
            Self::Image(_) => ExecutionValueKind::Image,
            Self::Artifact(_) => ExecutionValueKind::Artifact,
            Self::Null => ExecutionValueKind::Null,
        }
    }
}

/// Discriminant kind for [`ExecutionValue`]. Used by callers that
/// want to inspect or log the variant without consuming the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionValueKind {
    Param,
    Model,
    Clip,
    Vae,
    Latent,
    Conditioning,
    Image,
    Artifact,
    Null,
}

impl ExecutionValueKind {
    /// Stable string label for diagnostics and DTO serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Param => "param",
            Self::Model => "model",
            Self::Clip => "clip",
            Self::Vae => "vae",
            Self::Latent => "latent",
            Self::Conditioning => "conditioning",
            Self::Image => "image",
            Self::Artifact => "artifact",
            Self::Null => "null",
        }
    }
}

impl std::fmt::Display for ExecutionValueKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// Suppress dead_code when downstream crates do not use a specific
// `as_*` accessor yet. They are part of the public envelope surface
// and may be wired into backend validation, slot-kind dispatch, or
// diagnostic projection in upcoming issues.
#[allow(dead_code)]
fn _ensure_handle_accessors_used(_: &BackendTensorHandle) {}
