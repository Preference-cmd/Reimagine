//! Inference-backed node executor adapters.
//!
//! Each executor maps a built-in node type to an
//! [`InferenceOperationId`](crate::operation::InferenceOperationId),
//! builds an [`InferenceRequest`](crate::request::InferenceRequest),
//! calls the backend, validates the response, and returns
//! `NodeExecutionOutputs`.
//!
//! These are *adapters*, not backend implementations. They contain no
//! backend-specific behavior.

pub mod diffusion;
pub mod image;
pub mod latent;
pub mod model;
pub mod string;
pub mod text;
pub mod validation;
