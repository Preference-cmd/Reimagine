//! Inference-backed node executor adapters.
//!
//! Each executor maps a built-in node type to a typed capability
//! method on [`InferenceBackend`](crate::InferenceBackend), builds
//! the corresponding typed request DTO, calls the backend, and maps
//! the typed response into the workflow node's slot-shaped outputs.
//!
//! These are *adapters*, not backend implementations. They contain no
//! backend-specific behavior.

pub mod common;
pub mod diffusion;
pub mod image;
pub mod image_import;
pub mod latent;
pub mod latent_encode;
pub mod model;
pub mod string;
pub mod text;
pub mod validation;
