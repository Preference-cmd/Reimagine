//! Typed response DTOs grouped by capability.
//!
//! V1 typed responses return capability-specific values:
//!
//! - [`model`] — `LoadBundleResponse` (three handles)
//! - [`text`] — `TextEncodeResponse` (conditioning)
//! - [`latent`] — `CreateEmptyLatentResponse`, `LatentDecodeResponse`
//! - [`diffusion`] — `DiffusionSampleResponse`
//! - [`image`] — `ImageSaveResponse`, `ImagePreviewResponse`
//!
//! Responses never carry `SlotId` output mapping; slot mapping is
//! executor-owned.

pub mod diffusion;
pub mod image;
pub mod latent;
pub mod model;
pub mod text;
