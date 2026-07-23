//! Typed response DTOs grouped by capability.
//!
//! V1 typed responses return capability-specific values:
//!
//! - [`model`] — `LoadBundleResponse` (three handles)
//! - [`text`] — `TextEncodeResponse` (conditioning)
//! - [`latent`] — `CreateEmptyLatentResponse`, `LatentDecodeResponse`
//! - [`latent_encode`] — `LatentEncodeResponse` (`EncodedImage` latent)
//! - [`diffusion`] — `DiffusionSampleResponse`
//! - [`image`] — `ImageSaveResponse`, `ImagePreviewResponse`
//! - [`image_import`] — `ImageImportResponse`
//!
//! Responses never carry `SlotId` output mapping; slot mapping is
//! executor-owned.

pub mod diffusion;
pub mod image;
pub mod image_import;
pub mod latent;
pub mod latent_encode;
pub mod model;
pub mod text;
