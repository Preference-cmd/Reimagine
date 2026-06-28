//! Typed request DTOs grouped by capability.
//!
//! V1 typed requests own cheap, shareable core handles plus capability-
//! specific parameters. They do **not** carry `operation_id`: the
//! method call itself is the capability identity. They also do **not**
//! carry `SlotId` output mapping, because slot mapping belongs to the
//! inference executor that knows the workflow node shape.
//!
//! The submodules organize the request DTOs by capability:
//!
//! - [`model`] — `LoadBundleRequest`
//! - [`text`] — `TextEncodeRequest`
//! - [`latent`] — `CreateEmptyLatentRequest`, `LatentDecodeRequest`
//! - [`latent_encode`] — `LatentEncodeRequest`
//! - [`diffusion`] — `DiffusionSampleRequest`
//! - [`image`] — `ImageSaveRequest`, `ImagePreviewRequest`
//! - [`image_import`] — `ImageImportRequest`

pub mod diffusion;
pub mod image;
pub mod image_import;
pub mod latent;
pub mod latent_encode;
pub mod model;
pub mod text;
