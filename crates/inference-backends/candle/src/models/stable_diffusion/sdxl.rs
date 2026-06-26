//! SDXL-specific loaded-model bundle and validation helpers.
//!
//! The bundle owns backend-local SDXL metadata (resolved source path,
//! device, format) and the three lightweight payload keys that flow
//! into runtime as `ExecutionValue::Model / Clip / Vae` handles. It
//! does not own any `candle_core::Tensor` yet — those land alongside
//! the real CLIP / UNet / VAE kernels.

mod bundle;
pub mod checkpoint_import;
pub(crate) mod checkpoint_inventory;
pub(crate) mod checkpoint_projection;
pub mod diffusion;
pub(crate) mod diffusion_graph;
pub(crate) mod diffusion_sources;
pub mod text;
pub(crate) mod text_sources;
pub mod tokenizer;
pub mod vae;
pub(crate) mod vae_sources;

pub use bundle::LoadedSdxlBundle;
