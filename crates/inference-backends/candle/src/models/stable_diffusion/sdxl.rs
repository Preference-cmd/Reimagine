//! SDXL-specific loaded-model bundle and validation helpers.
//!
//! The bundle owns backend-local SDXL metadata (resolved source path,
//! device, format) and the three lightweight payload keys that flow
//! into runtime as `ExecutionValue::Model / Clip / Vae` handles. It
//! does not own any `candle_core::Tensor` yet — those land alongside
//! the real CLIP / UNet / VAE kernels.

mod bundle;
pub mod diffusion;
pub mod text;
pub mod tokenizer;
pub mod vae;

pub use bundle::LoadedSdxlBundle;
