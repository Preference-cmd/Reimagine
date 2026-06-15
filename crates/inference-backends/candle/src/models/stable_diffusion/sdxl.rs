//! SDXL-specific loaded-model bundle and validation helpers.
//!
//! The bundle owns backend-local SDXL metadata (resolved source path,
//! device, format) and the three lightweight payload keys that flow
//! into runtime as `RuntimeValue::Model / Clip / Vae` handles. It does
//! not own any `candle_core::Tensor` yet — those land alongside the
//! real CLIP / UNet / VAE kernels.

mod bundle;
pub mod text;
pub mod tokenizer;

pub use bundle::LoadedSdxlBundle;
