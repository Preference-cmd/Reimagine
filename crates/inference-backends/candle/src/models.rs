//! Backend-owned loaded model entries.
//!
//! `LoadedModelBundle` is the family-aware wrapper that
//! `CandleModelCache` stores. Each model family (stable_diffusion,
//! flux, ...) lives under its own module and contributes a variant
//! to the wrapper.

pub mod stable_diffusion;

pub use stable_diffusion::LoadedModelBundle;
pub use stable_diffusion::sdxl::LoadedSdxlBundle;
