//! Burn-private text-encoder contract primitives.
//!
//! This module owns the deterministic key-space and tensor-family vocabulary
//! needed by Burn's executable CLIP-L / CLIP-G module structs. It is data-only;
//! no Burn `Module` definitions live here, and no Burn tensors are allocated
//! at this layer.
//!
//! The actual SDXL profile that wires CLIP-L and CLIP-G lives in
//! `crate::models::stable_diffusion::sdxl::text_conditioning`. The seam is
//! intentionally narrow: just enough for the V1 SDXL path, with room for a
//! second concrete model to deepen the abstraction later.

pub(crate) mod clip;
pub(crate) mod keyspace;
pub(crate) mod specs;

pub use clip::{ClipTextEncoderProfile, ClipTextEncoderVariant};
pub use keyspace::{TextEncoderKeyspace, TextEncoderTensorFamily};
pub use specs::{OwnedTensorSpec, TextEncoderSpecSet, TextEncoderSpecSetBuilder, text_encoder_spec_set};
