//! SDXL text conditioning profile and loaded module graph.
//!
//! This module owns the SDXL-specific decisions about which CLIP-L
//! and CLIP-G profiles to use, what component-local prefix each
//! carries, and how the generated key-space maps to the existing
//! burn/03 contract vocabulary.
//!
//! It also provides the spec-generation bridge from the generic
//! [`TextEncoderSpecSet`](crate::text_encoder::specs::TextEncoderSpecSet)
//! into the [`BurnSdxlComponentContract`] surface so the existing
//! validation and writer infrastructure stays compatible.

pub(crate) mod cache;
pub mod loading;
pub mod module;
pub mod store;

use crate::text_encoder::specs::{TextEncoderSpecSet, TextEncoderSpecSetBuilder};

/// Generate the complete required spec set for a text-encoder
/// component identified by its role string. Returns `None` for
/// non-text roles (diffusion, vae) so the caller can dispatch.
pub fn sdxl_text_encoder_spec_set(component_role: &str) -> Option<TextEncoderSpecSet> {
    match component_role {
        "text_encoder" => Some(TextEncoderSpecSetBuilder::sdxl_clip_l()),
        "text_encoder_2" => Some(TextEncoderSpecSetBuilder::sdxl_open_clip_g()),
        _ => None,
    }
}
