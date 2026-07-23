use reimagine_core::model::{NodeDef, SlotKind};

use super::{BUILTIN_EMPTY_LATENT_IMAGE, BUILTIN_VAE_ENCODE, required_input, required_output};

pub fn empty_latent_image() -> NodeDef {
    NodeDef::new(BUILTIN_EMPTY_LATENT_IMAGE, "Empty Latent Image", "Latent")
        .with_input_slot(required_input("width", SlotKind::Integer, false))
        .with_input_slot(required_input("height", SlotKind::Integer, false))
        .with_input_slot(required_input("batch_size", SlotKind::Integer, false))
        .with_output_slot(required_output("latent", SlotKind::Latent))
}

/// `builtin.vae_encode` node definition.
///
/// Consumes a `Vae` handle and an `Image` handle and returns a
/// `Latent`. The latent's runtime content class is `EncodedImage`;
/// downstream capabilities (decode, partial-denoise sample) can
/// therefore distinguish encoded latents from empty geometry.
pub fn vae_encode() -> NodeDef {
    NodeDef::new(BUILTIN_VAE_ENCODE, "VAE Encode", "Latent")
        .with_input_slot(required_input("vae", SlotKind::Vae, true))
        .with_input_slot(required_input("image", SlotKind::Image, true))
        .with_output_slot(required_output("latent", SlotKind::Latent))
}
