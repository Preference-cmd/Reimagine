use reimagine_core::model::{
    ComponentRole, ModelFamily, NodeDef, NodeEffect, NodeResourceRequirements, SlotKind,
};

use super::{
    BUILTIN_PREVIEW_IMAGE, BUILTIN_SAVE_IMAGE, BUILTIN_VAE_DECODE, required_input, required_output,
};

pub fn vae_decode() -> NodeDef {
    NodeDef::new(BUILTIN_VAE_DECODE, "VAE Decode", "Image")
        .with_input_slot(required_input("vae", SlotKind::Vae, true))
        .with_input_slot(required_input("latent", SlotKind::Latent, true))
        .with_output_slot(required_output("image", SlotKind::Image))
        .with_resource_requirements(
            NodeResourceRequirements::new()
                .model_family(ModelFamily::stable_diffusion())
                .required_components(vec![ComponentRole::VaeDecoder])
                .estimated_vram_bytes(335_544_320), // ~320 MB
        )
}

pub fn save_image() -> NodeDef {
    NodeDef::new(BUILTIN_SAVE_IMAGE, "Save Image", "Image")
        .with_effect(NodeEffect::SideEffect)
        .with_input_slot(required_input("image", SlotKind::Image, true))
        .with_input_slot(required_input("filename_prefix", SlotKind::String, false))
}

pub fn preview_image() -> NodeDef {
    NodeDef::new(BUILTIN_PREVIEW_IMAGE, "Preview Image", "Image")
        .with_effect(NodeEffect::SideEffect)
        .with_input_slot(required_input("image", SlotKind::Image, true))
}
