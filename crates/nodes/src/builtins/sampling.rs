use reimagine_core::model::{
    ComponentRole, ModelFamily, NodeDef, NodeResourceRequirements, SlotKind,
};

use super::{BUILTIN_KSAMPLER, required_input, required_output};

pub fn ksampler() -> NodeDef {
    NodeDef::new(BUILTIN_KSAMPLER, "KSampler", "Sampling")
        .with_input_slot(required_input("model", SlotKind::Model, true))
        .with_input_slot(required_input("positive", SlotKind::Conditioning, true))
        .with_input_slot(required_input("negative", SlotKind::Conditioning, true))
        .with_input_slot(required_input("latent", SlotKind::Latent, true))
        .with_input_slot(required_input("seed", SlotKind::Seed, false))
        .with_input_slot(required_input("steps", SlotKind::Integer, false))
        .with_input_slot(required_input("cfg", SlotKind::Float, false))
        .with_input_slot(required_input("sampler", SlotKind::Select, false))
        .with_input_slot(required_input("scheduler", SlotKind::Select, false))
        .with_input_slot(required_input("denoise", SlotKind::Float, false))
        .with_output_slot(required_output("latent", SlotKind::Latent))
        .with_resource_requirements(
            NodeResourceRequirements::new()
                .model_family(ModelFamily::stable_diffusion())
                .required_components(vec![ComponentRole::DiffusionModel])
                .estimated_vram_bytes(3_758_096_384), // ~3.5 GB
        )
}
