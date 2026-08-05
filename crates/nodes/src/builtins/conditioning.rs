use reimagine_core::model::{ComponentRole, ModelFamily, NodeDef, NodeResourceRequirements, SlotKind};

use super::{BUILTIN_CLIP_TEXT_ENCODE, required_input, required_output};

pub fn clip_text_encode() -> NodeDef {
    NodeDef::new(BUILTIN_CLIP_TEXT_ENCODE, "CLIP Text Encode", "Conditioning")
        .with_input_slot(required_input("clip", SlotKind::Clip, true))
        .with_input_slot(required_input("text", SlotKind::String, true))
        .with_output_slot(required_output("conditioning", SlotKind::Conditioning))
        .with_resource_requirements(
            NodeResourceRequirements::new()
                .model_family(ModelFamily::stable_diffusion())
                .required_components(vec![
                    ComponentRole::TextEncoder,
                    ComponentRole::SecondaryTextEncoder,
                ])
                .estimated_vram_bytes(1_610_612_736), // ~1.5 GB
        )
}
