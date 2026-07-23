use reimagine_core::model::{NodeDef, SlotKind};

use super::{BUILTIN_CLIP_TEXT_ENCODE, required_input, required_output};

pub fn clip_text_encode() -> NodeDef {
    NodeDef::new(BUILTIN_CLIP_TEXT_ENCODE, "CLIP Text Encode", "Conditioning")
        .with_input_slot(required_input("clip", SlotKind::Clip, true))
        .with_input_slot(required_input("text", SlotKind::String, true))
        .with_output_slot(required_output("conditioning", SlotKind::Conditioning))
}
