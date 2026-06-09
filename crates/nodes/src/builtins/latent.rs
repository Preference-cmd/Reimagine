use reimagine_core::model::{NodeDef, SlotKind};

use super::{BUILTIN_EMPTY_LATENT_IMAGE, required_input, required_output};

pub fn empty_latent_image() -> NodeDef {
    NodeDef::new(BUILTIN_EMPTY_LATENT_IMAGE, "Empty Latent Image", "Latent")
        .with_input_slot(required_input("width", SlotKind::Integer, false))
        .with_input_slot(required_input("height", SlotKind::Integer, false))
        .with_input_slot(required_input("batch_size", SlotKind::Integer, false))
        .with_output_slot(required_output("latent", SlotKind::Latent))
}
