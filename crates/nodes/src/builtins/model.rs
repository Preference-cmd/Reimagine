use reimagine_core::model::{NodeDef, SlotKind};

use super::{BUILTIN_CHECKPOINT_LOADER, required_input, required_output};

pub fn checkpoint_loader() -> NodeDef {
    NodeDef::new(BUILTIN_CHECKPOINT_LOADER, "Checkpoint Loader", "Model")
        .with_input_slot(required_input("checkpoint", SlotKind::ModelRef, false))
        .with_output_slot(required_output("model", SlotKind::Model))
        .with_output_slot(required_output("clip", SlotKind::Clip))
        .with_output_slot(required_output("vae", SlotKind::Vae))
}
