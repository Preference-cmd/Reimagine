use reimagine_core::model::{NodeDef, NodeEffect, SlotKind};

use super::{BUILTIN_LOAD_IMAGE, BUILTIN_STRING, required_input, required_output};

pub fn string() -> NodeDef {
    NodeDef::new(BUILTIN_STRING, "String", "Input")
        .with_input_slot(required_input("value", SlotKind::String, false))
        .with_output_slot(required_output("value", SlotKind::String))
}

/// `builtin.load_image` node definition.
///
/// Reads a `Path` param (workspace-safe; the executor routes the
/// path through an app-host resolver before reaching the backend)
/// and returns an `Image` handle on the backend's payload store.
/// Effect is `Pure` — the side effect of decoding happens inside
/// the backend's payload store, not on the workflow node.
pub fn load_image() -> NodeDef {
    NodeDef::new(BUILTIN_LOAD_IMAGE, "Load Image", "Input")
        .with_effect(NodeEffect::Pure)
        .with_input_slot(required_input("image", SlotKind::Path, false))
        .with_output_slot(required_output("image", SlotKind::Image))
}
