use reimagine_core::model::{NodeDef, SlotKind};

use super::{BUILTIN_STRING, required_input, required_output};

pub fn string() -> NodeDef {
    NodeDef::new(BUILTIN_STRING, "String", "Input")
        .with_input_slot(required_input("value", SlotKind::String, false))
        .with_output_slot(required_output("value", SlotKind::String))
}
