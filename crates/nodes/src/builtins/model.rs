use reimagine_core::model::{
    ComponentRole, ModelFamily, NodeDef, NodeResourceRequirements, SlotConstraint, SlotKind,
};

use super::{BUILTIN_CHECKPOINT_LOADER, required_input, required_output};

/// Selectable checkpoint file names offered by the V1 checkpoint loader.
pub const CHECKPOINT_OPTIONS: &str =
    "sdxl_base_1.0.safetensors,dreamshaper_8.safetensors,rev_animated.safetensors";

pub fn checkpoint_loader() -> NodeDef {
    NodeDef::new(BUILTIN_CHECKPOINT_LOADER, "Checkpoint Loader", "Model")
        .with_input_slot(
            required_input("checkpoint", SlotKind::ModelRef, false)
                .with_constraint(SlotConstraint::new("options", CHECKPOINT_OPTIONS)),
        )
        .with_output_slot(required_output("model", SlotKind::Model))
        .with_output_slot(required_output("clip", SlotKind::Clip))
        .with_output_slot(required_output("vae", SlotKind::Vae))
        .with_resource_requirements(
            NodeResourceRequirements::new()
                .model_family(ModelFamily::stable_diffusion())
                .required_components(vec![ComponentRole::CheckpointBundle])
                .estimated_vram_bytes(6_442_450_944), // ~6 GB
        )
}
