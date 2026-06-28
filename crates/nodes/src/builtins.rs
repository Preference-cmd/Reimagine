use reimagine_core::model::NodeDef;

mod conditioning;
mod image;
mod inputs;
mod latent;
mod model;
mod sampling;

pub const BUILTIN_STRING: &str = "builtin.string";
pub const BUILTIN_CHECKPOINT_LOADER: &str = "builtin.checkpoint_loader";
pub const BUILTIN_CLIP_TEXT_ENCODE: &str = "builtin.clip_text_encode";
pub const BUILTIN_EMPTY_LATENT_IMAGE: &str = "builtin.empty_latent_image";
pub const BUILTIN_KSAMPLER: &str = "builtin.ksampler";
pub const BUILTIN_VAE_DECODE: &str = "builtin.vae_decode";
pub const BUILTIN_VAE_ENCODE: &str = "builtin.vae_encode";
pub const BUILTIN_LOAD_IMAGE: &str = "builtin.load_image";
pub const BUILTIN_SAVE_IMAGE: &str = "builtin.save_image";
pub const BUILTIN_PREVIEW_IMAGE: &str = "builtin.preview_image";

pub fn all_builtin_defs() -> Vec<NodeDef> {
    vec![
        inputs::string(),
        inputs::load_image(),
        model::checkpoint_loader(),
        conditioning::clip_text_encode(),
        latent::empty_latent_image(),
        latent::vae_encode(),
        sampling::ksampler(),
        image::vae_decode(),
        image::save_image(),
        image::preview_image(),
    ]
}

fn required_input(
    id: &str,
    kind: reimagine_core::model::SlotKind,
    dynamic: bool,
) -> reimagine_core::model::InputSlotDef {
    reimagine_core::model::InputSlotDef::new(id, kind)
        .dynamic(dynamic)
        .required(true)
}

fn required_output(
    id: &str,
    kind: reimagine_core::model::SlotKind,
) -> reimagine_core::model::OutputSlotDef {
    reimagine_core::model::OutputSlotDef::new(id, kind).required(true)
}
