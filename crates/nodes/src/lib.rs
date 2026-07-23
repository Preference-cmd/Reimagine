//! Built-in Reimagine node catalog.

#![deny(unsafe_code)]

mod aliases;
mod builtins;
mod registry;

pub use aliases::comfy_aliases;
pub use builtins::{
    BUILTIN_CHECKPOINT_LOADER, BUILTIN_CLIP_TEXT_ENCODE, BUILTIN_EMPTY_LATENT_IMAGE,
    BUILTIN_KSAMPLER, BUILTIN_LOAD_IMAGE, BUILTIN_PREVIEW_IMAGE, BUILTIN_SAVE_IMAGE,
    BUILTIN_STRING, BUILTIN_VAE_DECODE, BUILTIN_VAE_ENCODE, all_builtin_defs,
};
pub use registry::BuiltinNodeCatalog;
