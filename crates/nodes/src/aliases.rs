use std::collections::BTreeMap;

use crate::builtins::{
    BUILTIN_CHECKPOINT_LOADER, BUILTIN_CLIP_TEXT_ENCODE, BUILTIN_EMPTY_LATENT_IMAGE,
    BUILTIN_KSAMPLER, BUILTIN_SAVE_IMAGE, BUILTIN_VAE_DECODE,
};

pub fn comfy_aliases() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("CheckpointLoaderSimple", BUILTIN_CHECKPOINT_LOADER),
        ("CLIPTextEncode", BUILTIN_CLIP_TEXT_ENCODE),
        ("EmptyLatentImage", BUILTIN_EMPTY_LATENT_IMAGE),
        ("KSampler", BUILTIN_KSAMPLER),
        ("VAEDecode", BUILTIN_VAE_DECODE),
        ("SaveImage", BUILTIN_SAVE_IMAGE),
    ])
}
