//! Executor registration for V1 built-in node types.
//!
//! [`register_builtin_inference_executors`] installs all inference-backed
//! executors into a [`NodeExecutorRegistry`](reimagine_runtime::NodeExecutorRegistry).
//! It requires a backend, a resolver, and an already-existing
//! registry (usually the one owned by the [`RuntimeService`](reimagine_runtime::RuntimeService)).

use std::sync::Arc;

use reimagine_runtime::NodeExecutorRegistry;

use crate::backend::InferenceBackend;
use crate::executors::{
    diffusion::KSamplerExecutor, image::PreviewImageExecutor, image::SaveImageExecutor,
    image::VaeDecodeExecutor, latent::EmptyLatentImageExecutor, model::CheckpointLoaderExecutor,
    string::StringExecutor, text::ClipTextEncodeExecutor,
};
use crate::resolver::ModelResolver;

/// Register all V1 built-in inference-backed executors into the given
/// registry.
///
/// `backend` is the backend adapter that will handle all inference
/// operations. `resolver` is the model resolution capability
/// consumed by `builtin.checkpoint_loader`.
///
/// Returns an error if a node type id is already registered (the
/// registry rejects duplicates).
pub fn register_builtin_inference_executors(
    registry: &mut NodeExecutorRegistry,
    backend: Arc<dyn InferenceBackend>,
    resolver: Arc<dyn ModelResolver>,
) -> Result<(), reimagine_runtime::NodeExecutorRegistryError> {
    use reimagine_nodes::*;

    // builtin.string — pure passthrough, no backend call
    registry.register(BUILTIN_STRING, Arc::new(StringExecutor))?;

    // builtin.checkpoint_loader — model.load_bundle
    registry.register(
        BUILTIN_CHECKPOINT_LOADER,
        Arc::new(CheckpointLoaderExecutor::new(
            Arc::clone(&backend),
            Arc::clone(&resolver),
        )),
    )?;

    // builtin.clip_text_encode — text.encode
    registry.register(
        BUILTIN_CLIP_TEXT_ENCODE,
        Arc::new(ClipTextEncodeExecutor::new(Arc::clone(&backend))),
    )?;

    // builtin.empty_latent_image — latent.create_empty
    registry.register(
        BUILTIN_EMPTY_LATENT_IMAGE,
        Arc::new(EmptyLatentImageExecutor::new(Arc::clone(&backend))),
    )?;

    // builtin.ksampler — diffusion.sample
    registry.register(
        BUILTIN_KSAMPLER,
        Arc::new(KSamplerExecutor::new(Arc::clone(&backend))),
    )?;

    // builtin.vae_decode — latent.decode
    registry.register(
        BUILTIN_VAE_DECODE,
        Arc::new(VaeDecodeExecutor::new(Arc::clone(&backend))),
    )?;

    // builtin.save_image — image.save
    registry.register(
        BUILTIN_SAVE_IMAGE,
        Arc::new(SaveImageExecutor::new(Arc::clone(&backend))),
    )?;

    // builtin.preview_image — image.preview
    registry.register(
        BUILTIN_PREVIEW_IMAGE,
        Arc::new(PreviewImageExecutor::new(Arc::clone(&backend))),
    )?;

    Ok(())
}
