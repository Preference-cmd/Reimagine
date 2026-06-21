//! Executor registration for V1 built-in node types.
//!
//! [`register_builtin_inference_executors`] installs all inference-backed
//! executors into a [`NodeExecutorRegistry`]. It requires an inference
//! runtime/router and a resolver; the registry itself is typically
//! constructed by app-host and handed to the runtime.

use std::sync::Arc;

use crate::{InferenceRuntime, ModelResolver};
use reimagine_core::model::NodeTypeId;

use crate::executor::{NodeExecutorRegistry, NodeExecutorRegistryError};
use crate::executors::{
    diffusion::KSamplerExecutor, image::PreviewImageExecutor, image::SaveImageExecutor,
    image::VaeDecodeExecutor, latent::EmptyLatentImageExecutor, model::CheckpointLoaderExecutor,
    string::StringExecutor, text::ClipTextEncodeExecutor,
};

/// Register all V1 built-in inference-backed executors into the given
/// registry.
///
/// `inference` is the executor-facing runtime/router that will select
/// and validate concrete backend calls. `resolver` is the model
/// resolution capability consumed by `builtin.checkpoint_loader`.
///
/// Returns an error if a node type id is already registered (the
/// registry rejects duplicates).
pub fn register_builtin_inference_executors(
    registry: &mut NodeExecutorRegistry,
    inference: Arc<dyn InferenceRuntime>,
    resolver: Arc<dyn ModelResolver>,
) -> Result<(), NodeExecutorRegistryError> {
    // builtin.string — pure passthrough, no backend call
    registry.register(NodeTypeId::new("builtin.string"), Arc::new(StringExecutor))?;

    // builtin.checkpoint_loader — model.load_bundle
    registry.register(
        NodeTypeId::new("builtin.checkpoint_loader"),
        Arc::new(CheckpointLoaderExecutor::new(
            Arc::clone(&inference),
            Arc::clone(&resolver),
        )),
    )?;

    // builtin.clip_text_encode — text.encode
    registry.register(
        NodeTypeId::new("builtin.clip_text_encode"),
        Arc::new(ClipTextEncodeExecutor::new(Arc::clone(&inference))),
    )?;

    // builtin.empty_latent_image — latent.create_empty
    registry.register(
        NodeTypeId::new("builtin.empty_latent_image"),
        Arc::new(EmptyLatentImageExecutor::new(Arc::clone(&inference))),
    )?;

    // builtin.ksampler — diffusion.sample
    registry.register(
        NodeTypeId::new("builtin.ksampler"),
        Arc::new(KSamplerExecutor::new(Arc::clone(&inference))),
    )?;

    // builtin.vae_decode — latent.decode
    registry.register(
        NodeTypeId::new("builtin.vae_decode"),
        Arc::new(VaeDecodeExecutor::new(Arc::clone(&inference))),
    )?;

    // builtin.save_image — image.save
    registry.register(
        NodeTypeId::new("builtin.save_image"),
        Arc::new(SaveImageExecutor::new(Arc::clone(&inference))),
    )?;

    // builtin.preview_image — image.preview
    registry.register(
        NodeTypeId::new("builtin.preview_image"),
        Arc::new(PreviewImageExecutor::new(Arc::clone(&inference))),
    )?;

    Ok(())
}
