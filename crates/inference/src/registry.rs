//! Inference-backed node executor adapters.
//!
//! Each executor maps a built-in node type to a typed capability
//! method on [`InferenceBackend`](crate::InferenceBackend), builds
//! the corresponding typed request DTO, calls the backend, and maps
//! the typed response into the workflow node's slot-shaped outputs.
//!
//! These are *adapters*, not backend implementations. They contain no
//! backend-specific behavior.

use std::sync::Arc;

use crate::capability::InferenceCapability;
use crate::executors::image_import::{ImageSourceResolver, LoadImageExecutor};
use crate::{ModelResolver, RouterRef};
use reimagine_core::model::NodeTypeId;

use crate::executor::{NodeExecutorRegistry, NodeExecutorRegistryError};
use crate::executors::{
    diffusion::KSamplerExecutor, image::PreviewImageExecutor, image::SaveImageExecutor,
    image::VaeDecodeExecutor, latent::EmptyLatentImageExecutor, latent_encode::VaeEncodeExecutor,
    model::CheckpointLoaderExecutor, string::StringExecutor, text::ClipTextEncodeExecutor,
};

/// Register all V1 built-in inference-backed executors into the given
/// registry.
///
/// `inference` is the executor-facing runtime/router that will select
/// and validate concrete backend calls. `resolver` is the model
/// resolution capability consumed by `builtin.checkpoint_loader`.
/// `image_source_resolver` is the workspace-safe image source
/// resolver injected into `builtin.load_image`; the inference layer
/// never inspects raw paths.
///
/// Returns an error if a node type id is already registered (the
/// registry rejects duplicates).
pub fn register_builtin_inference_executors(
    registry: &mut NodeExecutorRegistry,
    inference: RouterRef,
    resolver: Arc<dyn ModelResolver>,
    image_source_resolver: Arc<dyn ImageSourceResolver>,
) -> Result<(), NodeExecutorRegistryError> {
    // builtin.string — pure passthrough, no backend call
    registry.register(NodeTypeId::new("builtin.string"), Arc::new(StringExecutor))?;

    // builtin.checkpoint_loader — model.load_bundle
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.checkpoint_loader"),
        Arc::new(CheckpointLoaderExecutor::new(
            Arc::clone(&inference),
            Arc::clone(&resolver),
        )),
        &[InferenceCapability::LoadBundle],
    )?;

    // builtin.load_image — image.import
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.load_image"),
        Arc::new(LoadImageExecutor::new(
            Arc::clone(&inference),
            Arc::clone(&image_source_resolver),
        )),
        &[InferenceCapability::ImageImport],
    )?;

    // builtin.clip_text_encode — text.encode
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.clip_text_encode"),
        Arc::new(ClipTextEncodeExecutor::new(Arc::clone(&inference))),
        &[InferenceCapability::TextEncode],
    )?;

    // builtin.empty_latent_image — latent.create_empty
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.empty_latent_image"),
        Arc::new(EmptyLatentImageExecutor::new(Arc::clone(&inference))),
        &[InferenceCapability::CreateEmptyLatent],
    )?;

    // builtin.vae_encode — latent.encode
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.vae_encode"),
        Arc::new(VaeEncodeExecutor::new(Arc::clone(&inference))),
        &[InferenceCapability::LatentEncode],
    )?;

    // builtin.ksampler — diffusion.sample
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.ksampler"),
        Arc::new(KSamplerExecutor::new(Arc::clone(&inference))),
        &[InferenceCapability::DiffusionSample],
    )?;

    // builtin.vae_decode — latent.decode
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.vae_decode"),
        Arc::new(VaeDecodeExecutor::new(Arc::clone(&inference))),
        &[InferenceCapability::LatentDecode],
    )?;

    // builtin.save_image — image.save
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.save_image"),
        Arc::new(SaveImageExecutor::new(Arc::clone(&inference))),
        &[InferenceCapability::ImageSave],
    )?;

    // builtin.preview_image — image.preview
    registry.register_with_capabilities(
        NodeTypeId::new("builtin.preview_image"),
        Arc::new(PreviewImageExecutor::new(Arc::clone(&inference))),
        &[InferenceCapability::ImagePreview],
    )?;

    Ok(())
}
