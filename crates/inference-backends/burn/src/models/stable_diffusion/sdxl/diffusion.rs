//! SDXL diffusion sampling — routine definitions and module-level
//! re-exports for the SDXL diffusion path.

mod loading;
pub mod module;
mod sampler;
pub(crate) mod scheduler;
pub mod unet;

use burn_tensor::Tensor;

use crate::active_backend::ActiveBurnBackend;
use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::{
    BurnLoadedModelBundle, BurnLoadedSdxlBundle, BurnSdxlComponentRole,
};
use crate::store::BurnConditioningPayload;

/// Run the SDXL euler/normal denoise loop.
///
/// Accepts a zero latent (EmptyGeometry), positive and negative
/// conditioning payloads, and returns a sampled latent tensor.
#[allow(clippy::too_many_arguments)]
pub fn sample_sdxl(
    bundle: &BurnLoadedModelBundle,
    latent: Tensor<ActiveBurnBackend, 4>,
    positive_cond: &BurnConditioningPayload,
    negative_cond: &BurnConditioningPayload,
    steps: u32,
    cfg: f32,
    seed: u64,
    backend: &BurnBackend,
) -> Result<Tensor<ActiveBurnBackend, 4>, BurnBackendError> {
    let sdxl = match bundle {
        BurnLoadedModelBundle::StableDiffusionSdxl(bundle) => bundle.as_ref(),
    };
    let unet = load_or_init_unet(sdxl, backend)?;
    let positive = project_conditioning_for_unet(positive_cond.active_text_embeddings()?, &unet)?;
    let negative = project_conditioning_for_unet(negative_cond.active_text_embeddings()?, &unet)?;
    sampler::euler_normal_cfg_sample(&unet, latent, positive, negative, steps, cfg, seed)
}

fn load_or_init_unet(
    bundle: &BurnLoadedSdxlBundle,
    backend: &BurnBackend,
) -> Result<module::SdxlUnet<ActiveBurnBackend>, BurnBackendError> {
    let runtime = backend.active_runtime();
    let mut unet = module::SdxlUnet::<ActiveBurnBackend>::init(runtime.device());
    if let Some(component) = bundle
        .components()
        .iter()
        .find(|component| component.component_role == BurnSdxlComponentRole::Diffusion)
    {
        loading::load_unet_module_from_path(runtime, &mut unet, &component.source_path)?;
    }
    Ok(unet)
}

fn project_conditioning_for_unet(
    conditioning: Tensor<ActiveBurnBackend, 3>,
    unet: &module::SdxlUnet<ActiveBurnBackend>,
) -> Result<Tensor<ActiveBurnBackend, 3>, BurnBackendError> {
    let Some(width) = unet.cross_attention_context_dim() else {
        return Ok(conditioning);
    };
    let [batch, seq, actual_width] = conditioning.shape().dims();
    if actual_width == width {
        return Ok(conditioning);
    }
    if actual_width < width {
        return Err(BurnBackendError::InvalidRequest(format!(
            "diffusion.sample conditioning width {actual_width} is smaller than UNet context width {width}"
        )));
    }
    Ok(conditioning.slice([0..batch, 0..seq, 0..width]))
}
