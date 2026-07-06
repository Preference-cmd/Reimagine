//! SDXL VAE decode — routine definitions and module-level re-exports.

mod loading;
mod module;

use burn_tensor::Tensor;

use crate::active_backend::ActiveBurnBackend;
use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::{
    BurnLoadedModelBundle, BurnLoadedSdxlBundle, BurnSdxlComponentRole,
};
use crate::store::BurnLatentPayload;

/// Run SDXL VAE decode on a sampled latent.
///
/// Returns a decoded image tensor in NCHW F32 format, normalized
/// to [0, 1] range.
pub fn decode_latent(
    bundle: &BurnLoadedModelBundle,
    latent: BurnLatentPayload,
    backend: &BurnBackend,
) -> Result<Tensor<ActiveBurnBackend, 4>, BurnBackendError> {
    let latent = latent.into_active_tensor()?;
    let decoder = load_or_init_decoder(bundle, backend)?;
    Ok(decoder.forward(latent))
}

fn load_or_init_decoder(
    bundle: &BurnLoadedModelBundle,
    backend: &BurnBackend,
) -> Result<module::SdxlVaeDecoder<ActiveBurnBackend>, BurnBackendError> {
    let sdxl = match bundle {
        BurnLoadedModelBundle::StableDiffusionSdxl(bundle) => bundle.as_ref(),
    };
    load_or_init_sdxl_decoder(sdxl, backend)
}

fn load_or_init_sdxl_decoder(
    bundle: &BurnLoadedSdxlBundle,
    backend: &BurnBackend,
) -> Result<module::SdxlVaeDecoder<ActiveBurnBackend>, BurnBackendError> {
    let runtime = backend.active_runtime();
    let mut decoder = module::SdxlVaeDecoder::<ActiveBurnBackend>::init(runtime.device());
    if let Some(component) = bundle
        .components()
        .iter()
        .find(|component| component.component_role == BurnSdxlComponentRole::Vae)
    {
        loading::load_vae_decoder_module_from_path(runtime, &mut decoder, &component.source_path)?;
    }
    Ok(decoder)
}
