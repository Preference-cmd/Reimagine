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
use crate::store::{BurnImagePayload, BurnLatentPayload};

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

/// Run SDXL VAE encode on an input image.
///
/// Returns a latent tensor in NCHW F32 format at 1/8 spatial scale.
/// The input image is normalized from [0, 1] to [-1, 1] (the VAE
/// encoder input convention) before the forward pass.
pub fn encode_image(
    bundle: &BurnLoadedModelBundle,
    image: BurnImagePayload,
    backend: &BurnBackend,
) -> Result<Tensor<ActiveBurnBackend, 4>, BurnBackendError> {
    let image = image.into_active_tensor()?;
    let encoder = load_or_init_encoder(bundle, backend)?;
    let normalized = image * 2.0 - 1.0;
    Ok(encoder.forward(normalized))
}

fn load_or_init_encoder(
    bundle: &BurnLoadedModelBundle,
    backend: &BurnBackend,
) -> Result<module::SdxlVaeEncoder<ActiveBurnBackend>, BurnBackendError> {
    let sdxl = match bundle {
        BurnLoadedModelBundle::StableDiffusionSdxl(bundle) => bundle.as_ref(),
    };
    load_or_init_sdxl_encoder(sdxl, backend)
}

fn load_or_init_sdxl_encoder(
    bundle: &BurnLoadedSdxlBundle,
    backend: &BurnBackend,
) -> Result<module::SdxlVaeEncoder<ActiveBurnBackend>, BurnBackendError> {
    let runtime = backend.active_runtime();
    let mut encoder = module::SdxlVaeEncoder::<ActiveBurnBackend>::init(runtime.device());
    if let Some(component) = bundle
        .components()
        .iter()
        .find(|component| component.component_role == BurnSdxlComponentRole::Vae)
    {
        let profile = if bundle.uses_tiny_sdxl_e2e_vae_profile() {
            loading::SdxlVaeEncoderLoadProfile::TinySdxlE2e
        } else {
            loading::SdxlVaeEncoderLoadProfile::SdxlBase
        };
        loading::load_vae_encoder_module_from_path_with_profile(
            runtime,
            &mut encoder,
            &component.source_path,
            profile,
        )?;
    }
    Ok(encoder)
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
        let profile = if bundle.uses_tiny_sdxl_e2e_vae_profile() {
            loading::SdxlVaeDecoderLoadProfile::TinySdxlE2e
        } else {
            loading::SdxlVaeDecoderLoadProfile::SdxlBase
        };
        loading::load_vae_decoder_module_from_path_with_profile(
            runtime,
            &mut decoder,
            &component.source_path,
            profile,
        )?;
    }
    Ok(decoder)
}
