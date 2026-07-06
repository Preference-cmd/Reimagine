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
    let profile = unet_topology_profile_for_bundle(bundle);
    if !profile.is_module_graph_supported() {
        return Err(BurnBackendError::InvalidRequest(format!(
            "UNet topology `{}` is not yet supported by the current Module graph; use a tiny fixture profile or complete the 15d full-topology follow-ups",
            profile.as_str()
        )));
    }

    let mut unet =
        module::SdxlUnet::<ActiveBurnBackend>::init_from_profile(profile, runtime.device());
    if let Some(component) = bundle
        .components()
        .iter()
        .find(|component| component.component_role == BurnSdxlComponentRole::Diffusion)
    {
        loading::load_unet_module_from_path_with_profile(
            runtime,
            &mut unet,
            &component.source_path,
            profile,
        )?;
    }
    Ok(unet)
}

fn unet_topology_profile_for_bundle(
    bundle: &BurnLoadedSdxlBundle,
) -> module::SdxlUnetTopologyProfile {
    if bundle.components().is_empty() || bundle.uses_tiny_sdxl_e2e_diffusion_profile() {
        return module::SdxlUnetTopologyProfile::TinySdxlE2e;
    }

    module::SdxlUnetTopologyProfile::SdxlBase
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use reimagine_core::model::ModelId;
    use reimagine_inference::BackendPayloadKey;

    use super::*;
    use crate::config::BurnBackendConfig;
    use crate::models::stable_diffusion::sdxl::BurnSdxlComponentRole;

    #[test]
    fn load_or_init_unet_rejects_full_sdxl_topology_until_graph_supported() {
        let backend =
            BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("burn backend");
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("sdxl-base"),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![(
            BurnSdxlComponentRole::Diffusion,
            PathBuf::from("/tmp/full-sdxl-diffusion.safetensors"),
        )]);

        let err = load_or_init_unet(&bundle, &backend)
            .expect_err("full SDXL topology should be explicitly rejected");
        let message = err.to_string();

        assert!(message.contains("UNet topology `sdxl_base`"), "{message}");
        assert!(
            message.contains("not yet supported by the current Module graph"),
            "{message}"
        );
    }

    #[test]
    fn unet_topology_profile_is_explicit_for_test_and_component_bundles() {
        let synthetic = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("synthetic-sdxl"),
            BackendPayloadKey::new("clip"),
        );
        assert_eq!(
            unet_topology_profile_for_bundle(&synthetic),
            module::SdxlUnetTopologyProfile::TinySdxlE2e
        );

        let tiny = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("tiny-sdxl"),
            BackendPayloadKey::new("clip"),
        )
        .with_test_tiny_fixture_components(vec![(
            BurnSdxlComponentRole::Diffusion,
            PathBuf::from("/tmp/tiny-sdxl-diffusion.safetensors"),
        )]);
        assert_eq!(
            unet_topology_profile_for_bundle(&tiny),
            module::SdxlUnetTopologyProfile::TinySdxlE2e
        );

        let full = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("sdxl-base"),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![(
            BurnSdxlComponentRole::Diffusion,
            PathBuf::from("/tmp/full-sdxl-diffusion.safetensors"),
        )]);
        assert_eq!(
            unet_topology_profile_for_bundle(&full),
            module::SdxlUnetTopologyProfile::SdxlBase
        );
    }
}
