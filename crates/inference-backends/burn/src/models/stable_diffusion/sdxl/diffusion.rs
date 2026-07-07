//! SDXL diffusion sampling — routine definitions and module-level
//! re-exports for the SDXL diffusion path.

mod loading;
pub mod module;
mod sampler;
pub(crate) mod scheduler;
pub mod unet;

use burn_tensor::{Tensor, TensorData};
use reimagine_inference::ConditioningMetadata;

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
    positive_metadata: &ConditioningMetadata,
    negative_metadata: &ConditioningMetadata,
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
    let positive_added = build_added_conditioning_for_unet(
        positive_cond.active_pooled_embeddings()?,
        positive_metadata,
        &unet,
    )?;
    let negative_added = build_added_conditioning_for_unet(
        negative_cond.active_pooled_embeddings()?,
        negative_metadata,
        &unet,
    )?;
    sampler::euler_normal_cfg_sample(
        &unet,
        latent,
        sampler::SdxlCfgConditioning::new(positive, positive_added),
        sampler::SdxlCfgConditioning::new(negative, negative_added),
        steps,
        cfg,
        seed,
    )
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

fn build_added_conditioning_for_unet(
    pooled: Tensor<ActiveBurnBackend, 2>,
    metadata: &ConditioningMetadata,
    unet: &module::SdxlUnet<ActiveBurnBackend>,
) -> Result<module::SdxlAddedConditioning<ActiveBurnBackend>, BurnBackendError> {
    let [pooled_width, time_ids_width, _] = unet.added_conditioning_dims();
    let pooled = project_pooled_conditioning_for_unet(pooled, pooled_width)?;
    let time_ids =
        time_ids_for_conditioning(metadata, pooled.dims()[0], time_ids_width, &pooled.device())?;
    Ok(module::SdxlAddedConditioning::new(pooled, time_ids))
}

fn project_pooled_conditioning_for_unet(
    pooled: Tensor<ActiveBurnBackend, 2>,
    width: usize,
) -> Result<Tensor<ActiveBurnBackend, 2>, BurnBackendError> {
    let [batch, actual_width] = pooled.dims();
    if actual_width == width {
        return Ok(pooled);
    }
    if actual_width < width {
        return Err(BurnBackendError::InvalidRequest(format!(
            "diffusion.sample pooled conditioning width {actual_width} is smaller than UNet added-conditioning width {width}"
        )));
    }
    Ok(pooled.slice([0..batch, 0..width]))
}

fn time_ids_for_conditioning(
    metadata: &ConditioningMetadata,
    batch: usize,
    width: usize,
    device: &burn_tensor::Device<ActiveBurnBackend>,
) -> Result<Tensor<ActiveBurnBackend, 2>, BurnBackendError> {
    let values = [
        metadata.width() as f32,
        metadata.height() as f32,
        metadata.crop_x() as f32,
        metadata.crop_y() as f32,
        metadata.target_width() as f32,
        metadata.target_height() as f32,
    ];
    if width != values.len() {
        return Err(BurnBackendError::InvalidRequest(format!(
            "diffusion.sample SDXL time ids width is {}; expected {}",
            width,
            values.len()
        )));
    }
    let mut data = Vec::with_capacity(batch * width);
    for _ in 0..batch {
        data.extend_from_slice(&values);
    }
    Ok(Tensor::<ActiveBurnBackend, 2>::from_data(
        TensorData::new(data, [batch, width]),
        device,
    ))
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
    fn load_or_init_unet_initializes_full_sdxl_topology_after_graph_support() {
        let backend =
            BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("burn backend");
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("sdxl-base"),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![(
            BurnSdxlComponentRole::Vae,
            PathBuf::from("/tmp/full-sdxl-vae.safetensors"),
        )]);

        let unet = load_or_init_unet(&bundle, &backend)
            .expect("full SDXL topology should initialize once the Module graph is supported");

        assert_eq!(
            unet.topology_profile(),
            module::SdxlUnetTopologyProfile::SdxlBase
        );
        assert!(unet.down_path_pushes_skip());
        assert!(unet.up_path_pops_skip());
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
