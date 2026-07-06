//! CFG denoise loop for SDXL diffusion sampling over active Burn tensors.

use burn_tensor::{Tensor, TensorData};

use crate::active_backend::ActiveBurnBackend;
use crate::error::BurnBackendError;

use super::module::{SdxlAddedConditioning, SdxlUnet};
use super::scheduler::EulerNormalScheduler;

/// Per-branch conditioning consumed by one CFG UNet forward.
#[derive(Debug, Clone)]
pub struct SdxlCfgConditioning {
    text: Tensor<ActiveBurnBackend, 3>,
    added: SdxlAddedConditioning<ActiveBurnBackend>,
}

impl SdxlCfgConditioning {
    pub fn new(
        text: Tensor<ActiveBurnBackend, 3>,
        added: SdxlAddedConditioning<ActiveBurnBackend>,
    ) -> Self {
        Self { text, added }
    }
}

/// Run the euler/normal classifier-free guidance loop over the Burn-native
/// UNet Module graph.
pub fn euler_normal_cfg_sample(
    unet: &SdxlUnet<ActiveBurnBackend>,
    latent: Tensor<ActiveBurnBackend, 4>,
    positive_conditioning: SdxlCfgConditioning,
    negative_conditioning: SdxlCfgConditioning,
    steps: u32,
    cfg: f32,
    seed: u64,
) -> Result<Tensor<ActiveBurnBackend, 4>, BurnBackendError> {
    let scheduler = EulerNormalScheduler::new(steps)?;
    let device = latent.device();
    let dims = latent.shape().dims();
    let mut latent = latent + seeded_noise(dims, seed, &device) * scheduler.init_noise_sigma as f32;

    for step in 0..steps as usize {
        let timestep = Tensor::<ActiveBurnBackend, 1>::from_data(
            TensorData::new(vec![scheduler.timesteps[step] as f32], [1]),
            &device,
        );
        let noise_uncond = unet.forward_with_added_conditioning(
            latent.clone(),
            timestep.clone(),
            negative_conditioning.text.clone(),
            negative_conditioning.added.clone(),
        );
        let noise_text = unet.forward_with_added_conditioning(
            latent.clone(),
            timestep,
            positive_conditioning.text.clone(),
            positive_conditioning.added.clone(),
        );
        let guided = noise_uncond.clone() + (noise_text - noise_uncond) * cfg;
        latent = scheduler.step_tensor(latent, guided, step)?;
    }

    Ok(latent)
}

fn seeded_noise(
    shape: [usize; 4],
    seed: u64,
    device: &burn_tensor::Device<ActiveBurnBackend>,
) -> Tensor<ActiveBurnBackend, 4> {
    let total = shape.iter().product::<usize>();
    let mut data = Vec::with_capacity(total);
    let mut state = seed.wrapping_add(1);
    for _ in 0..total {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (state >> 32) as f32 / u32::MAX as f32 * 2.0 - 1.0;
        data.push(val * 0.1);
    }
    Tensor::from_data(TensorData::new(data, shape), device)
}

#[cfg(test)]
mod tests {
    use burn_tensor::Tensor;

    use super::*;
    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use crate::models::stable_diffusion::sdxl::diffusion::module::{
        SdxlAddedConditioning, SdxlUnet,
    };

    #[test]
    fn cfg_sampler_runs_over_active_unet_module_and_preserves_shape() {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let unet = SdxlUnet::<ActiveBurnBackend>::init(&device);
        let latent = Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], &device);
        let positive = Tensor::<ActiveBurnBackend, 3>::ones([1, 77, 16], &device);
        let negative = Tensor::<ActiveBurnBackend, 3>::zeros([1, 77, 16], &device);

        let sampled = euler_normal_cfg_sample(
            &unet,
            latent,
            SdxlCfgConditioning::new(
                positive,
                SdxlAddedConditioning::new(
                    Tensor::<ActiveBurnBackend, 2>::ones([1, 8], &device),
                    Tensor::<ActiveBurnBackend, 2>::ones([1, 6], &device),
                ),
            ),
            SdxlCfgConditioning::new(
                negative,
                SdxlAddedConditioning::new(
                    Tensor::<ActiveBurnBackend, 2>::zeros([1, 8], &device),
                    Tensor::<ActiveBurnBackend, 2>::zeros([1, 6], &device),
                ),
            ),
            1,
            7.5,
            42,
        )
        .expect("active cfg sample");

        assert_eq!(sampled.shape().dims(), [1, 4, 8, 8]);
    }
}
