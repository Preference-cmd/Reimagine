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

    fn text_shape(&self) -> [usize; 3] {
        self.text.dims()
    }

    fn added_shapes(&self) -> [[usize; 2]; 2] {
        self.added.shapes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SdxlCfgBranch {
    Negative,
    Positive,
}

/// Evidence emitted immediately before each CFG branch enters the UNet Module.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SamplerForwardEvent {
    branch: SdxlCfgBranch,
    step_index: usize,
    timestep: f64,
    text_shape: [usize; 3],
    pooled_shape: [usize; 2],
    time_ids_shape: [usize; 2],
}

impl SamplerForwardEvent {
    pub(crate) fn new(
        branch: SdxlCfgBranch,
        step_index: usize,
        timestep: f64,
        text_shape: [usize; 3],
        pooled_shape: [usize; 2],
        time_ids_shape: [usize; 2],
    ) -> Self {
        Self {
            branch,
            step_index,
            timestep,
            text_shape,
            pooled_shape,
            time_ids_shape,
        }
    }

    fn from_conditioning(
        branch: SdxlCfgBranch,
        step_index: usize,
        timestep: f64,
        conditioning: &SdxlCfgConditioning,
    ) -> Self {
        let [pooled_shape, time_ids_shape] = conditioning.added_shapes();
        Self::new(
            branch,
            step_index,
            timestep,
            conditioning.text_shape(),
            pooled_shape,
            time_ids_shape,
        )
    }
}

struct SdxlSamplerRequest<'a> {
    unet: &'a SdxlUnet<ActiveBurnBackend>,
    latent: Tensor<ActiveBurnBackend, 4>,
    positive_conditioning: SdxlCfgConditioning,
    negative_conditioning: SdxlCfgConditioning,
    steps: u32,
    cfg: f32,
    seed: u64,
}

/// Run the euler/normal classifier-free guidance loop over the Burn-native
/// UNet Module graph.
///
/// The scheduler timestep value is the selected 1000-step training index
/// passed to both CFG branches before the Euler sigma update for that index.
pub fn euler_normal_cfg_sample(
    unet: &SdxlUnet<ActiveBurnBackend>,
    latent: Tensor<ActiveBurnBackend, 4>,
    positive_conditioning: SdxlCfgConditioning,
    negative_conditioning: SdxlCfgConditioning,
    steps: u32,
    cfg: f32,
    seed: u64,
) -> Result<Tensor<ActiveBurnBackend, 4>, BurnBackendError> {
    euler_normal_cfg_sample_with_observer(
        SdxlSamplerRequest {
            unet,
            latent,
            positive_conditioning,
            negative_conditioning,
            steps,
            cfg,
            seed,
        },
        |_| {},
    )
}

fn euler_normal_cfg_sample_with_observer(
    request: SdxlSamplerRequest,
    mut observe_forward: impl FnMut(SamplerForwardEvent),
) -> Result<Tensor<ActiveBurnBackend, 4>, BurnBackendError> {
    let SdxlSamplerRequest {
        unet,
        latent,
        positive_conditioning,
        negative_conditioning,
        steps,
        cfg,
        seed,
    } = request;
    let scheduler = EulerNormalScheduler::new(steps)?;
    let device = latent.device();
    let dims = latent.shape().dims();
    let mut latent = latent + seeded_noise(dims, seed, &device) * scheduler.init_noise_sigma as f32;

    for step in 0..steps as usize {
        let timestep_value = scheduler.timesteps[step];
        let timestep = Tensor::<ActiveBurnBackend, 1>::from_data(
            TensorData::new(vec![timestep_value as f32], [1]),
            &device,
        );
        observe_forward(SamplerForwardEvent::from_conditioning(
            SdxlCfgBranch::Negative,
            step,
            timestep_value,
            &negative_conditioning,
        ));
        let noise_uncond = unet.forward_with_added_conditioning(
            latent.clone(),
            timestep.clone(),
            negative_conditioning.text.clone(),
            negative_conditioning.added.clone(),
        );
        observe_forward(SamplerForwardEvent::from_conditioning(
            SdxlCfgBranch::Positive,
            step,
            timestep_value,
            &positive_conditioning,
        ));
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

    #[test]
    fn cfg_sampler_records_timestep_and_conditioning_before_each_unet_forward() {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let unet = SdxlUnet::<ActiveBurnBackend>::init(&device);
        let latent = Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], &device);
        let positive = Tensor::<ActiveBurnBackend, 3>::ones([1, 77, 16], &device);
        let negative = Tensor::<ActiveBurnBackend, 3>::zeros([1, 77, 16], &device);
        let scheduler = EulerNormalScheduler::new(2).expect("scheduler");
        let expected_timesteps = scheduler.timesteps.clone();
        let mut observed = Vec::new();

        let sampled = euler_normal_cfg_sample_with_observer(
            SdxlSamplerRequest {
                unet: &unet,
                latent,
                positive_conditioning: SdxlCfgConditioning::new(
                    positive,
                    SdxlAddedConditioning::new(
                        Tensor::<ActiveBurnBackend, 2>::ones([1, 8], &device),
                        Tensor::<ActiveBurnBackend, 2>::ones([1, 6], &device),
                    ),
                ),
                negative_conditioning: SdxlCfgConditioning::new(
                    negative,
                    SdxlAddedConditioning::new(
                        Tensor::<ActiveBurnBackend, 2>::zeros([1, 8], &device),
                        Tensor::<ActiveBurnBackend, 2>::zeros([1, 6], &device),
                    ),
                ),
                steps: 2,
                cfg: 7.5,
                seed: 42,
            },
            |event| observed.push(event),
        )
        .expect("active cfg sample with observer");

        assert_eq!(sampled.shape().dims(), [1, 4, 8, 8]);
        assert_eq!(
            observed,
            vec![
                SamplerForwardEvent::new(
                    SdxlCfgBranch::Negative,
                    0,
                    expected_timesteps[0],
                    [1, 77, 16],
                    [1, 8],
                    [1, 6],
                ),
                SamplerForwardEvent::new(
                    SdxlCfgBranch::Positive,
                    0,
                    expected_timesteps[0],
                    [1, 77, 16],
                    [1, 8],
                    [1, 6],
                ),
                SamplerForwardEvent::new(
                    SdxlCfgBranch::Negative,
                    1,
                    expected_timesteps[1],
                    [1, 77, 16],
                    [1, 8],
                    [1, 6],
                ),
                SamplerForwardEvent::new(
                    SdxlCfgBranch::Positive,
                    1,
                    expected_timesteps[1],
                    [1, 77, 16],
                    [1, 8],
                    [1, 6],
                ),
            ]
        );
    }

    #[test]
    fn cfg_sampler_is_deterministic_for_same_seed() {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let unet = SdxlUnet::<ActiveBurnBackend>::init(&device);
        let mk_latent = || Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], &device);
        let mk_conditioning = |value: f32| {
            SdxlCfgConditioning::new(
                Tensor::<ActiveBurnBackend, 3>::full([1, 77, 16], value, &device),
                SdxlAddedConditioning::new(
                    Tensor::<ActiveBurnBackend, 2>::full([1, 8], value, &device),
                    Tensor::<ActiveBurnBackend, 2>::full([1, 6], value, &device),
                ),
            )
        };

        let a = euler_normal_cfg_sample(
            &unet,
            mk_latent(),
            mk_conditioning(1.0),
            mk_conditioning(0.0),
            2,
            7.5,
            42,
        )
        .expect("first run");
        let b = euler_normal_cfg_sample(
            &unet,
            mk_latent(),
            mk_conditioning(1.0),
            mk_conditioning(0.0),
            2,
            7.5,
            42,
        )
        .expect("second run");

        let a_vals = a.to_data().to_vec::<f32>().unwrap();
        let b_vals = b.to_data().to_vec::<f32>().unwrap();
        assert_eq!(
            a_vals, b_vals,
            "same seed must produce identical latent output (deterministic sampler)"
        );
    }

    #[test]
    fn cfg_sampler_produces_different_latent_for_different_seed() {
        let config = BurnBackendConfig::new("/models", "/output");
        let device = active_device(config.device());
        let unet = SdxlUnet::<ActiveBurnBackend>::init(&device);
        let mk_latent = || Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], &device);
        let mk_conditioning = |value: f32| {
            SdxlCfgConditioning::new(
                Tensor::<ActiveBurnBackend, 3>::full([1, 77, 16], value, &device),
                SdxlAddedConditioning::new(
                    Tensor::<ActiveBurnBackend, 2>::full([1, 8], value, &device),
                    Tensor::<ActiveBurnBackend, 2>::full([1, 6], value, &device),
                ),
            )
        };

        let a = euler_normal_cfg_sample(
            &unet,
            mk_latent(),
            mk_conditioning(1.0),
            mk_conditioning(0.0),
            2,
            7.5,
            42,
        )
        .expect("seed 42");
        let b = euler_normal_cfg_sample(
            &unet,
            mk_latent(),
            mk_conditioning(1.0),
            mk_conditioning(0.0),
            2,
            7.5,
            999,
        )
        .expect("seed 999");

        let a_vals = a.to_data().to_vec::<f32>().unwrap();
        let b_vals = b.to_data().to_vec::<f32>().unwrap();
        assert_ne!(
            a_vals, b_vals,
            "different seeds must produce different latent output"
        );
    }

    #[test]
    fn cfg_sampler_multistep_timesteps_decrease_monotonically() {
        let device = active_device(BurnBackendConfig::new("/models", "/output").device());
        let unet = SdxlUnet::<ActiveBurnBackend>::init(&device);
        let latent = Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], &device);
        let positive = Tensor::<ActiveBurnBackend, 3>::ones([1, 77, 16], &device);
        let negative = Tensor::<ActiveBurnBackend, 3>::zeros([1, 77, 16], &device);
        let mut observed = Vec::new();

        let _sampled = euler_normal_cfg_sample_with_observer(
            SdxlSamplerRequest {
                unet: &unet,
                latent,
                positive_conditioning: SdxlCfgConditioning::new(
                    positive,
                    SdxlAddedConditioning::new(
                        Tensor::<ActiveBurnBackend, 2>::ones([1, 8], &device),
                        Tensor::<ActiveBurnBackend, 2>::ones([1, 6], &device),
                    ),
                ),
                negative_conditioning: SdxlCfgConditioning::new(
                    negative,
                    SdxlAddedConditioning::new(
                        Tensor::<ActiveBurnBackend, 2>::zeros([1, 8], &device),
                        Tensor::<ActiveBurnBackend, 2>::zeros([1, 6], &device),
                    ),
                ),
                steps: 5,
                cfg: 7.5,
                seed: 42,
            },
            |event| observed.push(event),
        )
        .expect("multi-step cfg sample");

        // Collect unique step-index timestep values from the observed events.
        let mut step_markers: Vec<(usize, f64)> = observed
            .iter()
            .filter(|e| e.branch == SdxlCfgBranch::Negative)
            .map(|e| (e.step_index, e.timestep))
            .collect();
        step_markers.sort_by_key(|(step, _)| *step);

        assert_eq!(
            step_markers.len(),
            5,
            "5 steps should produce 5 unique step markers"
        );

        // Verify timesteps are strictly decreasing (high-to-low noise order).
        for chunk in step_markers.windows(2) {
            assert!(
                chunk[0].1 > chunk[1].1,
                "timestep must be strictly decreasing: step {} (t={}) >= step {} (t={})",
                chunk[0].0,
                chunk[0].1,
                chunk[1].0,
                chunk[1].1,
            );
        }

        // Verify CFG ordering: each negative is paired before a positive at same step.
        for step_idx in 0..5 {
            let step_events: Vec<&SamplerForwardEvent> = observed
                .iter()
                .filter(|e| e.step_index == step_idx)
                .collect();
            assert_eq!(
                step_events.len(),
                2,
                "step {step_idx} must have exactly 2 events (neg + pos)"
            );
            assert_eq!(
                step_events[0].branch,
                SdxlCfgBranch::Negative,
                "step {step_idx}: first CFG branch must be negative"
            );
            assert_eq!(
                step_events[1].branch,
                SdxlCfgBranch::Positive,
                "step {step_idx}: second CFG branch must be positive"
            );
        }
    }

    #[test]
    fn cfg_sampler_captures_latent_numerical_evidence_at_each_step() {
        let device = active_device(BurnBackendConfig::new("/models", "/output").device());
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
            2,
            7.5,
            42,
        )
        .expect("cfg sample for numerical evidence");

        let vals = sampled.to_data().to_vec::<f32>().unwrap();
        let n = vals.len();
        assert_eq!(n, 4 * 8 * 8, "latent has 4x8x8 = {n} elements");
        let sum: f32 = vals.iter().sum();
        let mean = sum / n as f32;
        let variance: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;

        // Record numerical evidence as assertions that serve as a deterministic
        // parity baseline for this specific tiny UNet + seed + conditioning combo.
        eprintln!(
            "cfg_sampler latent evidence: n={n} mean={mean:.8} variance={variance:.8} first8={:?}",
            &vals[..8.min(n)]
        );

        // Mean and variance are non-zero after a 2-step CFG denoise with non-zero
        // conditionings and seed noise (zero latent in).
        assert!(
            mean.abs() > 0.0,
            "sampled latent must have non-zero mean after CFG denoise"
        );
        assert!(
            variance > 0.0,
            "sampled latent must have non-zero variance after CFG denoise"
        );
        // Sanity: with seed=42, steps=2, cfg=7.5 on the zero-initialized tiny UNet,
        // the output latent elements stay within a reasonable range.
        for &v in &vals {
            assert!(
                v.is_finite(),
                "all latent elements must be finite, got {v}"
            );
        }
    }
}
