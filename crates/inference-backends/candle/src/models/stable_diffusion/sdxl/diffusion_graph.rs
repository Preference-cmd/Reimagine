use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
};

use super::checkpoint_inventory::{SdxlCheckpointInventory, SdxlCheckpointInventoryError};
use super::checkpoint_projection::{
    SdxlCheckpointRole, SdxlCheckpointRoleProjection, project_checkpoint_role,
};
use super::diffusion::{SdxlSampleRequest, seeded_noise_like};
use super::diffusion_sources::SdxlDiffusionSources;
use crate::error::CandleBackendError;
use crate::store::CandleLatent;

#[derive(Debug, Clone)]
pub(crate) struct SdxlUnetForwardConditioning {
    pub text_embedding: Tensor,
}

pub(crate) trait SdxlDiffusionGraph: std::fmt::Debug + Send + Sync {
    fn sample(
        &self,
        input_latent: CandleLatent,
        positive: SdxlUnetForwardConditioning,
        negative: SdxlUnetForwardConditioning,
        request: &SdxlSampleRequest,
        device: &Device,
    ) -> Result<CandleLatent, CandleBackendError>;
}

pub(crate) fn load_diffusion_graph(
    sources: &SdxlDiffusionSources,
    device: &Device,
) -> Result<Arc<dyn SdxlDiffusionGraph>, CandleBackendError> {
    let inventory = SdxlCheckpointInventory::from_path(sources.path()).map_err(|err| {
        let path = sources.path().display();
        let fingerprint = sources.fingerprint();
        let message = match err {
            SdxlCheckpointInventoryError::InvalidHeader { reason, .. } => format!(
                "diffusion.sample invalid SDXL diffusion safetensors header at `{path}` ({fingerprint}): {reason}"
            ),
            other => format!(
                "diffusion.sample failed to inspect SDXL checkpoint inventory at `{path}` ({fingerprint}): {other}"
            ),
        };
        CandleBackendError::InvalidRequest(message)
    })?;
    let projection =
        project_checkpoint_role(sources.path(), SdxlCheckpointRole::Diffusion, &inventory)
            .map_err(|err| {
                CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample {} ({})",
                    err,
                    sources.fingerprint()
                ))
            })?;
    match projection {
        SdxlCheckpointRoleProjection::OriginalCheckpoint {
            recognized_families,
        } => Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample SDXL original checkpoint diffusion layout is detected at `{}` ({}); recognized families: {}; `model.diffusion_model.*` to Candle UNet key mapping is not implemented yet",
            sources.path().display(),
            sources.fingerprint(),
            recognized_families
                .iter()
                .map(|family| family.prefix())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
        SdxlCheckpointRoleProjection::DiffusersUnet => {
            let graph = DiffusersSdxlUnetGraph::load(sources.path(), device)?;
            Ok(Arc::new(graph))
        }
    }
}

#[derive(Debug)]
struct DiffusersSdxlUnetGraph {
    unet: UNet2DConditionModel,
}

impl DiffusersSdxlUnetGraph {
    fn load(path: &Path, device: &Device) -> Result<Self, CandleBackendError> {
        let data = std::fs::read(path).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample failed to read SDXL diffusion weights `{}`: {err}",
                path.display()
            ))
        })?;
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(data, DType::F32, device)
            .map_err(|err| {
                CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample failed to parse SDXL diffusion safetensors `{}`: {err}",
                    path.display()
                ))
            })?;
        let vb = if vb.contains_tensor("unet.conv_in.weight") {
            vb.pp("unet")
        } else {
            vb
        };
        let unet =
            UNet2DConditionModel::new(vb, 4, 4, false, sdxl_base_unet_config()).map_err(|err| {
                CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample failed to materialize SDXL UNet graph from `{}`: {err}",
                    path.display()
                ))
            })?;
        Ok(Self { unet })
    }
}

trait SdxlDenoiser {
    fn predict_noise(
        &self,
        latent_model_input: &Tensor,
        timestep: f64,
        conditioning: &SdxlUnetForwardConditioning,
    ) -> candle_core::Result<Tensor>;
}

impl SdxlDenoiser for UNet2DConditionModel {
    fn predict_noise(
        &self,
        latent_model_input: &Tensor,
        timestep: f64,
        conditioning: &SdxlUnetForwardConditioning,
    ) -> candle_core::Result<Tensor> {
        self.forward(latent_model_input, timestep, &conditioning.text_embedding)
    }
}

impl SdxlDiffusionGraph for DiffusersSdxlUnetGraph {
    fn sample(
        &self,
        input_latent: CandleLatent,
        positive: SdxlUnetForwardConditioning,
        negative: SdxlUnetForwardConditioning,
        request: &SdxlSampleRequest,
        device: &Device,
    ) -> Result<CandleLatent, CandleBackendError> {
        run_euler_normal_denoise_loop(
            &self.unet,
            input_latent,
            positive,
            negative,
            request,
            device,
        )
    }
}

fn run_euler_normal_denoise_loop(
    denoiser: &dyn SdxlDenoiser,
    input_latent: CandleLatent,
    positive: SdxlUnetForwardConditioning,
    negative: SdxlUnetForwardConditioning,
    request: &SdxlSampleRequest,
    device: &Device,
) -> Result<CandleLatent, CandleBackendError> {
    let latent_space = input_latent.latent_space().clone();
    let input = input_latent.into_tensor();
    if input.dtype() != DType::F32 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample expects f32 input latent geometry, got {:?}",
            input.dtype()
        )));
    }
    let shape = input.shape().dims().to_vec();
    let noise = seeded_noise_like(request.seed, &shape, device)?;
    let scheduler = EulerNormalScheduler::new(request.steps as usize)?;
    let is_full_denoise = request.denoise >= 1.0;
    let start_index = if is_full_denoise {
        0
    } else {
        scheduler.start_index_for_denoise(request.denoise)?
    };
    let mut sample = if is_full_denoise {
        (noise * scheduler.init_noise_sigma()).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample failed to scale seeded noise for SDXL euler/normal: {err}"
            ))
        })?
    } else {
        let sigma = scheduler.sigma_at(start_index)?;
        let scaled_noise = noise.affine(sigma, 0.0).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample failed to scale partial denoise seeded noise at scheduler step {start_index}: {err}"
            ))
        })?;
        (&input + &scaled_noise).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample failed to add partial denoise noise at scheduler step {start_index}: {err}"
            ))
        })?
    };

    for step in scheduler.steps_from(start_index) {
        let latent_model_input = scheduler.scale_model_input(sample.clone(), step.index)?;
        let negative_pred = denoiser
            .predict_noise(&latent_model_input, step.timestep, &negative)
            .map_err(|err| {
                CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample negative UNet prediction failed at timestep {}: {err}",
                    step.timestep
                ))
            })?;
        let positive_pred = denoiser
            .predict_noise(&latent_model_input, step.timestep, &positive)
            .map_err(|err| {
                CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample positive UNet prediction failed at timestep {}: {err}",
                    step.timestep
                ))
            })?;
        let guidance_delta = (&positive_pred - &negative_pred).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample CFG delta failed at timestep {}: {err}",
                step.timestep
            ))
        })?;
        let guided = (&negative_pred + guidance_delta * request.cfg).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample CFG combine failed at timestep {}: {err}",
                step.timestep
            ))
        })?;
        sample = scheduler.step(&guided, step.index, &sample)?;
    }
    Ok(CandleLatent::new(sample, latent_space))
}

#[derive(Debug)]
struct EulerNormalScheduler {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
}

#[derive(Debug, Clone, Copy)]
struct EulerNormalStep {
    index: usize,
    timestep: f64,
}

impl EulerNormalScheduler {
    const BETA_START: f64 = 0.00085;
    const BETA_END: f64 = 0.012;
    const TRAIN_TIMESTEPS: usize = 1000;
    const STEPS_OFFSET: usize = 1;

    fn new(inference_steps: usize) -> Result<Self, CandleBackendError> {
        if inference_steps == 0 {
            return Err(CandleBackendError::InvalidRequest(
                "diffusion.sample euler/normal scheduler requires at least one step".to_string(),
            ));
        }
        let step_ratio = Self::TRAIN_TIMESTEPS / inference_steps;
        if step_ratio == 0 {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler supports at most {} steps, got {inference_steps}",
                Self::TRAIN_TIMESTEPS
            )));
        }

        let timesteps = (0..inference_steps)
            .map(|step| {
                (step * step_ratio + Self::STEPS_OFFSET).min(Self::TRAIN_TIMESTEPS - 1) as f64
            })
            .rev()
            .collect::<Vec<_>>();
        let training_sigmas = Self::training_sigmas();
        let training_indexes = (0..training_sigmas.len())
            .map(|index| index as f64)
            .collect::<Vec<_>>();
        let mut sigmas = interpolate(&timesteps, &training_indexes, &training_sigmas)?;
        sigmas.push(0.0);
        let init_noise_sigma = sigmas
            .iter()
            .copied()
            .fold(0.0f64, |max, sigma| max.max(sigma));

        Ok(Self {
            timesteps,
            sigmas,
            init_noise_sigma,
        })
    }

    fn training_sigmas() -> Vec<f64> {
        let mut alpha_cumprod = 1.0f64;
        (0..Self::TRAIN_TIMESTEPS)
            .map(|step| {
                let beta_root = if Self::TRAIN_TIMESTEPS == 1 {
                    Self::BETA_START.sqrt()
                } else {
                    let ratio = step as f64 / (Self::TRAIN_TIMESTEPS - 1) as f64;
                    Self::BETA_START.sqrt()
                        + ratio * (Self::BETA_END.sqrt() - Self::BETA_START.sqrt())
                };
                let beta = beta_root * beta_root;
                alpha_cumprod *= 1.0 - beta;
                ((1.0 - alpha_cumprod) / alpha_cumprod).sqrt()
            })
            .collect()
    }

    fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    fn steps(&self) -> impl Iterator<Item = EulerNormalStep> + '_ {
        self.timesteps
            .iter()
            .copied()
            .enumerate()
            .map(|(index, timestep)| EulerNormalStep { index, timestep })
    }

    fn steps_from(&self, start_index: usize) -> impl Iterator<Item = EulerNormalStep> + '_ {
        self.steps().skip(start_index)
    }

    fn start_index_for_denoise(&self, denoise: f64) -> Result<usize, CandleBackendError> {
        if !denoise.is_finite() || !(0.0..=1.0).contains(&denoise) {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample denoise must be a finite number within [0, 1], got {denoise}"
            )));
        }
        if denoise == 0.0 {
            return Err(CandleBackendError::InvalidRequest(
                "diffusion.sample denoise 0.0 is unsupported in V1 because it is a no-op/pass-through; save/preview/direct-source handling should bypass diffusion.sample in a future shortcut"
                    .to_string(),
            ));
        }
        let steps = self.timesteps.len();
        let raw_index = steps.saturating_sub((steps as f64 * denoise).floor() as usize);
        Ok(raw_index.clamp(0, steps.saturating_sub(1)))
    }

    fn sigma_at(&self, step_index: usize) -> Result<f64, CandleBackendError> {
        self.sigmas.get(step_index).copied().ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler step {step_index} is out of sigma bounds"
            ))
        })
    }

    fn scale_model_input(
        &self,
        sample: Tensor,
        step_index: usize,
    ) -> Result<Tensor, CandleBackendError> {
        let sigma = self.sigma_at(step_index)?;
        (sample / (sigma * sigma + 1.0).sqrt()).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler input scaling failed at step {step_index}: {err}"
            ))
        })
    }

    fn step(
        &self,
        model_output: &Tensor,
        step_index: usize,
        sample: &Tensor,
    ) -> Result<Tensor, CandleBackendError> {
        let sigma = self.sigmas.get(step_index).copied().ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler step {step_index} is out of sigma bounds"
            ))
        })?;
        let sigma_next = self.sigmas.get(step_index + 1).copied().ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler next step {} is out of sigma bounds",
                step_index + 1
            ))
        })?;
        let pred_original_sample = (sample - (model_output * sigma).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler prediction scaling failed at step {step_index}: {err}"
            ))
        })?)
        .map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler original-sample prediction failed at step {step_index}: {err}"
            ))
        })?;
        let derivative = ((sample - &pred_original_sample).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler derivative numerator failed at step {step_index}: {err}"
            ))
        })? / sigma)
            .map_err(|err| {
                CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample euler/normal scheduler derivative failed at step {step_index}: {err}"
                ))
            })?;
        (sample + derivative * (sigma_next - sigma)).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler step failed at step {step_index}: {err}"
            ))
        })
    }
}

fn interpolate(x: &[f64], xp: &[f64], fp: &[f64]) -> Result<Vec<f64>, CandleBackendError> {
    if xp.len() != fp.len() || xp.len() < 2 {
        return Err(CandleBackendError::InvalidRequest(
            "diffusion.sample euler/normal scheduler interpolation table is invalid".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(x.len());
    for value in x {
        if *value < xp[0] || *value > xp[xp.len() - 1] {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler timestep {value} is out of training sigma bounds"
            )));
        }
        let upper = xp.partition_point(|candidate| *candidate <= *value);
        let lower = upper.saturating_sub(1).min(xp.len() - 2);
        let x0 = xp[lower];
        let x1 = xp[lower + 1];
        let y0 = fp[lower];
        let y1 = fp[lower + 1];
        let ratio = if x1 == x0 {
            0.0
        } else {
            (*value - x0) / (x1 - x0)
        };
        out.push(y0 + ratio * (y1 - y0));
    }
    Ok(out)
}

fn sdxl_base_unet_config() -> UNet2DConditionModelConfig {
    let block = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
        out_channels,
        use_cross_attn,
        attention_head_dim,
    };
    UNet2DConditionModelConfig {
        blocks: vec![
            block(320, None, 5),
            block(640, Some(2), 10),
            block(1280, Some(10), 20),
        ],
        center_input_sample: false,
        cross_attention_dim: 2048,
        downsample_padding: 1,
        flip_sin_to_cos: true,
        freq_shift: 0.,
        layers_per_block: 2,
        mid_block_scale_factor: 1.,
        norm_eps: 1e-5,
        norm_num_groups: 32,
        sliced_attention_size: None,
        use_linear_projection: true,
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct TestSdxlDiffusionGraph;

#[cfg(test)]
impl SdxlDiffusionGraph for TestSdxlDiffusionGraph {
    fn sample(
        &self,
        input_latent: CandleLatent,
        positive: SdxlUnetForwardConditioning,
        negative: SdxlUnetForwardConditioning,
        request: &SdxlSampleRequest,
        device: &Device,
    ) -> Result<CandleLatent, CandleBackendError> {
        let _conditioning_dims = (
            positive.text_embedding.shape().dims(),
            negative.text_embedding.shape().dims(),
        );
        super::diffusion::SdxlSampler::new()
            .sample(input_latent, request, device)
            .map(|result| result.latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};
    use reimagine_inference::LatentSpaceMetadata;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct FakeDenoiser {
        calls: Mutex<Vec<(f64, f32, Vec<usize>)>>,
    }

    impl FakeDenoiser {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<(f64, f32, Vec<usize>)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl SdxlDenoiser for FakeDenoiser {
        fn predict_noise(
            &self,
            latent_model_input: &Tensor,
            timestep: f64,
            conditioning: &SdxlUnetForwardConditioning,
        ) -> candle_core::Result<Tensor> {
            let value = conditioning
                .text_embedding
                .flatten_all()?
                .to_vec1::<f32>()?
                .first()
                .copied()
                .unwrap_or(0.0);
            self.calls.lock().unwrap().push((
                timestep,
                value,
                latent_model_input.shape().dims().to_vec(),
            ));
            Tensor::full(
                value,
                latent_model_input.shape(),
                latent_model_input.device(),
            )
        }
    }

    fn conditioning(value: f32) -> SdxlUnetForwardConditioning {
        SdxlUnetForwardConditioning {
            text_embedding: Tensor::full(value, (1, 77, 2048), &Device::Cpu).unwrap(),
        }
    }

    fn request(seed: u64, steps: u32, cfg: f64) -> SdxlSampleRequest {
        request_with_denoise(seed, steps, cfg, 1.0)
    }

    fn request_with_denoise(seed: u64, steps: u32, cfg: f64, denoise: f64) -> SdxlSampleRequest {
        SdxlSampleRequest {
            seed,
            steps,
            cfg,
            sampler_name: "euler".to_string(),
            scheduler_name: "normal".to_string(),
            denoise,
        }
    }

    fn tensor_data(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn sample_with_fake(
        seed: u64,
        steps: u32,
        cfg: f64,
        positive_value: f32,
        negative_value: f32,
    ) -> Tensor {
        let denoiser = FakeDenoiser::new();
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        run_euler_normal_denoise_loop(
            &denoiser,
            latent,
            conditioning(positive_value),
            conditioning(negative_value),
            &request(seed, steps, cfg),
            &Device::Cpu,
        )
        .unwrap()
        .into_tensor()
    }

    #[test]
    fn euler_normal_loop_uses_seeded_noise_not_zero_latent() {
        let denoiser = FakeDenoiser::new();
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );

        let output = run_euler_normal_denoise_loop(
            &denoiser,
            latent,
            conditioning(0.0),
            conditioning(0.0),
            &request(7, 1, 1.0),
            &Device::Cpu,
        )
        .unwrap()
        .into_tensor();

        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            values.iter().any(|v| *v != 0.0),
            "zero denoiser should leave seeded initial noise observable"
        );
        assert_eq!(output.shape().dims(), &[1, 4, 4, 4]);
    }

    #[test]
    fn euler_normal_loop_is_deterministic_for_same_seed_and_changes_for_different_seed() {
        let first = sample_with_fake(42, 2, 1.0, 0.0, 0.0)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let second = sample_with_fake(42, 2, 1.0, 0.0, 0.0)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let different = sample_with_fake(43, 2, 1.0, 0.0, 0.0)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, different);
    }

    #[test]
    fn euler_normal_loop_applies_classifier_free_guidance() {
        let without_guidance = sample_with_fake(13, 2, 0.0, 5.0, 1.0)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let with_guidance = sample_with_fake(13, 2, 2.0, 5.0, 1.0)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_ne!(without_guidance, with_guidance);
    }

    #[test]
    fn euler_normal_loop_calls_positive_and_negative_denoiser_once_per_step() {
        let denoiser = FakeDenoiser::new();
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );

        let output = run_euler_normal_denoise_loop(
            &denoiser,
            latent,
            conditioning(3.0),
            conditioning(1.0),
            &request(9, 3, 1.5),
            &Device::Cpu,
        )
        .unwrap()
        .into_tensor();

        assert_eq!(output.shape().dims(), &[1, 4, 4, 4]);
        let calls = denoiser.calls();
        assert_eq!(calls.len(), 6);
        assert_eq!(
            calls.iter().map(|(_, v, _)| *v).collect::<Vec<_>>(),
            vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0]
        );
        assert!(calls.chunks_exact(2).all(|pair| pair[0].0 == pair[1].0));
        assert!(calls.iter().all(|(_, _, dims)| dims == &[1, 4, 4, 4]));
    }

    #[test]
    fn euler_normal_scheduler_derives_partial_denoise_start_index() {
        let scheduler = EulerNormalScheduler::new(20).unwrap();
        assert_eq!(scheduler.start_index_for_denoise(1.0).unwrap(), 0);
        assert_eq!(scheduler.start_index_for_denoise(0.5).unwrap(), 10);
        assert_eq!(scheduler.start_index_for_denoise(0.25).unwrap(), 15);
        assert_eq!(scheduler.start_index_for_denoise(0.05).unwrap(), 19);
    }

    #[test]
    fn euler_normal_scheduler_steps_from_preserves_original_step_indexes() {
        let scheduler = EulerNormalScheduler::new(20).unwrap();
        let suffix = scheduler.steps_from(10).collect::<Vec<_>>();
        assert_eq!(suffix.len(), 10);
        assert_eq!(suffix.first().unwrap().index, 10);
        assert_eq!(suffix.last().unwrap().index, 19);
        assert_eq!(
            suffix[0].timestep,
            scheduler.steps().nth(10).unwrap().timestep
        );
    }

    #[test]
    fn euler_normal_partial_denoise_initializes_from_input_latent_plus_start_sigma_noise() {
        let denoiser = FakeDenoiser::new();
        let shape = vec![1usize, 4, 2, 2];
        let input_values = (0..16).map(|value| value as f32 * 0.25).collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values.clone(), shape.clone(), &Device::Cpu).unwrap();
        let latent = CandleLatent::new(input.clone(), LatentSpaceMetadata::sdxl_base());
        let request = request_with_denoise(99, 20, 1.0, 0.05);
        let scheduler = EulerNormalScheduler::new(20).unwrap();
        let start_index = scheduler.start_index_for_denoise(request.denoise).unwrap();
        let start_sigma = scheduler.sigma_at(start_index).unwrap();
        let scaled_noise = seeded_noise_like(request.seed, &shape, &Device::Cpu)
            .unwrap()
            .affine(start_sigma, 0.0)
            .unwrap();
        let expected = (&input + &scaled_noise).unwrap();

        let output = run_euler_normal_denoise_loop(
            &denoiser,
            latent,
            conditioning(0.0),
            conditioning(0.0),
            &request,
            &Device::Cpu,
        )
        .unwrap()
        .into_tensor();

        assert_eq!(tensor_data(&output), tensor_data(&expected));
        let calls = denoiser.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0].0,
            scheduler.steps().nth(start_index).unwrap().timestep
        );
        assert_eq!(calls[1].0, calls[0].0);
    }

    #[test]
    fn euler_normal_partial_denoise_runs_only_scheduler_suffix() {
        let denoiser = FakeDenoiser::new();
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        let request = request_with_denoise(9, 20, 1.5, 0.25);
        let scheduler = EulerNormalScheduler::new(20).unwrap();
        let start_index = scheduler.start_index_for_denoise(request.denoise).unwrap();

        let output = run_euler_normal_denoise_loop(
            &denoiser,
            latent,
            conditioning(3.0),
            conditioning(1.0),
            &request,
            &Device::Cpu,
        )
        .unwrap()
        .into_tensor();

        assert_eq!(output.shape().dims(), &[1, 4, 4, 4]);
        let calls = denoiser.calls();
        let expected_steps = request.steps as usize - start_index;
        assert_eq!(calls.len(), expected_steps * 2);
        let expected_timesteps = scheduler
            .steps_from(start_index)
            .flat_map(|step| [step.timestep, step.timestep])
            .collect::<Vec<_>>();
        assert_eq!(
            calls
                .iter()
                .map(|(timestep, _, _)| *timestep)
                .collect::<Vec<_>>(),
            expected_timesteps
        );
    }
}
