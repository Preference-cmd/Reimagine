use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::unet_2d::{
    BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
};

use super::diffusion::{SdxlSampleRequest, seeded_noise_like};
use super::diffusion_sources::SdxlDiffusionSources;
use super::diffusion_weights::{
    SdxlDiffusionWeightLayout, detect_diffusion_weight_layout_from_file,
};
use crate::error::CandleBackendError;
use crate::store::CandleLatent;

#[derive(Debug, Clone)]
pub(crate) struct SdxlDiffusionConditioning {
    pub text_embedding: Tensor,
    pub pooled_embedding: Tensor,
}

pub(crate) trait SdxlDiffusionGraph: std::fmt::Debug + Send + Sync {
    fn sample(
        &self,
        input_latent: CandleLatent,
        positive: SdxlDiffusionConditioning,
        negative: SdxlDiffusionConditioning,
        request: &SdxlSampleRequest,
        device: &Device,
    ) -> Result<CandleLatent, CandleBackendError>;
}

pub(crate) fn load_diffusion_graph(
    sources: &SdxlDiffusionSources,
    device: &Device,
) -> Result<Arc<dyn SdxlDiffusionGraph>, CandleBackendError> {
    let layout = detect_diffusion_weight_layout_from_file(sources.path())?;
    match layout {
        SdxlDiffusionWeightLayout::OriginalCheckpoint => {
            Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample SDXL original checkpoint diffusion layout is detected at `{}` ({}), but the original `model.diffusion_model.*` to Candle UNet key adapter is not implemented yet",
                sources.path().display(),
                sources.fingerprint()
            )))
        }
        SdxlDiffusionWeightLayout::DiffusersUnet => {
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
        conditioning: &SdxlDiffusionConditioning,
    ) -> candle_core::Result<Tensor>;
}

impl SdxlDenoiser for UNet2DConditionModel {
    fn predict_noise(
        &self,
        latent_model_input: &Tensor,
        timestep: f64,
        conditioning: &SdxlDiffusionConditioning,
    ) -> candle_core::Result<Tensor> {
        self.forward(latent_model_input, timestep, &conditioning.text_embedding)
    }
}

impl SdxlDiffusionGraph for DiffusersSdxlUnetGraph {
    fn sample(
        &self,
        input_latent: CandleLatent,
        positive: SdxlDiffusionConditioning,
        negative: SdxlDiffusionConditioning,
        request: &SdxlSampleRequest,
        device: &Device,
    ) -> Result<CandleLatent, CandleBackendError> {
        let _pooled_dims = (
            positive.pooled_embedding.shape().dims(),
            negative.pooled_embedding.shape().dims(),
        );
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
    positive: SdxlDiffusionConditioning,
    negative: SdxlDiffusionConditioning,
    request: &SdxlSampleRequest,
    device: &Device,
) -> Result<CandleLatent, CandleBackendError> {
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
    let mut sample = (noise * scheduler.init_noise_sigma()).map_err(|err| {
        CandleBackendError::InvalidRequest(format!(
            "diffusion.sample failed to scale seeded noise for SDXL euler/normal: {err}"
        ))
    })?;

    for step in scheduler.steps() {
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
    Ok(CandleLatent::new(sample))
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

    fn scale_model_input(
        &self,
        sample: Tensor,
        step_index: usize,
    ) -> Result<Tensor, CandleBackendError> {
        let sigma = self.sigmas.get(step_index).copied().ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample euler/normal scheduler step {step_index} is out of sigma bounds"
            ))
        })?;
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
        positive: SdxlDiffusionConditioning,
        negative: SdxlDiffusionConditioning,
        request: &SdxlSampleRequest,
        device: &Device,
    ) -> Result<CandleLatent, CandleBackendError> {
        let _conditioning_dims = (
            positive.text_embedding.shape().dims(),
            positive.pooled_embedding.shape().dims(),
            negative.text_embedding.shape().dims(),
            negative.pooled_embedding.shape().dims(),
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
            conditioning: &SdxlDiffusionConditioning,
        ) -> candle_core::Result<Tensor> {
            let value = conditioning
                .pooled_embedding
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

    fn conditioning(value: f32) -> SdxlDiffusionConditioning {
        SdxlDiffusionConditioning {
            text_embedding: Tensor::zeros((1, 77, 2048), DType::F32, &Device::Cpu).unwrap(),
            pooled_embedding: Tensor::full(value, (1, 1280), &Device::Cpu).unwrap(),
        }
    }

    fn request(seed: u64, steps: u32, cfg: f64) -> SdxlSampleRequest {
        SdxlSampleRequest {
            seed,
            steps,
            cfg,
            sampler_name: "euler".to_string(),
            scheduler_name: "normal".to_string(),
            denoise: 1.0,
        }
    }

    fn sample_with_fake(
        seed: u64,
        steps: u32,
        cfg: f64,
        positive_value: f32,
        negative_value: f32,
    ) -> Tensor {
        let denoiser = FakeDenoiser::new();
        let latent =
            CandleLatent::new(Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap());
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
        let latent =
            CandleLatent::new(Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap());

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
        let latent =
            CandleLatent::new(Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap());

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
}
