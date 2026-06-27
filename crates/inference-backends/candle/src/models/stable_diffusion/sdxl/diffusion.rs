//! SDXL diffusion sampler.
//!
//! The production path lives in `diffusion_graph.rs`; this module owns
//! backend-neutral SDXL sample parameter validation plus test helpers.

#[cfg(test)]
use candle_core::DType;
use candle_core::{Device, Tensor};

use crate::error::CandleBackendError;
#[cfg(test)]
use crate::store::CandleLatent;
#[cfg(test)]
use reimagine_inference::LatentSpaceMetadata;

/// User-controllable sampling parameters extracted from the
/// backend-neutral request.
///
/// `sampler_name` and `scheduler_name` are kept as strings so the
/// generic operation layer can validate them without forcing the
/// backend to enumerate the scheduler matrix. The first supported
/// sampler is `euler`; the first supported scheduler is `normal`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SdxlSampleRequest {
    pub seed: u64,
    pub steps: u32,
    pub cfg: f64,
    pub sampler_name: String,
    pub scheduler_name: String,
    pub denoise: f64,
}

impl SdxlSampleRequest {
    /// Build a sampler request directly from typed capability fields.
    ///
    /// Used by the typed `diffusion.sample` capability call. Validates
    /// the same V1 constraints as [`SdxlSampleRequest::from_params`].
    pub fn new(
        seed: u64,
        steps: u32,
        cfg: f32,
        sampler_name: impl Into<String>,
        scheduler_name: impl Into<String>,
        denoise: f32,
    ) -> Result<Self, CandleBackendError> {
        let sampler_name = sampler_name.into().to_ascii_lowercase();
        let scheduler_name = scheduler_name.into().to_ascii_lowercase();
        let cfg = cfg as f64;
        let denoise = denoise as f64;

        if sampler_name != "euler" {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample sampler `{sampler_name}` is not supported in V1; expected `euler`"
            )));
        }
        if scheduler_name != "normal" {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample scheduler `{scheduler_name}` is not supported in V1; expected `normal`"
            )));
        }
        if !(0.0..=1.0).contains(&denoise) {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample denoise must be within [0, 1], got {denoise}"
            )));
        }
        if denoise < 1.0 {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample partial denoise/img2img is not supported in V1; expected denoise 1.0, got {denoise}"
            )));
        }
        if steps == 0 {
            return Err(CandleBackendError::InvalidRequest(
                "diffusion.sample steps must be positive".to_string(),
            ));
        }
        if !cfg.is_finite() {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample cfg must be a finite number, got {cfg}"
            )));
        }

        Ok(Self {
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler_name,
            denoise,
        })
    }

    /// Build a sampler request from typed node parameters, validating
    /// the values the backend supports in V1.
    #[allow(dead_code)]
    pub fn from_params(
        params: &std::collections::HashMap<
            reimagine_core::model::SlotId,
            reimagine_core::model::ParamValue,
        >,
    ) -> Result<Self, CandleBackendError> {
        let seed = extract_seed(params)?;
        let steps = extract_u32(params, "steps")?;
        let cfg = extract_f64(params, "cfg")?;
        let denoise = extract_f64(params, "denoise")?;
        let sampler_name = extract_select(params, "sampler")?.to_ascii_lowercase();
        let scheduler_name = extract_select(params, "scheduler")?.to_ascii_lowercase();

        if sampler_name != "euler" {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample sampler `{sampler_name}` is not supported in V1; expected `euler`"
            )));
        }
        if scheduler_name != "normal" {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample scheduler `{scheduler_name}` is not supported in V1; expected `normal`"
            )));
        }
        if !(0.0..=1.0).contains(&denoise) {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample denoise must be within [0, 1], got {denoise}"
            )));
        }
        if denoise < 1.0 {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample partial denoise/img2img is not supported in V1; expected denoise 1.0, got {denoise}"
            )));
        }
        if steps == 0 {
            return Err(CandleBackendError::InvalidRequest(
                "diffusion.sample steps must be positive".to_string(),
            ));
        }
        if !cfg.is_finite() {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample cfg must be a finite number, got {cfg}"
            )));
        }

        Ok(Self {
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler_name,
            denoise,
        })
    }
}

#[allow(dead_code)]
fn extract_seed(
    params: &std::collections::HashMap<
        reimagine_core::model::SlotId,
        reimagine_core::model::ParamValue,
    >,
) -> Result<u64, CandleBackendError> {
    let slot = reimagine_core::model::SlotId::new("seed");
    match params.get(&slot) {
        Some(reimagine_core::model::ParamValue::Seed(v)) => Ok(*v),
        Some(reimagine_core::model::ParamValue::Integer(v)) => {
            if *v < 0 {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample param `seed` must be non-negative, got {v}"
                )));
            }
            Ok(*v as u64)
        }
        Some(_) => Err(CandleBackendError::InvalidRequest(
            "diffusion.sample param `seed` must be a seed or integer".to_string(),
        )),
        None => Err(CandleBackendError::InvalidRequest(
            "diffusion.sample requires a `seed` param".to_string(),
        )),
    }
}

#[allow(dead_code)]
fn extract_u32(
    params: &std::collections::HashMap<
        reimagine_core::model::SlotId,
        reimagine_core::model::ParamValue,
    >,
    slot: &str,
) -> Result<u32, CandleBackendError> {
    let key = reimagine_core::model::SlotId::new(slot);
    match params.get(&key) {
        Some(reimagine_core::model::ParamValue::Integer(v)) => {
            if *v < 0 || *v > u32::MAX as i64 {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample param `{slot}` must fit in u32 (got {v})"
                )));
            }
            Ok(*v as u32)
        }
        Some(_) => Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample param `{slot}` must be an integer"
        ))),
        None => Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample requires a `{slot}` param"
        ))),
    }
}

#[allow(dead_code)]
fn extract_f64(
    params: &std::collections::HashMap<
        reimagine_core::model::SlotId,
        reimagine_core::model::ParamValue,
    >,
    slot: &str,
) -> Result<f64, CandleBackendError> {
    let key = reimagine_core::model::SlotId::new(slot);
    match params.get(&key) {
        Some(reimagine_core::model::ParamValue::Float(v)) => Ok(*v),
        Some(reimagine_core::model::ParamValue::Integer(v)) => Ok(*v as f64),
        Some(_) => Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample param `{slot}` must be a number"
        ))),
        None => Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample requires a `{slot}` param"
        ))),
    }
}

#[allow(dead_code)]
fn extract_select<'a>(
    params: &'a std::collections::HashMap<
        reimagine_core::model::SlotId,
        reimagine_core::model::ParamValue,
    >,
    slot: &str,
) -> Result<&'a str, CandleBackendError> {
    let key = reimagine_core::model::SlotId::new(slot);
    match params.get(&key) {
        Some(reimagine_core::model::ParamValue::Select(v)) => Ok(v.as_str()),
        Some(_) => Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample param `{slot}` must be a select"
        ))),
        None => Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample requires a `{slot}` param"
        ))),
    }
}

/// Result of a V1 sampling pass.
///
/// V1 stores the sampled latent tensor through the existing
/// [`CandleStore`] accessors; the operation layer wraps it into a
/// lightweight `ExecutionValue::Latent` handle. The `summary` is kept
/// on the public surface so downstream diagnostic / event sinks can
/// consume it once the operation layer exposes a sampling trace
/// channel; V1 currently only validates it through unit tests.
#[derive(Debug)]
#[cfg(test)]
pub struct SdxlSampleResult {
    pub latent: CandleLatent,
    #[allow(dead_code)]
    pub summary: SampleSummary,
}

/// Compact, human-readable summary of a sampling pass.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub struct SampleSummary {
    pub seed: u64,
    pub steps: u32,
    pub cfg_milli: i64,
    pub sampler: String,
    pub scheduler: String,
    pub denoise_milli: i64,
}

#[cfg(test)]
impl SampleSummary {
    fn from_request(req: &SdxlSampleRequest) -> Self {
        Self {
            seed: req.seed,
            steps: req.steps,
            cfg_milli: (req.cfg * 1000.0).round() as i64,
            sampler: req.sampler_name.clone(),
            scheduler: req.scheduler_name.clone(),
            denoise_milli: (req.denoise * 1000.0).round() as i64,
        }
    }
}

/// Deterministic V1 SDXL sampler.
///
/// The sampler mixes a seeded noise tensor into the input latent with
/// the requested denoise ratio. `steps` is consumed only to scale the
/// effective sampling weight (deterministic, monotone); the placeholder
/// does not run an actual scheduler loop.
#[derive(Debug, Default)]
#[cfg(test)]
pub struct SdxlSampler;

#[cfg(test)]
impl SdxlSampler {
    pub fn new() -> Self {
        Self
    }

    /// Sample a new latent tensor using the V1 deterministic pipeline.
    ///
    /// `input_latent` is the seed latent produced by
    /// `latent.create_empty` (zeros in V1). `request` is the typed
    /// sampler configuration. `device` is the configured Candle device
    /// for the backend.
    ///
    /// The placeholder pipeline:
    ///
    /// 1. Generate a deterministic noise tensor with the same shape as
    ///    the input latent via an xorshift64* PRNG seeded by `seed`.
    /// 2. Scale the noise by `denoise` and the step count so that the
    ///    output remains bounded.
    /// 3. Mix the scaled noise with the input latent:
    ///    `output = input * (1 - denoise) + noise * denoise`.
    /// 4. Multiply the result by `(1 + cfg)` so the
    ///    classifier-free-guidance ratio is observable in tests.
    ///
    /// The exact tensor contents are not part of the contract; only
    /// the determinism guarantee and the shape preservation are.
    pub fn sample(
        &self,
        input_latent: CandleLatent,
        request: &SdxlSampleRequest,
        device: &Device,
    ) -> Result<SdxlSampleResult, CandleBackendError> {
        let latent_space = input_latent.latent_space().clone();
        let tensor = input_latent.into_tensor();
        if tensor.dtype() != DType::F32 {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample expects f32 input latent, got {:?}",
                tensor.dtype()
            )));
        }
        let shape = tensor.shape().dims().to_vec();
        let elem_count: usize = shape.iter().product();

        let mut rng = Xorshift64Star::new(request.seed);
        let noise_cpu = rng.fill_vec(elem_count);
        let noise_tensor = Tensor::from_vec(noise_cpu, shape.clone(), device).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample failed to allocate noise tensor for shape {shape:?}: {err}"
            ))
        })?;

        let denoise = request.denoise as f32;
        let step_scale = (1.0 / (request.steps as f32).max(1.0)) * denoise;
        let cfg_scale = 1.0 + (request.cfg as f32).max(0.0) * 0.05;

        let scaled_noise = (noise_tensor * step_scale as f64).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample noise scaling failed: {err}"
            ))
        })?;
        let mix_keep = tensor.affine(1.0 - denoise as f64, 0.0).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample latent scaling failed: {err}"
            ))
        })?;
        let mixed = (&mix_keep + &scaled_noise).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample latent + noise failed: {err}"
            ))
        })?;
        let scaled = mixed.affine(cfg_scale as f64, 0.0).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample cfg scaling failed: {err}"
            ))
        })?;

        Ok(SdxlSampleResult {
            latent: CandleLatent::new(scaled, latent_space),
            summary: SampleSummary::from_request(request),
        })
    }
}

/// xorshift64* PRNG used to produce deterministic noise without
/// pulling in a new dependency on `rand`.
///
/// The state fits in `u64` and the generator produces values in
/// `[0, u64::MAX]`. The output is mapped to `f32` in `[-1, 1]` by
/// `fill_vec` so it can be turned into a Candle `f32` tensor.
#[derive(Debug, Clone)]
struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    /// Seed the generator from any `u64` seed. Zero seeds are nudged
    /// away from the fixed point at `state == 0`.
    fn new(seed: u64) -> Self {
        let initial = seed ^ 0x9E37_79B9_7F4A_7C15;
        let state = if initial == 0 { 0x1 } else { initial };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Fill a `Vec<f32>` with deterministic noise in `[-1, 1]`.
    fn fill_vec(&mut self, len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let raw = self.next_u64();
            // Use the upper bits to avoid bias from the small modulus
            // when reducing to f32.
            let unit = ((raw >> 11) as f64) / ((1u64 << 53) as f64);
            out.push((unit * 2.0 - 1.0) as f32);
        }
        out
    }
}

pub(crate) fn seeded_noise_like(
    seed: u64,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor, CandleBackendError> {
    let elem_count: usize = shape.iter().product();
    let mut rng = Xorshift64Star::new(seed);
    Tensor::from_vec(rng.fill_vec(elem_count), shape, device).map_err(|err| {
        CandleBackendError::InvalidRequest(format!(
            "diffusion.sample failed to allocate seeded noise tensor for shape {shape:?}: {err}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use reimagine_core::model::{ParamValue, SlotId};
    use std::collections::HashMap;

    fn params(seed: u64, steps: i64, cfg: f64, denoise: f64) -> HashMap<SlotId, ParamValue> {
        let mut p = HashMap::new();
        p.insert(SlotId::new("seed"), ParamValue::Seed(seed));
        p.insert(SlotId::new("steps"), ParamValue::Integer(steps));
        p.insert(SlotId::new("cfg"), ParamValue::Float(cfg));
        p.insert(
            SlotId::new("sampler"),
            ParamValue::Select("euler".to_string()),
        );
        p.insert(
            SlotId::new("scheduler"),
            ParamValue::Select("normal".to_string()),
        );
        p.insert(SlotId::new("denoise"), ParamValue::Float(denoise));
        p
    }

    #[test]
    fn sample_request_accepts_seed_and_integer_steps() {
        let req = SdxlSampleRequest::from_params(&params(7, 20, 7.0, 1.0)).unwrap();
        assert_eq!(req.seed, 7);
        assert_eq!(req.steps, 20);
        assert_eq!(req.cfg, 7.0);
        assert_eq!(req.denoise, 1.0);
        assert_eq!(req.sampler_name, "euler");
        assert_eq!(req.scheduler_name, "normal");
    }

    #[test]
    fn sample_request_rejects_missing_seed() {
        let mut p = params(7, 20, 7.0, 1.0);
        p.remove(&SlotId::new("seed"));
        let err = SdxlSampleRequest::from_params(&p).unwrap_err();
        assert!(err.to_string().contains("seed"));
    }

    #[test]
    fn sample_request_rejects_unknown_sampler() {
        let mut p = params(7, 20, 7.0, 1.0);
        p.insert(
            SlotId::new("sampler"),
            ParamValue::Select("dpmpp_2m".to_string()),
        );
        let err = SdxlSampleRequest::from_params(&p).unwrap_err();
        assert!(err.to_string().contains("dpmpp_2m"));
        assert!(err.to_string().contains("V1"));
    }

    #[test]
    fn sample_request_rejects_unknown_scheduler() {
        let mut p = params(7, 20, 7.0, 1.0);
        p.insert(
            SlotId::new("scheduler"),
            ParamValue::Select("karras".to_string()),
        );
        let err = SdxlSampleRequest::from_params(&p).unwrap_err();
        assert!(err.to_string().contains("karras"));
    }

    #[test]
    fn sample_request_rejects_out_of_range_denoise() {
        let err = SdxlSampleRequest::from_params(&params(7, 20, 7.0, 1.5)).unwrap_err();
        assert!(err.to_string().contains("denoise"));
        let err = SdxlSampleRequest::from_params(&params(7, 20, 7.0, -0.1)).unwrap_err();
        assert!(err.to_string().contains("denoise"));
    }

    #[test]
    fn sample_request_rejects_zero_steps() {
        let err = SdxlSampleRequest::from_params(&params(7, 0, 7.0, 1.0)).unwrap_err();
        assert!(err.to_string().contains("steps"));
    }

    #[test]
    fn sampler_is_deterministic_for_same_seed() {
        let sampler = SdxlSampler::new();
        let device = Device::Cpu;
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::F32, &device).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        let req = SdxlSampleRequest::from_params(&params(12345, 20, 7.0, 1.0)).unwrap();

        let first = sampler
            .sample(latent.clone(), &req, &device)
            .unwrap()
            .latent
            .into_tensor();
        let second = sampler
            .sample(latent, &req, &device)
            .unwrap()
            .latent
            .into_tensor();

        let first_data = first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let second_data = second.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(first_data, second_data, "sampling must be deterministic");
    }

    #[test]
    fn sampler_changes_output_for_different_seed() {
        let sampler = SdxlSampler::new();
        let device = Device::Cpu;
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::F32, &device).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );

        let a = sampler
            .sample(
                latent.clone(),
                &SdxlSampleRequest::from_params(&params(1, 20, 7.0, 1.0)).unwrap(),
                &device,
            )
            .unwrap()
            .latent
            .into_tensor();
        let b = sampler
            .sample(
                latent,
                &SdxlSampleRequest::from_params(&params(2, 20, 7.0, 1.0)).unwrap(),
                &device,
            )
            .unwrap()
            .latent
            .into_tensor();

        let a_data = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b_data = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(a_data, b_data);
    }

    #[test]
    fn sampler_preserves_shape() {
        let sampler = SdxlSampler::new();
        let device = Device::Cpu;
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 16, 16), DType::F32, &device).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        let req = SdxlSampleRequest::from_params(&params(7, 10, 7.0, 1.0)).unwrap();
        let result = sampler.sample(latent, &req, &device).unwrap();
        assert_eq!(result.latent.dims(), vec![1, 4, 16, 16]);
        assert_eq!(result.latent.dtype(), DType::F32);
    }

    #[test]
    fn sampler_rejects_non_f32_input() {
        let sampler = SdxlSampler::new();
        let device = Device::Cpu;
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::U32, &device).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        let req = SdxlSampleRequest::from_params(&params(7, 10, 7.0, 1.0)).unwrap();
        let err = sampler.sample(latent, &req, &device).unwrap_err();
        assert!(err.to_string().contains("f32"));
    }

    #[test]
    fn sample_summary_captures_request() {
        let req = SdxlSampleRequest::from_params(&params(42, 30, 6.5, 1.0)).unwrap();
        let summary = SampleSummary::from_request(&req);
        assert_eq!(summary.seed, 42);
        assert_eq!(summary.steps, 30);
        assert_eq!(summary.cfg_milli, 6500);
        assert_eq!(summary.denoise_milli, 1000);
        assert_eq!(summary.sampler, "euler");
        assert_eq!(summary.scheduler, "normal");
    }
}
