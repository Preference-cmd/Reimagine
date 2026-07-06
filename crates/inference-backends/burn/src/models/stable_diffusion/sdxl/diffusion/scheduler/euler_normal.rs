//! Euler Normal scheduler for SDXL diffusion sampling.
//!
//! Pure-math discrete scheduler computing the linear beta schedule,
//! cumulative alpha products, and sigma values from a 1000-step
//! training schedule, then selecting `num_inference_steps` uniformly
//! spaced timesteps.
//!
//! The `step` method applies the standard Euler normal update
//! operating on 4-dimensional latent tensors:
//!
//! ```text
//! x_{t-1} = sqrt(alpha_bar_prev / alpha_bar_t) *
//!           (x_t - sigma_t * noise_pred) + sigma_prev * noise_pred
//! ```

use burn_tensor::{Tensor, backend::Backend};

use crate::error::BurnBackendError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Total number of training timesteps the noise schedule was trained on.
#[allow(dead_code)]
const NUM_TRAIN_TSTEPS: u32 = 1000;
/// Minimum beta value in the linear schedule.
#[allow(dead_code)]
const BETA_START: f64 = 0.00085;
/// Maximum beta value in the linear schedule.
#[allow(dead_code)]
const BETA_END: f64 = 0.012;

// ---------------------------------------------------------------------------
// Scheduler struct
// ---------------------------------------------------------------------------

/// Euler Normal scheduler for SDXL diffusion sampling.
///
/// Stores precomputed noise levels (`sigmas`) and cumulative alpha products
/// (`alphas_cumprod`) indexed by inference timestep.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct EulerNormalScheduler {
    /// All inference timestep indices used for this run, in
    /// descending noise order (t=0 is the noisiest step).
    pub timesteps: Vec<f64>,
    /// Sigma (noise level) for each inference timestep index.
    pub sigmas: Vec<f64>,
    /// Cumulative product of (1 - beta) for each inference timestep index.
    pub alphas_cumprod: Vec<f64>,
    /// Sigma value used to initialise the noised latent (always 1.0).
    pub init_noise_sigma: f64,
}

#[allow(dead_code)]
impl EulerNormalScheduler {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build a scheduler for `num_inference_steps` denoise steps.
    ///
    /// Computes the full 1000-step beta / alpha / sigma schedule and
    /// selects `num_inference_steps` uniformly spaced timestep indices.
    ///
    /// Returns `Err(BurnBackendError::InvalidRequest(_))` when
    /// `num_inference_steps` is zero.
    pub fn new(num_inference_steps: u32) -> Result<Self, BurnBackendError> {
        let n = num_inference_steps as usize;
        if n == 0 {
            return Err(BurnBackendError::InvalidRequest(
                "num_inference_steps must be at least 1".to_string(),
            ));
        }

        // Full 1000-step training schedule
        let betas = linspace(BETA_START, BETA_END, NUM_TRAIN_TSTEPS as usize);
        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();
        let alphas_cumprod_full = cumprod(&alphas);
        let sigmas_full: Vec<f64> = alphas_cumprod_full
            .iter()
            .map(|ac| ((1.0 - ac) / ac).sqrt())
            .collect();

        // Select inference timesteps uniformly from the full schedule, then
        // reverse so timesteps go high→low (high noise first → low noise last).
        // Specification: timesteps = [round(i * step_ratio) for i in 0..N]
        let step_ratio = NUM_TRAIN_TSTEPS as f64 / num_inference_steps as f64;
        let mut indices: Vec<usize> = (0..n)
            .map(|i| (i as f64 * step_ratio).round() as usize)
            .collect();
        indices.reverse();

        let timesteps: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
        let sigmas: Vec<f64> = indices.iter().map(|&i| sigmas_full[i]).collect();
        let alphas_cumprod: Vec<f64> = indices.iter().map(|&i| alphas_cumprod_full[i]).collect();

        Ok(Self {
            timesteps,
            sigmas,
            alphas_cumprod,
            init_noise_sigma: 1.0,
        })
    }

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Sigma (noise level) at the given timestep index (0-based).
    ///
    /// Returns `Err(BurnBackendError::InvalidRequest(_))` when `t` is
    /// out of range.
    pub fn sigma_at(&self, t: usize) -> Result<f64, BurnBackendError> {
        self.sigmas.get(t).copied().ok_or_else(|| {
            BurnBackendError::InvalidRequest(format!(
                "timestep index {t} out of range (sigmas len={})",
                self.sigmas.len()
            ))
        })
    }

    // -----------------------------------------------------------------------
    // Single sampling step
    // -----------------------------------------------------------------------

    /// Apply one Euler normal step over an active Burn backend tensor.
    pub fn step_tensor<B: Backend>(
        &self,
        x: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        t: usize,
    ) -> Result<Tensor<B, 4>, BurnBackendError> {
        if t >= self.sigmas.len() {
            return Err(BurnBackendError::InvalidRequest(format!(
                "timestep index {t} out of range (len={})",
                self.sigmas.len()
            )));
        }

        let sigma_t = self.sigmas[t];
        let sigma_prev = if t + 1 < self.sigmas.len() {
            self.sigmas[t + 1]
        } else {
            0.0
        };
        let alpha_bar_t = self.alphas_cumprod[t];
        let alpha_bar_prev = if t + 1 < self.alphas_cumprod.len() {
            self.alphas_cumprod[t + 1]
        } else {
            1.0
        };

        let sigma_t_s = sigma_t as f32;
        let sigma_prev_s = sigma_prev as f32;
        let sqrt_ratio = (alpha_bar_prev / alpha_bar_t).sqrt() as f32;

        let denoised = x - noise_pred.clone() * sigma_t_s;
        Ok(denoised * sqrt_ratio + noise_pred * sigma_prev_s)
    }
}

// ---------------------------------------------------------------------------
// Private schedule helpers
// ---------------------------------------------------------------------------

/// Linearly interpolate `n` values between `start` and `end` (inclusive).
#[allow(dead_code)]
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// Cumulative product of the input slice.
#[allow(dead_code)]
fn cumprod(v: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    let mut acc = 1.0;
    for &x in v {
        acc *= x;
        out.push(acc);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;

    // -----------------------------------------------------------------------
    // Schedule construction
    // -----------------------------------------------------------------------

    #[test]
    fn new_20_produces_20_timesteps() {
        let s = EulerNormalScheduler::new(20).unwrap();
        assert_eq!(s.timesteps.len(), 20);
        assert_eq!(s.sigmas.len(), 20);
        assert_eq!(s.alphas_cumprod.len(), 20);
    }

    #[test]
    fn new_zero_steps_returns_error() {
        let err = EulerNormalScheduler::new(0).unwrap_err();
        match err {
            BurnBackendError::InvalidRequest(_) => {}
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn sigmas_are_monotonically_decreasing() {
        let s = EulerNormalScheduler::new(20).unwrap();
        for i in 0..s.sigmas.len() - 1 {
            assert!(
                s.sigmas[i] > s.sigmas[i + 1],
                "sigmas[{}]={} is not > sigmas[{}]={}",
                i,
                s.sigmas[i],
                i + 1,
                s.sigmas[i + 1]
            );
        }
    }

    #[test]
    fn init_noise_sigma_is_one() {
        let s = EulerNormalScheduler::new(10).unwrap();
        assert!(
            (s.init_noise_sigma - 1.0).abs() < 1e-12,
            "init_noise_sigma={}",
            s.init_noise_sigma
        );
    }

    #[test]
    fn beta_schedule_matches_linspace_to_1e6() {
        let s = EulerNormalScheduler::new(20).unwrap();
        let expected_betas = linspace(BETA_START, BETA_END, NUM_TRAIN_TSTEPS as usize);
        let expected_alphas: Vec<f64> = expected_betas.iter().map(|b| 1.0 - b).collect();
        let expected_ac = cumprod(&expected_alphas);

        // Indices after reversal: last n indices reversed
        let step_ratio = NUM_TRAIN_TSTEPS as f64 / 20.0;
        let mut full_indices: Vec<usize> = (0..20usize)
            .map(|i| (i as f64 * step_ratio).round() as usize)
            .collect();
        full_indices.reverse();

        for (i, full_idx) in full_indices.iter().copied().enumerate().take(20usize) {
            let expected = expected_ac[full_idx];
            assert!(
                (s.alphas_cumprod[i] - expected).abs() < 1e-6,
                "alphas_cumprod[{}] (full idx {}): got {}, expected {}",
                i,
                full_idx,
                s.alphas_cumprod[i],
                expected
            );
        }
    }

    #[test]
    fn alpha_cumprod_matches_cumprod_of_one_minus_beta() {
        let s = EulerNormalScheduler::new(20).unwrap();
        let betas = linspace(BETA_START, BETA_END, NUM_TRAIN_TSTEPS as usize);
        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();
        let expected_full = cumprod(&alphas);

        // Indices after reversal
        let step_ratio = NUM_TRAIN_TSTEPS as f64 / 20.0;
        let mut full_indices: Vec<usize> = (0..20usize)
            .map(|i| (i as f64 * step_ratio).round() as usize)
            .collect();
        full_indices.reverse();

        for (i, full_idx) in full_indices.iter().copied().enumerate().take(20usize) {
            let expected = expected_full[full_idx];
            assert!(
                (s.alphas_cumprod[i] - expected).abs() < 1e-6,
                "alphas_cumprod[{}]: got {}, expected {}",
                i,
                s.alphas_cumprod[i],
                expected
            );
        }
    }

    #[test]
    fn sigma_at_matches_sigmas_field() {
        let s = EulerNormalScheduler::new(20).unwrap();
        for i in 0..20 {
            assert_eq!(s.sigma_at(i).unwrap(), s.sigmas[i]);
        }
    }

    #[test]
    fn sigma_at_out_of_range_returns_error() {
        let s = EulerNormalScheduler::new(20).unwrap();
        let err = s.sigma_at(20).unwrap_err();
        match err {
            BurnBackendError::InvalidRequest(_) => {}
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn timesteps_uniformly_spaced() {
        let s = EulerNormalScheduler::new(10).unwrap();
        // step_ratio = 1000 / 10 = 100
        // timesteps after reversal = [900, 800, 700, ..., 0]
        let expected: Vec<f64> = (0..10).rev().map(|i| (i as f64 * 100.0).round()).collect();
        assert_eq!(s.timesteps, expected);
    }

    // -----------------------------------------------------------------------
    // step() math
    // -----------------------------------------------------------------------

    /// Manually compute the expected output for element-wise step.
    fn hand_compute_step(
        x_vals: &[f32],
        pred_vals: &[f32],
        sigma_t: f64,
        sigma_prev: f64,
        alpha_bar_t: f64,
        alpha_bar_prev: f64,
    ) -> Vec<f32> {
        let st = sigma_t as f32;
        let sp = sigma_prev as f32;
        let sr = (alpha_bar_prev / alpha_bar_t).sqrt() as f32;
        x_vals
            .iter()
            .zip(pred_vals.iter())
            .map(|(&x, &p)| sr * (x - st * p) + sp * p)
            .collect()
    }

    fn test_device() -> burn_tensor::Device<ActiveBurnBackend> {
        let config = BurnBackendConfig::new("/models", "/output");
        active_device(config.device())
    }

    #[test]
    fn step_matches_hand_computed_example() {
        let s = EulerNormalScheduler::new(20).unwrap();
        let sigma_t = s.sigmas[0];
        let sigma_prev = s.sigmas[1];
        let alpha_bar_t = s.alphas_cumprod[0];
        let alpha_bar_prev = s.alphas_cumprod[1];

        let x_vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let pred_vals = vec![0.1f32, 0.2, 0.3, 0.4];

        let device = test_device();
        let x = Tensor::<ActiveBurnBackend, 4>::from_data(
            TensorData::new(x_vals.clone(), burn_tensor::Shape::new([1, 1, 2, 2])),
            &device,
        );
        let pred = Tensor::<ActiveBurnBackend, 4>::from_data(
            TensorData::new(pred_vals.clone(), burn_tensor::Shape::new([1, 1, 2, 2])),
            &device,
        );

        let result = s.step_tensor(x, pred, 0).unwrap();
        let out_vals = result.to_data().to_vec::<f32>().unwrap();

        let expected = hand_compute_step(
            &x_vals,
            &pred_vals,
            sigma_t,
            sigma_prev,
            alpha_bar_t,
            alpha_bar_prev,
        );
        for (i, (got, exp)) in out_vals.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }

    #[test]
    fn step_preserves_shape() {
        let s = EulerNormalScheduler::new(10).unwrap();
        let device = test_device();
        let x = Tensor::<ActiveBurnBackend, 4>::zeros([2, 4, 8, 8], &device);
        let pred = Tensor::<ActiveBurnBackend, 4>::zeros([2, 4, 8, 8], &device);
        let result = s.step_tensor(x, pred, 0).unwrap();
        assert_eq!(result.shape().dims(), [2, 4, 8, 8]);
    }

    #[test]
    fn step_out_of_range_index_returns_error() {
        let s = EulerNormalScheduler::new(5).unwrap();
        let device = test_device();
        let x = Tensor::<ActiveBurnBackend, 4>::zeros([1, 1, 2, 2], &device);
        let pred = Tensor::<ActiveBurnBackend, 4>::zeros([1, 1, 2, 2], &device);
        let err = s.step_tensor(x, pred, 5).unwrap_err();
        match err {
            BurnBackendError::InvalidRequest(_) => {}
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn step_last_index_uses_zero_sigma_prev() {
        let s = EulerNormalScheduler::new(10).unwrap();
        let last = 9;
        let pred_vals = vec![0.0f32; 4];
        let x_vals = vec![1.0f32, -1.0, 0.5, -0.5];

        let device = test_device();
        let x = Tensor::<ActiveBurnBackend, 4>::from_data(
            TensorData::new(x_vals.clone(), burn_tensor::Shape::new([1, 1, 2, 2])),
            &device,
        );
        let pred = Tensor::<ActiveBurnBackend, 4>::from_data(
            TensorData::new(pred_vals.clone(), burn_tensor::Shape::new([1, 1, 2, 2])),
            &device,
        );

        let result = s.step_tensor(x, pred, last).unwrap();
        let out_vals = result.to_data().to_vec::<f32>().unwrap();

        // At last step: alpha_bar_prev=1.0, sigma_prev=0.0, pred=0 => x_{t-1} = sqrt(1/alpha_bar_t) * x
        let sqrt_ratio = (1.0 / s.alphas_cumprod[last]).sqrt() as f32;
        let expected: Vec<f32> = x_vals.iter().map(|&v| sqrt_ratio * v).collect();

        for (i, (got, exp)) in out_vals.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }
}
