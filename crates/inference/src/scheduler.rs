//! Backend-neutral diffusion scheduler trait.
//!
//! A scheduler computes the noise schedule (sigmas, timesteps) and
//! provides the math for denoising steps. This trait lives in
//! `reimagine-inference` so backends implement it without the runtime
//! needing to depend on backend crates.
//!
//! The trait covers the pure-math interface (timesteps, sigmas, alpha
//! products) that is common across all scheduler algorithms. The
//! actual tensor-level `step` operation remains backend-specific
//! because it depends on the concrete tensor type.

use crate::inference_error::InferenceError;

/// A diffusion scheduler that computes timesteps and noise schedule.
///
/// The trait is model-neutral: it operates on opaque timestep/sigma
/// values, not on specific tensor types. Each backend provides a
/// concrete scheduler that wraps this trait with backend-specific
/// tensor operations.
pub trait DiffusionScheduler: Send + Sync {
    /// Initialize the scheduler with the given number of inference steps.
    ///
    /// After calling this, `timesteps()`, `sigmas()`, and
    /// `alphas_cumprod()` return the computed schedule.
    fn set_timesteps(&mut self, num_inference_steps: u32) -> Result<(), InferenceError>;

    /// The current list of timesteps (descending noise order).
    ///
    /// Timesteps are in noise-schedule index space (0 = clean,
    /// 999 = maximum noise for a 1000-step training schedule).
    fn timesteps(&self) -> &[f64];

    /// The sigma (noise level) for each inference timestep.
    fn sigmas(&self) -> &[f64];

    /// The cumulative alpha product for each inference timestep.
    fn alphas_cumprod(&self) -> &[f64];

    /// The initial noise sigma for seeding the latent.
    fn init_noise_sigma(&self) -> f64;

    /// Sigma at a specific timestep index.
    fn sigma_at(&self, index: usize) -> Result<f64, InferenceError>;

    /// Cumulative alpha product at a specific timestep index.
    fn alpha_cumprod_at(&self, index: usize) -> Result<f64, InferenceError>;
}

/// A sampler that drives the denoising loop (e.g. Euler, DPM++, Heun).
///
/// The sampler owns the step logic and calls into the model for noise
/// prediction. This trait is intentionally narrow — the full CFG loop
/// and model forward are backend-specific.
pub trait DiffusionSampler: Send + Sync {
    /// The sampler name (e.g. "euler", "dpmpp_2m").
    fn name(&self) -> &str;
}

/// Callback invoked between denoising steps that may modify the latent.
///
/// Implementations can use this to apply prompt scheduling, CFG cutoff,
/// latent manipulation, or other per-step interventions.
///
/// The callback receives the step index and current timestep value.
/// Return `Ok(true)` to continue denoising, or `Ok(false)` to abort
/// the loop early (the sampler will return the current latent).
pub trait DenoisingCallback: Send + Sync {
    /// Called after each denoising step completes.
    ///
    /// - `step_index`: 0-based index of the step that just completed.
    /// - `timestep`: The noise-schedule timestep for this step.
    ///
    /// Returns `Ok(true)` to continue, `Ok(false)` to abort early.
    fn on_step(&self, step_index: usize, timestep: f64) -> Result<bool, InferenceError>;
}
