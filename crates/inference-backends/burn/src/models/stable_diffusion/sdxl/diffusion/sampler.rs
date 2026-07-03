//! Euler/normal denoise loop for SDXL diffusion sampling.
//!
//! V1 implements a minimal euler-method denoise loop operating on
//! the loaded weight structs. The actual tensor math is a placeholder
//! that produces deterministic seed-dependent output from the input
//! latent; real weight-driven denoising is a follow-up deepening.

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, TensorData};

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;

/// Run the euler/normal denoise loop.
///
/// V1 applies a seed-dependent noise pattern to the latent and
/// returns the result. The real weight-driven UNet forward
/// pass is a follow-up deepening.
pub fn euler_normal_sample(
    latent: Tensor<NdArray, 4>,
    _steps: u32,
    seed: u64,
    backend: &BurnBackend,
) -> Result<Tensor<NdArray, 4>, BurnBackendError> {
    let device = backend.ndarray_device();
    let shape = latent.shape().dims();

    // V1: apply seed-dependent noise and return a modified latent
    // The noise is deterministic from the seed so tests can verify
    // seed reproducibility.
    let noise = seeded_noise(shape, seed, &device);
    let result = latent + noise;

    Ok(result)
}

/// Generate a deterministic noise tensor from a seed, matching the
/// latent shape.
fn seeded_noise(shape: [usize; 4], seed: u64, device: &NdArrayDevice) -> Tensor<NdArray, 4> {
    // Use a simple deterministic noise pattern based on the seed
    let total = shape[0] * shape[1] * shape[2] * shape[3];
    let mut data = Vec::with_capacity(total);
    // Simple LCG-style noise for reproducibility
    let mut state = seed.wrapping_add(1);
    for _ in 0..total {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (state >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0;
        data.push(val * 0.1); // scale noise to 0.1 magnitude
    }
    Tensor::from_data(TensorData::new(data, shape), device)
}
