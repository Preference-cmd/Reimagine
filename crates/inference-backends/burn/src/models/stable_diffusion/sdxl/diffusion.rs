//! SDXL diffusion sampling — routine definitions and module-level
//! re-exports for the SDXL diffusion path.

mod loading;
pub mod module;
mod sampler;

use burn_ndarray::NdArray;
use burn_tensor::Tensor;

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::BurnLoadedModelBundle;
use crate::store::BurnConditioningPayload;

/// Run the SDXL euler/normal denoise loop.
///
/// Accepts a zero latent (EmptyGeometry), positive and negative
/// conditioning payloads, and returns a sampled latent tensor.
pub fn sample_sdxl(
    _bundle: &BurnLoadedModelBundle,
    latent: Tensor<NdArray, 4>,
    _positive_cond: &BurnConditioningPayload,
    _negative_cond: &BurnConditioningPayload,
    steps: u32,
    _cfg: f32,
    seed: u64,
    backend: &BurnBackend,
) -> Result<Tensor<NdArray, 4>, BurnBackendError> {
    // V1: apply seed-dependent noise and return a modified latent.
    // The real weight-driven UNet forward pass is a follow-up deepening.
    let result = sampler::euler_normal_sample(latent, steps, seed, backend)?;
    Ok(result)
}
