//! SDXL VAE decode — routine definitions and module-level re-exports.

mod decode;
mod module;

use burn_ndarray::NdArray;
use burn_tensor::Tensor;

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::BurnLoadedModelBundle;
use crate::store::BurnLatentPayload;

/// Run SDXL VAE decode on a sampled latent.
///
/// Returns a decoded image tensor in NCHW F32 format, normalized
/// to [0, 1] range.
pub fn decode_latent(
    _bundle: &BurnLoadedModelBundle,
    latent: BurnLatentPayload,
    backend: &BurnBackend,
) -> Result<Tensor<NdArray, 4>, BurnBackendError> {
    // V1: apply a minimal decode that produces a correct-shape output
    // from a real latent. The actual VAE weight-driven decode is a
    // follow-up deepening.
    let tensor = decode::vae_decode_placeholder(latent, backend)?;
    Ok(tensor)
}
