//! SDXL VAE decode — sampled latent to F32 RGB NCHW image.
//!
//! V1 provides a placeholder decode that produces a deterministic
//! output tensor in the correct shape. The real weight-driven VAE
//! decode is a follow-up deepening.

use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, TensorData};

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::store::BurnLatentPayload;

/// Placeholder VAE decode returning a minimal image tensor.
///
/// V1 converts the latent to a simple gradient pattern in the correct
/// [1, 3, height*8, width*8] shape so the downstream image.save/
/// preview capabilities can verify the full pipeline works.
pub fn vae_decode_placeholder(
    latent: BurnLatentPayload,
    _backend: &BurnBackend,
) -> Result<Tensor<NdArray, 4>, BurnBackendError> {
    let dims = latent.dims();
    let batch = dims[0];
    let latent_h = dims[2];
    let latent_w = dims[3];

    // VAE spatial scale factor is 8 for SDXL
    let height = latent_h * 8;
    let width = latent_w * 8;

    // V1: produce a simple deterministic output from the latent
    // The real VAE decode would run through the decoder modules.
    let total = batch * 3 * height * width;
    let data = vec![0.5f32; total]; // mid-gray

    let tensor = Tensor::<NdArray, 4>::from_data(
        TensorData::new(data, [batch, 3, height, width]),
        &NdArrayDevice::Cpu,
    );

    Ok(tensor)
}
