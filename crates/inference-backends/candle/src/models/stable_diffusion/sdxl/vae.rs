//! SDXL VAE decoder implementation.
//!
//! V1 uses a deterministic, shape-correct placeholder decoder so that
//! the backend operation contract stays testable without requiring real
//! VAE decoder weights. The decoder converts a sampled SDXL latent
//! tensor into an RGB image tensor.
//!
//! ## Responsibilities
//!
//! - Validates input latent shape `[batch, 4, h, w]` and dtype `F32`
//! - Applies the official SDXL VAE latent scale factor (`0.18215`)
//! - Upscales spatial dimensions by 8x (`[h, w]` → `[h*8, w*8]`)
//! - Maps 4 latent channels to 3 RGB channels (takes first 3 channels)
//! - Maps value range from `[-1, 1]` to `[0, 1]`
//! - Returns a deterministic output tensor for the same input
//!
//! ## Non-responsibilities
//!
//! - Does NOT load real VAE decoder weights (V1 placeholder)
//! - Does NOT perform actual VAE decode (卷积 decode) — spatial upscaling
//!   is done via nearest-neighbor interpolation
//! - Does NOT apply real VAE sigmoid / normalization (V1 uses affine
//!   rescale to `[0, 1]`)
//! - The exact pixel values are not part of the contract; only shape,
//!   dtype, value-range, and determinism are guaranteed
//!
//! ## SDXL VAE constants
//!
//! - **Latent scale factor:** `0.18215` — the official SDXL VAE scaling
//!   constant. V1 divides the latent by this factor before upscaling.
//! - **Spatial upscale factor:** 8x — SDXL VAE decodes `4×h×w` latent
//!   to `3×8h×8w` RGB image.
//! - **Channel mapping:** 4 latent channels → 3 RGB channels. V1 takes
//!   the first 3 channels of the latent (discards channel 4). An
//!   alternative of summing all 4 channels and broadcasting to 3 was
//!   considered but not used in V1.

use candle_core::{DType, Device, Tensor};

use crate::error::CandleBackendError;
use crate::store::{CandleImage, CandleLatent};

/// SDXL VAE latent scale factor (official constant).
///
/// The SDXL VAE encodes images into latents that are scaled by this
/// factor. To decode, we divide by the scale factor to restore the
/// VAE's expected input range before upscaling.
const VAE_SCALE_FACTOR: f32 = 0.18215;

/// V1 deterministic SDXL VAE decoder.
///
/// The decoder takes a sampled SDXL latent tensor (from `diffusion.sample`)
/// and produces an RGB image tensor. V1 uses nearest-neighbor 8x upscaling
/// and a simple value-range remap; real VAE decode (卷积 upsampling) is
/// deferred to a future milestone.
#[derive(Debug)]
pub struct SdxlVaeDecoder;

impl SdxlVaeDecoder {
    /// Create a new VAE decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decode a sampled SDXL latent tensor into an RGB image tensor.
    ///
    /// V1 uses a placeholder: the decoder divides the latent by the SDXL
    /// VAE scale factor (`0.18215`), upscales spatial dimensions by 8x
    /// via nearest-neighbor interpolation, takes the first 3 of 4 latent
    /// channels as RGB, and remaps the value range from `[-1, 1]` to
    /// `[0, 1]`. The placeholder does NOT load real VAE weights; it still
    /// produces a deterministic real tensor with the correct shape and
    /// dtype.
    ///
    /// # Arguments
    ///
    /// * `latent` — SDXL latent in shape `[batch, 4, h, w]` with values
    ///   roughly in `[-1, 1]` (the sampler output in V1 is already bounded
    ///   in this range).
    /// * `device` — the configured Candle device for the backend.
    ///
    /// # Returns
    ///
    /// Returns a [`CandleImage`] containing:
    /// - `tensor` of shape `[batch, 3, h*8, w*8]` and dtype `F32`
    ///   with values in `[0, 1]`
    /// - `width` = `w * 8`
    /// - `height` = `h * 8`
    /// - `batch` = `batch`
    /// - `color_space` = `"rgb"`
    ///
    /// # Errors
    ///
    /// Returns [`SdxlVaeError::Tensor`] if Candle tensor operations fail.
    /// Returns [`SdxlVaeError::Shape`] if the input shape is invalid.
    pub fn decode(
        &self,
        latent: &CandleLatent,
        device: &Device,
    ) -> Result<CandleImage, SdxlVaeError> {
        let tensor = latent.tensor();

        if tensor.dtype() != DType::F32 {
            return Err(SdxlVaeError::Tensor(format!(
                "VAE decoder expects f32 input latent, got {:?}",
                tensor.dtype()
            )));
        }

        let dims = tensor.shape().dims();
        if dims.len() != 4 {
            return Err(SdxlVaeError::Shape(format!(
                "VAE decoder expects 4D input latent [batch, 4, h, w], got {}-D shape {:?}",
                dims.len(),
                dims
            )));
        }

        let batch = dims[0];
        let channels = dims[1];
        let h = dims[2];
        let w = dims[3];

        if channels != 4 {
            return Err(SdxlVaeError::Shape(format!(
                "VAE decoder expects exactly 4 latent channels, got {channels}"
            )));
        }

        if h == 0 || w == 0 {
            return Err(SdxlVaeError::Shape(format!(
                "VAE decoder expects positive spatial dimensions h>0 and w>0, got h={h}, w={w}"
            )));
        }

        // Step 1: Apply the SDXL VAE scale factor (divide by 0.18215).
        // This converts the latent back to the VAE's expected input range.
        let scale_tensor = Tensor::new(VAE_SCALE_FACTOR, device)
            .map_err(|e| SdxlVaeError::Tensor(format!("failed to create scale tensor: {e}")))?;
        let scaled = tensor
            .broadcast_div(&scale_tensor)
            .map_err(|e| SdxlVaeError::Tensor(format!("VAE scale division failed: {e}")))?;

        // Step 2: Upscale spatial dimensions by 8x using nearest-neighbor.
        let h_out = h * 8;
        let w_out = w * 8;
        let upscaled = scaled
            .upsample_nearest2d(h_out, w_out)
            .map_err(|e| SdxlVaeError::Tensor(format!("VAE upsample failed: {e}")))?;

        // Step 3: Map 4 latent channels to 3 RGB channels — take the first 3.
        let rgb = upscaled
            .narrow(1, 0, 3)
            .map_err(|e| SdxlVaeError::Tensor(format!("VAE channel slice failed: {e}")))?;

        // Step 4: Remap the value range from [-1, 1] to [0, 1] with a single affine:
        // output = input * 0.5 + 0.5. The V1 placeholder diffusion sampler
        // produces bounded latent values, so the output is expected to land
        // in [0, 1] for normal inputs.
        let remapped = rgb
            .affine(0.5, 0.5)
            .map_err(|e| SdxlVaeError::Tensor(format!("VAE rescale failed: {e}")))?;

        Ok(CandleImage::new(
            remapped,
            w_out as u32,
            h_out as u32,
            batch as u32,
            "rgb".to_string(),
        ))
    }
}

impl Default for SdxlVaeDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during VAE decoding.
#[derive(Debug)]
pub enum SdxlVaeError {
    /// A Candle tensor operation failed.
    Tensor(String),
    /// The input latent had an invalid shape.
    Shape(String),
}

impl std::fmt::Display for SdxlVaeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(msg) => write!(f, "VAE decoder tensor error: {msg}"),
            Self::Shape(msg) => write!(f, "VAE decoder shape error: {msg}"),
        }
    }
}

impl std::error::Error for SdxlVaeError {}

impl From<SdxlVaeError> for CandleBackendError {
    fn from(err: SdxlVaeError) -> Self {
        CandleBackendError::InvalidRequest(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> &'static Device {
        &Device::Cpu
    }

    fn f32_latent(shape: &[usize]) -> CandleLatent {
        CandleLatent::new(Tensor::zeros(shape, DType::F32, cpu()).unwrap())
    }

    #[test]
    fn decoder_rejects_non_f32_input() {
        let decoder = SdxlVaeDecoder::new();
        let latent = CandleLatent::new(Tensor::zeros((1, 4, 8, 8), DType::U32, cpu()).unwrap());
        let err = decoder.decode(&latent, cpu()).unwrap_err();
        assert!(err.to_string().contains("f32"), "got: {}", err);
        assert!(matches!(err, SdxlVaeError::Tensor(_)));
    }

    #[test]
    fn decoder_rejects_wrong_channels() {
        let decoder = SdxlVaeDecoder::new();
        let latent = f32_latent(&[1, 3, 8, 8]);
        let err = decoder.decode(&latent, cpu()).unwrap_err();
        assert!(err.to_string().contains("4"), "got: {}", err);
        assert!(err.to_string().contains("channels"), "got: {}", err);
        assert!(matches!(err, SdxlVaeError::Shape(_)));
    }

    #[test]
    fn decoder_rejects_non_4d_input() {
        let decoder = SdxlVaeDecoder::new();
        let latent = CandleLatent::new(Tensor::zeros((1, 4, 16), DType::F32, cpu()).unwrap());
        let err = decoder.decode(&latent, cpu()).unwrap_err();
        assert!(err.to_string().contains("4D"), "got: {}", err);
        assert!(matches!(err, SdxlVaeError::Shape(_)));
    }

    #[test]
    fn decoder_produces_correct_shape() {
        let decoder = SdxlVaeDecoder::new();
        let latent = f32_latent(&[1, 4, 16, 16]);
        let image = decoder.decode(&latent, cpu()).unwrap();

        assert_eq!(image.tensor().shape().dims(), &[1, 3, 128, 128]);
        assert_eq!(image.tensor().dtype(), DType::F32);
        assert_eq!(image.width(), 128);
        assert_eq!(image.height(), 128);
        assert_eq!(image.batch(), 1);
        assert_eq!(image.color_space(), "rgb");
    }

    #[test]
    fn decoder_handles_batch_size_greater_than_one() {
        let decoder = SdxlVaeDecoder::new();
        let latent = f32_latent(&[2, 4, 8, 8]);
        let image = decoder.decode(&latent, cpu()).unwrap();

        assert_eq!(image.tensor().shape().dims(), &[2, 3, 64, 64]);
        assert_eq!(image.batch(), 2);
    }

    #[test]
    fn decoder_output_values_in_unit_range() {
        let decoder = SdxlVaeDecoder::new();
        let latent = f32_latent(&[1, 4, 8, 8]);
        let image = decoder.decode(&latent, cpu()).unwrap();

        let data = image
            .tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            data.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "all values should be in [0, 1], got range [{:.3}, {:.3}]",
            data.iter().cloned().fold(f32::INFINITY, f32::min),
            data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
    }

    #[test]
    fn decoder_is_deterministic_for_same_input() {
        let decoder = SdxlVaeDecoder::new();
        let latent = f32_latent(&[1, 4, 8, 8]);

        let first = decoder.decode(&latent, cpu()).unwrap().into_tensor();
        let second = decoder.decode(&latent, cpu()).unwrap().into_tensor();

        let first_data = first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let second_data = second.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(first_data, second_data, "decoder must be deterministic");
    }
}
