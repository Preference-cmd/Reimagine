//! SDXL VAE graph implementation.
//!
//! V1 real encode/decode targets Candle-compatible split VAE
//! component sources (a single safetensors file with bare
//! `decoder.* / post_quant_conv.* / encoder.* / quant_conv.*`
//! keys produced by the
//! [`crate::models::stable_diffusion::sdxl::checkpoint_import`]
//! pipeline). The same backend-private graph owns one Candle
//! [`AutoEncoderKL`] and exposes both latent-to-image decode and
//! image-to-latent encode behind model-neutral operation names.
//!
//! ## Layering
//!
//! - [`SdxlVaeGraph`] — backend-private graph facade. Owns
//!   the per-loaded-graph VAE state. Exposes
//!   [`SdxlVaeGraph::decode`] and [`SdxlVaeGraph::encode`] to
//!   operation code, never the underlying Candle modules.
//! - [`SdxlRealVae`] — Candle-owned VAE module using
//!   `candle_transformers::models::stable_diffusion::vae::AutoEncoderKL`.
//! - Test-only placeholder path is opt-in via
//!   [`SdxlVaeGraph::test_placeholder`] so production code
//!   cannot accidentally encode/decode without real VAE weights.
//!
//! ## Output contract
//!
//! - Input latent shape: `[batch, 4, latent_h, latent_w]`
//! - Output image shape: `[batch, 3, height, width]` where
//!   `height = latent_h * 8` and `width = latent_w * 8`
//! - Dtype: `F32`
//! - Color space: `rgb`
//! - Value range: backend decoder returns logits in approximately
//!   `[-1, 1]`; the operation contract requires `F32`, RGB, values
//!   clamped into `[0, 1]` before reaching `image.save`. The facade
//!   applies an affine remap (`output * 0.5 + 0.5`) so downstream PNG
//!   encoding clamps sensibly without surprises.
//!
//! ## Non-responsibilities
//!
//! - Does not load raw single-file checkpoint VAE weights directly
//!   (`first_stage_model.*`); that is the importer's job.
//! - Does not batch; V1 rejects `batch != 1` at the operation layer
//!   before reaching the VAE graph.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};

use crate::error::CandleBackendError;
use reimagine_inference::LatentSpaceMetadata;

use crate::store::{CandleImage, CandleLatent};

/// SDXL VAE latent scale factor (official constant).
///
/// The SDXL VAE encodes images into latents that are scaled by this
/// factor; the public `latent.decode` path divides the latent by
/// this factor before decoding.
const SDXL_VAE_SCALE_FACTOR: f32 = 0.18215;

/// SDXL VAE spatial upscale factor. Latent dimensions are multiplied
/// by this factor to obtain pixel dimensions.
const SDXL_VAE_SPATIAL_UPSCALE: usize = 8;

/// SDXL VAE in/out channels for the AutoEncoderKL graph.
const SDXL_VAE_IN_CHANNELS: usize = 3;
const SDXL_VAE_OUT_CHANNELS: usize = 3;
const SDXL_VAE_LATENT_CHANNELS: usize = 4;

/// Backend-private SDXL VAE configuration. Hard-coded for
/// `stable_diffusion/sdxl/base` per the V1 backend-private config
/// guidance; future variants should add their own backend-private
/// config here rather than exposing these fields through public DTOs.
fn sdxl_base_vae_config() -> AutoEncoderKLConfig {
    // https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/config.json
    AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: SDXL_VAE_LATENT_CHANNELS,
        norm_num_groups: 32,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

/// Backend-private error type for VAE loading and forward execution.
#[derive(Debug)]
pub enum SdxlVaeError {
    /// A Candle tensor / kernel operation failed.
    Tensor(String),
    /// The input latent had an invalid shape / dtype.
    Shape(String),
    /// Reading or parsing the VAE weight file failed, or the loaded
    /// tensor names did not match the SDXL base VAE target surface.
    WeightLoad(String),
    /// The VAE source is a checkpoint bundle; split VAE weights must
    /// be imported first via
    /// `import_sdxl_checkpoint_to_candle_example_split`.
    RequiresSplitImport { path: PathBuf },
}

impl std::fmt::Display for SdxlVaeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(msg) => write!(f, "VAE tensor error: {msg}"),
            Self::Shape(msg) => write!(f, "VAE shape error: {msg}"),
            Self::WeightLoad(msg) => write!(f, "VAE weight load error: {msg}"),
            Self::RequiresSplitImport { path } => write!(
                f,
                "SDXL VAE operations require a Candle-compatible split VAE source; only the original checkpoint `{}` is present. Run `import_sdxl_checkpoint_to_candle_example_split` first to produce `vae/model.safetensors` with bare Candle example keys, then re-supply it with `component=vae`",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SdxlVaeError {}

impl From<SdxlVaeError> for CandleBackendError {
    fn from(err: SdxlVaeError) -> Self {
        CandleBackendError::InvalidRequest(err.to_string())
    }
}

/// Backend-private real SDXL VAE.
///
/// Owns a Candle [`AutoEncoderKL`] and lazily constructed input
/// tensor metadata. Loaded once per graph via
/// [`SdxlRealVae::load`] and reused across encode/decode calls.
#[derive(Debug)]
pub struct SdxlRealVae {
    autoencoder: AutoEncoderKL,
    latent_scale: f32,
}

impl SdxlRealVae {
    /// Load real VAE weights from a Candle-compatible split VAE
    /// safetensors file. `source` must already have been validated
    /// by [`crate::models::stable_diffusion::sdxl::vae_sources`] and
    /// refer to a safetensors file with bare Candle example keys.
    pub fn load(source: &Path, device: &Device) -> Result<Self, SdxlVaeError> {
        let data = std::fs::read(source).map_err(|err| {
            SdxlVaeError::WeightLoad(format!(
                "failed to read SDXL VAE weights `{}`: {err}",
                source.display()
            ))
        })?;
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(data.clone(), DType::F32, device)
            .map_err(|err| {
                SdxlVaeError::WeightLoad(format!(
                    "failed to parse SDXL VAE safetensors `{}`: {err}",
                    source.display()
                ))
            })?;
        let autoencoder = AutoEncoderKL::new(
            vb,
            SDXL_VAE_IN_CHANNELS,
            SDXL_VAE_OUT_CHANNELS,
            sdxl_base_vae_config(),
        )
        .map_err(|err| {
            SdxlVaeError::WeightLoad(format!(
                "failed to materialize SDXL VAE from `{}`: {err}",
                source.display()
            ))
        })?;
        Ok(Self {
            autoencoder,
            latent_scale: SDXL_VAE_SCALE_FACTOR,
        })
    }

    /// Run real VAE decode on `latent`. Returns a backend-owned
    /// [`CandleImage`] in `[batch, 3, height, width]` F32 RGB.
    pub fn decode(
        &self,
        latent: &CandleLatent,
        device: &Device,
    ) -> Result<CandleImage, SdxlVaeError> {
        let tensor = latent.tensor();

        if tensor.dtype() != DType::F32 {
            return Err(SdxlVaeError::Tensor(format!(
                "real VAE decoder expects f32 input latent, got {:?}",
                tensor.dtype()
            )));
        }

        let dims = tensor.shape().dims();
        if dims.len() != 4 {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder expects 4D input latent [batch, 4, h, w], got {}-D shape {:?}",
                dims.len(),
                dims
            )));
        }

        let batch = dims[0];
        let channels = dims[1];
        let h = dims[2];
        let w = dims[3];

        if batch != 1 {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder V1 supports only batch=1, got batch={batch}"
            )));
        }
        if channels != SDXL_VAE_LATENT_CHANNELS {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder expects exactly {SDXL_VAE_LATENT_CHANNELS} latent channels, got {channels}"
            )));
        }
        if h == 0 || w == 0 {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder expects positive spatial dimensions h>0 and w>0, got h={h}, w={w}"
            )));
        }
        if !h.is_multiple_of(SDXL_VAE_SPATIAL_UPSCALE)
            || !w.is_multiple_of(SDXL_VAE_SPATIAL_UPSCALE)
        {
            // SDXL VAE always upsamples by 8; latent spatial
            // dimensions must yield integer pixel sizes after the
            // upscale. The latent space is already on integer grid
            // but guard against fractional overflow.
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder latent spatial dims must produce integer pixel sizes; h={h}, w={w}"
            )));
        }

        let tensor_on_device = if tensor.device().same_device(device) {
            tensor.clone()
        } else {
            tensor.to_device(device).map_err(|err| {
                SdxlVaeError::Tensor(format!("real VAE decoder device transfer failed: {err}"))
            })?
        };

        let scale_tensor = Tensor::new(self.latent_scale, device).map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "real VAE decoder failed to create latent scale tensor: {err}"
            ))
        })?;
        let scaled = tensor_on_device
            .broadcast_div(&scale_tensor)
            .map_err(|err| {
                SdxlVaeError::Tensor(format!(
                    "real VAE decoder latent scale division failed: {err}"
                ))
            })?;

        let decoded = self.autoencoder.decode(&scaled).map_err(|err| {
            SdxlVaeError::Tensor(format!("real VAE decoder forward failed: {err}"))
        })?;

        // The decoder outputs values in approximately [-1, 1].
        // Normalize to [0, 1] so downstream PNG encoding clamps
        // sensibly. `image.save` still clamps defensively when
        // converting to PNG bytes.
        let normalized = decoded.affine(0.5, 0.5).map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "real VAE decoder output normalization failed: {err}"
            ))
        })?;

        let out_dims = normalized.shape().dims();
        if out_dims.len() != 4 {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder expected 4D output image, got {}-D shape {:?}",
                out_dims.len(),
                out_dims
            )));
        }
        let out_batch = out_dims[0];
        let out_channels = out_dims[1];
        let out_h = out_dims[2];
        let out_w = out_dims[3];
        if out_channels != SDXL_VAE_OUT_CHANNELS {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder expected {SDXL_VAE_OUT_CHANNELS} RGB channels, got {out_channels}"
            )));
        }

        let expected_h = h * SDXL_VAE_SPATIAL_UPSCALE;
        let expected_w = w * SDXL_VAE_SPATIAL_UPSCALE;
        if out_h != expected_h || out_w != expected_w {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder output spatial dims mismatch: expected ({expected_h}, {expected_w}), got ({out_h}, {out_w})"
            )));
        }
        if out_batch != batch {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE decoder output batch mismatch: expected {batch}, got {out_batch}"
            )));
        }

        Ok(CandleImage::new(
            normalized,
            out_w as u32,
            out_h as u32,
            out_batch as u32,
            "rgb".to_string(),
        ))
    }

    /// Run real VAE encode on `image`. The input image is the
    /// backend-standard `[0, 1]` F32 RGB NCHW payload; this method
    /// remaps it to the VAE input domain, samples from Candle's
    /// latent distribution, and scales the latent by the SDXL VAE
    /// scale factor.
    ///
    /// Candle 0.10.2 exposes `DiagonalGaussianDistribution::sample()`
    /// but not a public mean accessor, so V1 encode is stochastic.
    pub fn encode(
        &self,
        image: &CandleImage,
        device: &Device,
        latent_space: LatentSpaceMetadata,
    ) -> Result<CandleLatent, SdxlVaeError> {
        let tensor = image.tensor();

        if tensor.dtype() != DType::F32 {
            return Err(SdxlVaeError::Tensor(format!(
                "real VAE encoder expects f32 input image, got {:?}",
                tensor.dtype()
            )));
        }

        let dims = tensor.shape().dims();
        if dims.len() != 4 {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder expects 4D input image [batch, 3, height, width], got {}-D shape {:?}",
                dims.len(),
                dims
            )));
        }

        let batch = dims[0];
        let channels = dims[1];
        let h = dims[2];
        let w = dims[3];
        if batch != 1 {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder V1 supports only batch=1, got batch={batch}"
            )));
        }
        if channels != SDXL_VAE_IN_CHANNELS {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder expects exactly {SDXL_VAE_IN_CHANNELS} RGB channels, got {channels}"
            )));
        }
        if h == 0 || w == 0 {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder expects positive spatial dimensions height>0 and width>0, got height={h}, width={w}"
            )));
        }
        if !h.is_multiple_of(SDXL_VAE_SPATIAL_UPSCALE)
            || !w.is_multiple_of(SDXL_VAE_SPATIAL_UPSCALE)
        {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder image spatial dims must be divisible by {SDXL_VAE_SPATIAL_UPSCALE}; got height={h}, width={w}"
            )));
        }

        let tensor_on_device = if tensor.device().same_device(device) {
            tensor.clone()
        } else {
            tensor.to_device(device).map_err(|err| {
                SdxlVaeError::Tensor(format!("real VAE encoder device transfer failed: {err}"))
            })?
        };

        let encoder_input = tensor_on_device.affine(2.0, -1.0).map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "real VAE encoder input normalization failed: {err}"
            ))
        })?;
        let distribution = self.autoencoder.encode(&encoder_input).map_err(|err| {
            SdxlVaeError::Tensor(format!("real VAE encoder forward failed: {err}"))
        })?;
        let sampled = distribution.sample().map_err(|err| {
            SdxlVaeError::Tensor(format!("real VAE encoder latent sampling failed: {err}"))
        })?;
        let scale_tensor = Tensor::new(self.latent_scale, device).map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "real VAE encoder failed to create latent scale tensor: {err}"
            ))
        })?;
        let scaled = sampled.broadcast_mul(&scale_tensor).map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "real VAE encoder latent scale multiplication failed: {err}"
            ))
        })?;

        let latent_dims = scaled.shape().dims();
        if latent_dims.len() != 4 {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder expected 4D output latent, got {}-D shape {:?}",
                latent_dims.len(),
                latent_dims
            )));
        }
        let latent_batch = latent_dims[0];
        let latent_channels = latent_dims[1];
        let latent_h = latent_dims[2];
        let latent_w = latent_dims[3];
        if latent_batch != batch {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder output batch mismatch: expected {batch}, got {latent_batch}"
            )));
        }
        if latent_channels != SDXL_VAE_LATENT_CHANNELS {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder expected {SDXL_VAE_LATENT_CHANNELS} latent channels, got {latent_channels}"
            )));
        }
        let expected_h = h / SDXL_VAE_SPATIAL_UPSCALE;
        let expected_w = w / SDXL_VAE_SPATIAL_UPSCALE;
        if latent_h != expected_h || latent_w != expected_w {
            return Err(SdxlVaeError::Shape(format!(
                "real VAE encoder output spatial dims mismatch: expected ({expected_h}, {expected_w}), got ({latent_h}, {latent_w})"
            )));
        }

        Ok(CandleLatent::new(scaled, latent_space))
    }
}

/// Backend-private graph facade for the SDXL VAE.
///
/// Production graphs run in [`SdxlVaeMode::Real`] mode, which loads
/// real VAE weights from a split VAE source and runs the actual VAE
/// forwards. Test-only placeholder mode is opt-in via
/// [`SdxlVaeGraph::test_placeholder`].
#[derive(Debug)]
pub struct SdxlVaeGraph {
    mode: SdxlVaeMode,
}

#[derive(Debug)]
enum SdxlVaeMode {
    Real {
        vae: Box<SdxlRealVae>,
    },
    /// Test-only mode used by graph/unit tests that cannot load real
    /// VAE weights. Not exposed in production. The shape contract
    /// matches the real decoder so consumers do not need to branch.
    #[allow(dead_code)]
    TestPlaceholder,
}

impl SdxlVaeGraph {
    /// Materialize a real VAE graph from a split VAE source path.
    /// Returns a precise error if the source cannot be loaded.
    pub fn load(source: &Path, device: &Device) -> Result<Self, SdxlVaeError> {
        let vae = SdxlRealVae::load(source, device)?;
        Ok(Self {
            mode: SdxlVaeMode::Real { vae: Box::new(vae) },
        })
    }

    /// Construct a test-only placeholder graph. Production code
    /// should never call this; tests that need to assert shape
    /// without real weights may use it.
    #[doc(hidden)]
    #[allow(dead_code)]
    pub fn test_placeholder() -> Self {
        Self {
            mode: SdxlVaeMode::TestPlaceholder,
        }
    }

    /// Run the VAE forward pass on `latent`. Returns a backend-owned
    /// [`CandleImage`].
    pub fn decode(
        &self,
        latent: &CandleLatent,
        device: &Device,
    ) -> Result<CandleImage, SdxlVaeError> {
        match &self.mode {
            SdxlVaeMode::Real { vae } => vae.as_ref().decode(latent, device),
            SdxlVaeMode::TestPlaceholder => test_placeholder_decode(latent, device),
        }
    }

    /// Run the VAE encode pass on `image`. Returns a backend-owned
    /// [`CandleLatent`]. The first real Candle path samples from
    /// the VAE distribution because Candle 0.10.2 does not expose a
    /// deterministic mean accessor.
    pub fn encode(
        &self,
        image: &CandleImage,
        device: &Device,
        latent_space: LatentSpaceMetadata,
    ) -> Result<CandleLatent, SdxlVaeError> {
        match &self.mode {
            SdxlVaeMode::Real { vae } => vae.as_ref().encode(image, device, latent_space),
            SdxlVaeMode::TestPlaceholder => test_placeholder_encode(image, device, latent_space),
        }
    }
}

#[doc(hidden)]
#[allow(dead_code)]
fn test_placeholder_decode(
    latent: &CandleLatent,
    device: &Device,
) -> Result<CandleImage, SdxlVaeError> {
    // Mirror the deterministic shape contract from the original
    // V1 placeholder decoder. Kept here so unit tests can continue
    // asserting shape/dtype semantics without loading real weights.
    let tensor = latent.tensor();

    if tensor.dtype() != DType::F32 {
        return Err(SdxlVaeError::Tensor(format!(
            "test placeholder VAE decoder expects f32 input latent, got {:?}",
            tensor.dtype()
        )));
    }

    let dims = tensor.shape().dims();
    if dims.len() != 4 {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE decoder expects 4D input latent [batch, 4, h, w], got {}-D shape {:?}",
            dims.len(),
            dims
        )));
    }

    let batch = dims[0];
    let channels = dims[1];
    let h = dims[2];
    let w = dims[3];

    if batch != 1 {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE decoder V1 supports only batch=1, got batch={batch}"
        )));
    }
    if channels != SDXL_VAE_LATENT_CHANNELS {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE decoder expects exactly {SDXL_VAE_LATENT_CHANNELS} latent channels, got {channels}"
        )));
    }
    if h == 0 || w == 0 {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE decoder expects positive spatial dimensions h>0 and w>0, got h={h}, w={w}"
        )));
    }

    let scale_tensor = Tensor::new(SDXL_VAE_SCALE_FACTOR, device).map_err(|err| {
        SdxlVaeError::Tensor(format!(
            "test placeholder failed to create scale tensor: {err}"
        ))
    })?;
    let scaled = tensor.broadcast_div(&scale_tensor).map_err(|err| {
        SdxlVaeError::Tensor(format!("test placeholder VAE scale division failed: {err}"))
    })?;

    let h_out = h * SDXL_VAE_SPATIAL_UPSCALE;
    let w_out = w * SDXL_VAE_SPATIAL_UPSCALE;
    let upscaled = scaled.upsample_nearest2d(h_out, w_out).map_err(|err| {
        SdxlVaeError::Tensor(format!("test placeholder VAE upsample failed: {err}"))
    })?;
    let rgb = upscaled
        .narrow(1, 0, SDXL_VAE_OUT_CHANNELS)
        .map_err(|err| {
            SdxlVaeError::Tensor(format!("test placeholder VAE channel slice failed: {err}"))
        })?;
    let remapped = rgb.affine(0.5, 0.5).map_err(|err| {
        SdxlVaeError::Tensor(format!("test placeholder VAE rescale failed: {err}"))
    })?;

    Ok(CandleImage::new(
        remapped,
        w_out as u32,
        h_out as u32,
        batch as u32,
        "rgb".to_string(),
    ))
}

#[doc(hidden)]
#[allow(dead_code)]
fn test_placeholder_encode(
    image: &CandleImage,
    device: &Device,
    latent_space: LatentSpaceMetadata,
) -> Result<CandleLatent, SdxlVaeError> {
    let tensor = image.tensor();

    if tensor.dtype() != DType::F32 {
        return Err(SdxlVaeError::Tensor(format!(
            "test placeholder VAE encoder expects f32 input image, got {:?}",
            tensor.dtype()
        )));
    }

    let dims = tensor.shape().dims();
    if dims.len() != 4 {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE encoder expects 4D input image [batch, 3, height, width], got {}-D shape {:?}",
            dims.len(),
            dims
        )));
    }

    let batch = dims[0];
    let channels = dims[1];
    let h = dims[2];
    let w = dims[3];
    if batch != 1 {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE encoder V1 supports only batch=1, got batch={batch}"
        )));
    }
    if channels != SDXL_VAE_IN_CHANNELS {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE encoder expects exactly {SDXL_VAE_IN_CHANNELS} RGB channels, got {channels}"
        )));
    }
    if h == 0 || w == 0 {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE encoder expects positive spatial dimensions height>0 and width>0, got height={h}, width={w}"
        )));
    }
    if !h.is_multiple_of(SDXL_VAE_SPATIAL_UPSCALE) || !w.is_multiple_of(SDXL_VAE_SPATIAL_UPSCALE) {
        return Err(SdxlVaeError::Shape(format!(
            "test placeholder VAE encoder image spatial dims must be divisible by {SDXL_VAE_SPATIAL_UPSCALE}; got height={h}, width={w}"
        )));
    }

    let tensor_on_device = if tensor.device().same_device(device) {
        tensor.clone()
    } else {
        tensor.to_device(device).map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "test placeholder VAE encoder device transfer failed: {err}"
            ))
        })?
    };
    let latent_h = h / SDXL_VAE_SPATIAL_UPSCALE;
    let latent_w = w / SDXL_VAE_SPATIAL_UPSCALE;
    let pooled = tensor_on_device
        .avg_pool2d(SDXL_VAE_SPATIAL_UPSCALE)
        .map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "test placeholder VAE encoder pooling failed: {err}"
            ))
        })?;
    let mean = pooled
        .mean_keepdim(1)
        .map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "test placeholder VAE encoder channel mean failed: {err}"
            ))
        })?
        .broadcast_as((batch, SDXL_VAE_LATENT_CHANNELS, latent_h, latent_w))
        .map_err(|err| {
            SdxlVaeError::Tensor(format!(
                "test placeholder VAE encoder latent channel broadcast failed: {err}"
            ))
        })?;
    let scale_tensor = Tensor::new(SDXL_VAE_SCALE_FACTOR, device).map_err(|err| {
        SdxlVaeError::Tensor(format!(
            "test placeholder failed to create scale tensor: {err}"
        ))
    })?;
    let scaled = mean.broadcast_mul(&scale_tensor).map_err(|err| {
        SdxlVaeError::Tensor(format!(
            "test placeholder VAE encoder latent scale multiplication failed: {err}"
        ))
    })?;

    Ok(CandleLatent::new(scaled, latent_space))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> &'static Device {
        &Device::Cpu
    }

    fn f32_latent(shape: &[usize]) -> CandleLatent {
        CandleLatent::new(
            Tensor::zeros(shape, DType::F32, cpu()).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        )
    }

    #[test]
    fn graph_test_placeholder_rejects_non_f32_input() {
        let graph = SdxlVaeGraph::test_placeholder();
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::U32, cpu()).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        let err = graph.decode(&latent, cpu()).unwrap_err();
        assert!(err.to_string().contains("f32"), "got: {}", err);
        assert!(matches!(err, SdxlVaeError::Tensor(_)));
    }

    #[test]
    fn graph_test_placeholder_rejects_wrong_channels() {
        let graph = SdxlVaeGraph::test_placeholder();
        let latent = f32_latent(&[1, 3, 8, 8]);
        let err = graph.decode(&latent, cpu()).unwrap_err();
        assert!(err.to_string().contains("4"), "got: {}", err);
        assert!(err.to_string().contains("channels"), "got: {}", err);
        assert!(matches!(err, SdxlVaeError::Shape(_)));
    }

    #[test]
    fn graph_test_placeholder_rejects_non_4d_input() {
        let graph = SdxlVaeGraph::test_placeholder();
        let latent = CandleLatent::new(
            Tensor::zeros((1, 4, 16), DType::F32, cpu()).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        let err = graph.decode(&latent, cpu()).unwrap_err();
        assert!(err.to_string().contains("4D"), "got: {}", err);
        assert!(matches!(err, SdxlVaeError::Shape(_)));
    }

    #[test]
    fn graph_test_placeholder_rejects_batch_greater_than_one() {
        let graph = SdxlVaeGraph::test_placeholder();
        let latent = f32_latent(&[2, 4, 8, 8]);
        let err = graph.decode(&latent, cpu()).unwrap_err();
        assert!(err.to_string().contains("batch=2"), "got: {}", err);
        assert!(matches!(err, SdxlVaeError::Shape(_)));
    }

    #[test]
    fn graph_test_placeholder_produces_correct_shape() {
        let graph = SdxlVaeGraph::test_placeholder();
        let latent = f32_latent(&[1, 4, 16, 16]);
        let image = graph.decode(&latent, cpu()).unwrap();

        assert_eq!(image.tensor().shape().dims(), &[1, 3, 128, 128]);
        assert_eq!(image.tensor().dtype(), DType::F32);
        assert_eq!(image.width(), 128);
        assert_eq!(image.height(), 128);
        assert_eq!(image.batch(), 1);
        assert_eq!(image.color_space(), "rgb");
    }

    #[test]
    fn graph_test_placeholder_encode_produces_correct_shape() {
        let graph = SdxlVaeGraph::test_placeholder();
        let image = CandleImage::new(
            Tensor::zeros((1, 3, 64, 64), DType::F32, cpu()).unwrap(),
            64,
            64,
            1,
            "rgb".to_string(),
        );
        let latent = graph
            .encode(&image, cpu(), LatentSpaceMetadata::sdxl_base())
            .unwrap();

        assert_eq!(latent.tensor().shape().dims(), &[1, 4, 8, 8]);
        assert_eq!(latent.tensor().dtype(), DType::F32);
        assert_eq!(latent.latent_space(), &LatentSpaceMetadata::sdxl_base());
    }

    #[test]
    fn graph_test_placeholder_output_values_in_unit_range() {
        let graph = SdxlVaeGraph::test_placeholder();
        let latent = f32_latent(&[1, 4, 8, 8]);
        let image = graph.decode(&latent, cpu()).unwrap();

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
    fn graph_test_placeholder_decode_is_deterministic_for_same_input() {
        let graph = SdxlVaeGraph::test_placeholder();
        let latent = f32_latent(&[1, 4, 8, 8]);

        let first = graph.decode(&latent, cpu()).unwrap().into_tensor();
        let second = graph.decode(&latent, cpu()).unwrap().into_tensor();

        let first_data = first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let second_data = second.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(first_data, second_data, "decoder must be deterministic");
    }

    #[test]
    fn real_vae_load_rejects_unrelated_safetensors() {
        // Production VAE operations cannot be constructed without
        // real VAE weights. Confirm the load path produces a precise
        // error when a non-VAE safetensors is supplied.
        use candle_core::safetensors;
        use std::collections::HashMap;

        let dir = std::env::temp_dir().join(format!(
            "reimagine-vae-load-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("not-a-vae.safetensors");

        let mut tensors = HashMap::new();
        let unrelated = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &Device::Cpu).unwrap();
        tensors.insert("unrelated.weight".to_string(), unrelated);
        safetensors::save(&tensors, &path).unwrap();

        let err = SdxlRealVae::load(&path, &Device::Cpu).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, SdxlVaeError::WeightLoad(_)));
        assert!(
            msg.contains("VAE weight load error"),
            "expected WeightLoad error, got {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn requires_split_import_error_message_is_actionable() {
        let err = SdxlVaeError::RequiresSplitImport {
            path: PathBuf::from("/models/sdxl.safetensors"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("import_sdxl_checkpoint_to_candle_example_split"),
            "msg: {msg}"
        );
        assert!(msg.contains("/models/sdxl.safetensors"), "msg: {msg}");
        assert!(msg.contains("component=vae"), "msg: {msg}");
    }
}
