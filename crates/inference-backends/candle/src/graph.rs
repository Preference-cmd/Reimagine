//! Backend-local model graph facade.
//!
//! `LoadedModelBundle` acts as a small enum-dispatch facade for model-family
//! specific computation. Standard capability methods in [`crate::operation`]
//! translate [`reimagine_inference`] requests into the backend-local
//! input types defined here, then call the matching facade method. The facade
//! dispatches to the concrete implementation for the loaded model family
//! (currently only `stable_diffusion/sdxl`) without exposing family-specific
//! kernel modules to the operation layer.
//!
//! V1 intentionally keeps this layer small: it is an enum-dispatch facade on
//! [`crate::models::LoadedModelBundle`] rather than a broad trait-object
//! plugin framework. Operation modules never import
//! `models/stable_diffusion/sdxl/*` directly; they only import the facade and
//! the backend-local input/result types in this module.

use candle_core::Device;
use reimagine_core::model::ModelRole;
use reimagine_inference::{
    BackendKind, LoadBundleResponse, RuntimeClipHandle, RuntimeModelHandle, RuntimeVaeHandle,
};

use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;
use crate::models::stable_diffusion::sdxl::diffusion::{SdxlSampleRequest, SdxlSampler};
use crate::models::stable_diffusion::sdxl::text::SdxlTextEncoder;
use crate::models::stable_diffusion::sdxl::vae::SdxlVaeDecoder;
use crate::store::{CandleImage, CandleLatent};

/// Backend-local input for text encoding.
pub struct TextEncodeInput {
    pub prompt: String,
}

/// Backend-local result of text encoding.
///
/// The raw text and pooled embedding tensors are returned to the operation
/// layer so it can store the conditioning payload and build the lightweight
/// runtime handle.
#[derive(Debug)]
pub struct TextEncodeResult {
    pub text_embedding: candle_core::Tensor,
    pub pooled_embedding: candle_core::Tensor,
}

/// Backend-local sampler parameters extracted from a generic
/// [`reimagine_inference::DiffusionSampleRequest`].
pub struct DiffusionSampleInput {
    pub seed: u64,
    pub steps: u32,
    pub cfg: f32,
    pub sampler_name: String,
    pub scheduler_name: String,
    pub denoise: f32,
}

/// Backend-local result of diffusion sampling.
#[derive(Debug)]
pub struct DiffusionSampleResult {
    pub latent: CandleLatent,
}

/// Backend-local input for latent decoding.
pub struct LatentDecodeInput {
    pub latent: CandleLatent,
}

/// Backend-local result of latent decoding.
#[derive(Debug)]
pub struct LatentDecodeResult {
    pub image: CandleImage,
}

impl LoadedModelBundle {
    /// Validate that `clip` points at the loaded bundle's text encoder payload.
    pub fn validate_clip_handle(&self, clip: &RuntimeClipHandle) -> Result<(), CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
                if clip.payload_key() != &sdxl.clip_payload_key {
                    return Err(CandleBackendError::InvalidRequest(format!(
                        "text.encode `clip` payload `{}` does not match loaded {} CLIP payload `{}` for model `{}`",
                        clip.payload_key().as_str(),
                        self.family_label(),
                        sdxl.clip_payload_key.as_str(),
                        sdxl.model_id.as_str()
                    )));
                }
                Ok(())
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Ok(()),
        }
    }

    /// Validate that `model` points at the loaded bundle's diffusion payload.
    pub fn validate_model_handle(
        &self,
        model: &RuntimeModelHandle,
    ) -> Result<(), CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
                if model.payload_key() != &sdxl.model_payload_key {
                    return Err(CandleBackendError::InvalidRequest(format!(
                        "diffusion.sample `model` payload `{}` does not match loaded {} model payload `{}` for model `{}`",
                        model.payload_key().as_str(),
                        self.family_label(),
                        sdxl.model_payload_key.as_str(),
                        sdxl.model_id.as_str()
                    )));
                }
                Ok(())
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Ok(()),
        }
    }

    /// Validate that `vae` points at the loaded bundle's decoder payload.
    pub fn validate_vae_handle(&self, vae: &RuntimeVaeHandle) -> Result<(), CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
                if vae.payload_key() != &sdxl.vae_payload_key {
                    return Err(CandleBackendError::InvalidRequest(format!(
                        "latent.decode `vae` payload `{}` does not match loaded {} VAE payload `{}` for model `{}`",
                        vae.payload_key().as_str(),
                        self.family_label(),
                        sdxl.vae_payload_key.as_str(),
                        sdxl.model_id.as_str()
                    )));
                }
                Ok(())
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Ok(()),
        }
    }

    /// Build the standard `model.load_bundle` response for this loaded bundle.
    pub fn load_bundle_response(
        &self,
        backend_kind: BackendKind,
        device_label: &str,
    ) -> Result<LoadBundleResponse, CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
                let model = RuntimeModelHandle::new(
                    sdxl.model_id.clone(),
                    ModelRole::CheckpointBundle,
                    backend_kind.clone(),
                    sdxl.model_payload_key.clone(),
                )
                .with_device(device_label);
                let clip = RuntimeClipHandle::new(
                    sdxl.model_id.clone(),
                    backend_kind.clone(),
                    sdxl.clip_payload_key.clone(),
                )
                .with_device(device_label);
                let vae = RuntimeVaeHandle::new(
                    sdxl.model_id.clone(),
                    backend_kind,
                    sdxl.vae_payload_key.clone(),
                )
                .with_device(device_label);
                Ok(LoadBundleResponse::new(model, clip, vae))
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Err(CandleBackendError::InvalidRequest(
                "test placeholder bundles cannot be loaded through model.load_bundle".to_string(),
            )),
        }
    }

    /// Encode `prompt` into text/pooled embedding tensors for the loaded model
    /// family.
    ///
    /// The operation layer is responsible for storing the resulting tensors as
    /// a [`crate::store::CandleConditioning`] payload and building the runtime
    /// handle.
    pub fn encode_text(
        &self,
        input: TextEncodeInput,
        device: &Device,
    ) -> Result<TextEncodeResult, CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(_) => {
                let encoder = SdxlTextEncoder::new();
                let (text_embedding, pooled_embedding) = encoder.encode(&input.prompt, device)?;
                Ok(TextEncodeResult {
                    text_embedding,
                    pooled_embedding,
                })
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Err(CandleBackendError::InvalidRequest(
                "text.encode is not supported by the test placeholder bundle".to_string(),
            )),
        }
    }

    /// Validate sampler parameters without touching stored payloads.
    ///
    /// The operation layer calls this early so model-family-specific request
    /// errors (unknown sampler, unsupported scheduler, out-of-range denoise,
    /// etc.) are reported before expensive store lookups.
    pub fn validate_sample_input(
        &self,
        input: &DiffusionSampleInput,
    ) -> Result<(), CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(_) => {
                // `SdxlSampleRequest::new` performs the V1 validation.
                let _ = SdxlSampleRequest::new(
                    input.seed,
                    input.steps,
                    input.cfg,
                    &input.sampler_name,
                    &input.scheduler_name,
                    input.denoise,
                )?;
                Ok(())
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Err(CandleBackendError::InvalidRequest(
                "diffusion.sample is not supported by the test placeholder bundle".to_string(),
            )),
        }
    }

    /// Sample a latent tensor from `input_latent` using the loaded model
    /// family's sampler.
    ///
    /// The operation layer is responsible for storing the resulting latent as
    /// a [`crate::store::CandleLatent`] payload and building the runtime
    /// handle. The caller should pre-validate `input` via
    /// [`Self::validate_sample_input`] so request-level errors are surfaced
    /// before expensive store/model work.
    pub fn sample_diffusion(
        &self,
        input: DiffusionSampleInput,
        input_latent: CandleLatent,
        device: &Device,
    ) -> Result<DiffusionSampleResult, CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(_) => {
                let request = SdxlSampleRequest::new(
                    input.seed,
                    input.steps,
                    input.cfg,
                    input.sampler_name,
                    input.scheduler_name,
                    input.denoise,
                )?;
                let sampler = SdxlSampler::new();
                let result = sampler.sample(input_latent, &request, device)?;
                Ok(DiffusionSampleResult {
                    latent: result.latent,
                })
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Err(CandleBackendError::InvalidRequest(
                "diffusion.sample is not supported by the test placeholder bundle".to_string(),
            )),
        }
    }

    /// Decode `latent` into an image tensor for the loaded model family.
    ///
    /// The operation layer is responsible for storing the resulting image as a
    /// [`crate::store::CandleImage`] payload and building the runtime handle.
    pub fn decode_latent(
        &self,
        input: LatentDecodeInput,
        device: &Device,
    ) -> Result<LatentDecodeResult, CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(_) => {
                let decoder = SdxlVaeDecoder::new();
                let image = decoder.decode(&input.latent, device)?;
                Ok(LatentDecodeResult { image })
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Err(CandleBackendError::InvalidRequest(
                "latent.decode is not supported by the test placeholder bundle".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use reimagine_core::model::ModelId;
    use reimagine_inference::ModelFormat;
    use std::sync::Arc;

    fn cpu_device() -> Arc<Device> {
        Arc::new(Device::Cpu)
    }

    fn sdxl_bundle() -> LoadedModelBundle {
        let model_id = ModelId::new("sdxl-test");
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("reimagine-graph-test-{nonce}"));
        std::fs::create_dir_all(&dir).expect("test temp dir");
        let source_path = dir.join("model.safetensors");
        // Create a dummy file so bundle construction succeeds.
        std::fs::write(&source_path, b"placeholder").expect("test placeholder file");
        let bundle = crate::models::LoadedSdxlBundle::from_resolved(
            model_id,
            source_path,
            ModelFormat::SafeTensors,
            cpu_device(),
        )
        .expect("test bundle");
        LoadedModelBundle::StableDiffusionSdxl(bundle)
    }

    #[test]
    fn encode_text_sdxl_produces_correct_shapes() {
        let bundle = sdxl_bundle();
        let result = bundle
            .encode_text(
                TextEncodeInput {
                    prompt: "a test prompt".to_string(),
                },
                &Device::Cpu,
            )
            .expect("encode_text should succeed");
        assert_eq!(
            result.text_embedding.shape().dims(),
            &[1, 77, 2048],
            "text embedding shape should be [1, 77, 2048]"
        );
        assert_eq!(
            result.pooled_embedding.shape().dims(),
            &[1, 1280],
            "pooled embedding shape should be [1, 1280]"
        );
    }

    #[test]
    fn sample_diffusion_sdxl_preserves_shape() {
        let bundle = sdxl_bundle();
        let input_latent =
            CandleLatent::new(Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap());
        let result = bundle
            .sample_diffusion(
                DiffusionSampleInput {
                    seed: 42,
                    steps: 10,
                    cfg: 7.0,
                    sampler_name: "euler".to_string(),
                    scheduler_name: "normal".to_string(),
                    denoise: 1.0,
                },
                input_latent,
                &Device::Cpu,
            )
            .expect("sample_diffusion should succeed");
        assert_eq!(result.latent.dims(), vec![1, 4, 8, 8]);
    }

    #[test]
    fn decode_latent_sdxl_produces_image_shape() {
        let bundle = sdxl_bundle();
        let input_latent =
            CandleLatent::new(Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap());
        let result = bundle
            .decode_latent(
                LatentDecodeInput {
                    latent: input_latent,
                },
                &Device::Cpu,
            )
            .expect("decode_latent should succeed");
        assert_eq!(result.image.tensor().shape().dims(), &[1, 3, 64, 64]);
        assert_eq!(result.image.width(), 64);
        assert_eq!(result.image.height(), 64);
    }

    #[test]
    fn encode_text_placeholder_returns_precise_error() {
        let bundle = LoadedModelBundle::TestPlaceholder;
        let err = bundle
            .encode_text(
                TextEncodeInput {
                    prompt: "test".to_string(),
                },
                &Device::Cpu,
            )
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("text.encode") && msg.contains("test placeholder bundle"),
            "expected precise unsupported-bundle diagnostic, got {msg}"
        );
    }

    #[test]
    fn sample_diffusion_placeholder_returns_precise_error() {
        let bundle = LoadedModelBundle::TestPlaceholder;
        let input_latent =
            CandleLatent::new(Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap());
        let err = bundle
            .sample_diffusion(
                DiffusionSampleInput {
                    seed: 0,
                    steps: 1,
                    cfg: 1.0,
                    sampler_name: "euler".to_string(),
                    scheduler_name: "normal".to_string(),
                    denoise: 1.0,
                },
                input_latent,
                &Device::Cpu,
            )
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("diffusion.sample") && msg.contains("test placeholder bundle"),
            "expected precise unsupported-bundle diagnostic, got {msg}"
        );
    }

    #[test]
    fn decode_latent_placeholder_returns_precise_error() {
        let bundle = LoadedModelBundle::TestPlaceholder;
        let input_latent =
            CandleLatent::new(Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap());
        let err = bundle
            .decode_latent(
                LatentDecodeInput {
                    latent: input_latent,
                },
                &Device::Cpu,
            )
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("latent.decode") && msg.contains("test placeholder bundle"),
            "expected precise unsupported-bundle diagnostic, got {msg}"
        );
    }
}
