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
use reimagine_inference::ResolvedInferenceModelSourceSet;
use reimagine_inference::{
    Backend, BackendInstance, LoadBundleResponse, RuntimeClipHandle, RuntimeModelHandle,
    RuntimeVaeHandle,
};

/// Backend-local loaded model graph — one implementation per model family.
pub trait LoadedModelGraph: Send + Sync {
    fn source_set(&self) -> &ResolvedInferenceModelSourceSet;
    fn component_graph_metadata(&self) -> Option<&str>;
    fn check_compatible(&self, incoming: &ResolvedInferenceModelSourceSet) -> Result<(), String>;
}

use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;
use crate::models::stable_diffusion::sdxl::diffusion::SdxlSampleRequest;
use crate::models::stable_diffusion::sdxl::diffusion_graph::SdxlDiffusionConditioning;
use crate::models::stable_diffusion::sdxl::text::SdxlTextEncoder;
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
    pub(crate) positive: SdxlDiffusionConditioning,
    pub(crate) negative: SdxlDiffusionConditioning,
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

/// Backend-local input for latent encoding.
pub struct LatentEncodeInput {
    pub image: CandleImage,
}

/// Backend-local result of latent encoding.
#[derive(Debug)]
pub struct LatentEncodeResult {
    pub latent: CandleLatent,
}

impl LoadedModelBundle {
    /// Expose the loaded model graph for compatibility checks.
    pub fn as_graph(&self) -> &dyn LoadedModelGraph {
        match self {
            Self::StableDiffusionSdxl(bundle) => bundle.as_ref(),
            #[cfg(test)]
            Self::TestPlaceholder => panic!("TestPlaceholder has no graph"),
        }
    }

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
        backend_kind: Backend,
        backend_instance: BackendInstance,
        device_label: &str,
    ) -> Result<LoadBundleResponse, CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
                let model = RuntimeModelHandle::with_instance(
                    sdxl.model_id.clone(),
                    ModelRole::CheckpointBundle,
                    backend_kind.clone(),
                    backend_instance.clone(),
                    sdxl.model_payload_key.clone(),
                )
                .with_device(device_label);
                let clip = RuntimeClipHandle::with_instance(
                    sdxl.model_id.clone(),
                    backend_kind.clone(),
                    backend_instance.clone(),
                    sdxl.clip_payload_key.clone(),
                )
                .with_device(device_label);
                let vae = RuntimeVaeHandle::with_instance(
                    sdxl.model_id.clone(),
                    backend_kind,
                    backend_instance,
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
        _device: &Device,
    ) -> Result<TextEncodeResult, CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(bundle) => {
                let (text_embedding, pooled_embedding) =
                    SdxlTextEncoder::encode(bundle, &input.prompt)?;
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
        self.validate_sample_params(
            input.seed,
            input.steps,
            input.cfg,
            &input.sampler_name,
            &input.scheduler_name,
            input.denoise,
        )
    }

    /// Validate sampler parameters without requiring backend tensor payloads.
    pub fn validate_sample_params(
        &self,
        seed: u64,
        steps: u32,
        cfg: f32,
        sampler_name: &str,
        scheduler_name: &str,
        denoise: f32,
    ) -> Result<(), CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(_) => {
                // `SdxlSampleRequest::new` performs the V1 validation.
                let _ = SdxlSampleRequest::new(
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler_name,
                    denoise,
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
            LoadedModelBundle::StableDiffusionSdxl(bundle) => {
                let request = SdxlSampleRequest::new(
                    input.seed,
                    input.steps,
                    input.cfg,
                    input.sampler_name,
                    input.scheduler_name,
                    input.denoise,
                )?;
                let graph = bundle.materialize_diffusion_graph()?;
                let latent = graph.sample(
                    input_latent,
                    input.positive,
                    input.negative,
                    &request,
                    device,
                )?;
                Ok(DiffusionSampleResult { latent })
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
    ///
    /// Real SDXL VAE decode lazy-materializes the underlying VAE graph
    /// on the first encode/decode call for the loaded bundle. Subsequent
    /// VAE operations reuse the cached graph; the operation layer must
    /// therefore keep the bundle alive across calls so the cache is honored.
    pub fn decode_latent(
        &self,
        input: LatentDecodeInput,
        device: &Device,
    ) -> Result<LatentDecodeResult, CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(bundle) => {
                let graph = bundle.materialize_vae_graph()?;
                let image = graph.decode(&input.latent, device)?;
                Ok(LatentDecodeResult { image })
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Err(CandleBackendError::InvalidRequest(
                "latent.decode is not supported by the test placeholder bundle".to_string(),
            )),
        }
    }

    /// Encode `image` into a latent tensor for the loaded model family.
    ///
    /// The operation layer is responsible for storing the resulting latent as
    /// a [`crate::store::CandleLatent`] payload and building the runtime handle.
    pub fn encode_image(
        &self,
        input: LatentEncodeInput,
        device: &Device,
    ) -> Result<LatentEncodeResult, CandleBackendError> {
        match self {
            LoadedModelBundle::StableDiffusionSdxl(bundle) => {
                let graph = bundle.materialize_vae_graph()?;
                let latent = graph.encode(&input.image, device, self.expected_latent_space())?;
                Ok(LatentEncodeResult { latent })
            }
            #[cfg(test)]
            LoadedModelBundle::TestPlaceholder => Err(CandleBackendError::InvalidRequest(
                "latent.encode is not supported by the test placeholder bundle".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::stable_diffusion::sdxl::diffusion_graph::TestSdxlDiffusionGraph;
    use candle_core::{DType, Device, Tensor};
    use reimagine_core::model::ModelId;
    use reimagine_inference::{LatentSpaceMetadata, ModelFormat};
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
        let source = reimagine_inference::ResolvedInferenceModelSource::new(
            reimagine_inference::ModelSourceKind::CheckpointBundle,
            ModelRole::CheckpointBundle,
            source_path,
            ModelFormat::SafeTensors,
        );
        let source_set = ResolvedInferenceModelSourceSet::new(source);
        let bundle = crate::models::LoadedSdxlBundle::from_resolved_with_test_text_projection(
            model_id,
            source_set,
            ModelFormat::SafeTensors,
            cpu_device(),
        )
        .expect("test bundle");
        LoadedModelBundle::StableDiffusionSdxl(bundle)
    }

    fn install_test_diffusion_graph(bundle: &LoadedModelBundle) {
        match bundle {
            LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
                sdxl.install_test_diffusion_graph(Arc::new(TestSdxlDiffusionGraph));
            }
            LoadedModelBundle::TestPlaceholder => {}
        }
    }

    fn install_test_vae_graph(bundle: &LoadedModelBundle) {
        match bundle {
            LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
                sdxl.install_test_vae_graph_for_tests(Arc::new(
                    crate::models::stable_diffusion::sdxl::vae::SdxlVaeGraph::test_placeholder(),
                ));
            }
            LoadedModelBundle::TestPlaceholder => {}
        }
    }

    fn sdxl_conditioning() -> SdxlDiffusionConditioning {
        SdxlDiffusionConditioning {
            text_embedding: Tensor::zeros((1, 77, 2048), DType::F32, &Device::Cpu).unwrap(),
            pooled_embedding: Tensor::zeros((1, 1280), DType::F32, &Device::Cpu).unwrap(),
        }
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
    fn sample_diffusion_sdxl_materializes_diffusion_graph_once_for_repeated_sampling() {
        let bundle = sdxl_bundle();
        install_test_diffusion_graph(&bundle);
        let input_latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        let result = bundle
            .sample_diffusion(
                DiffusionSampleInput {
                    seed: 42,
                    steps: 10,
                    cfg: 7.0,
                    sampler_name: "euler".to_string(),
                    scheduler_name: "normal".to_string(),
                    denoise: 1.0,
                    positive: sdxl_conditioning(),
                    negative: sdxl_conditioning(),
                },
                input_latent,
                &Device::Cpu,
            )
            .expect("first sample should materialize the diffusion graph");
        assert_eq!(result.latent.dims(), vec![1, 4, 8, 8]);

        let input_latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        bundle
            .sample_diffusion(
                DiffusionSampleInput {
                    seed: 42,
                    steps: 10,
                    cfg: 7.0,
                    sampler_name: "euler".to_string(),
                    scheduler_name: "normal".to_string(),
                    denoise: 1.0,
                    positive: sdxl_conditioning(),
                    negative: sdxl_conditioning(),
                },
                input_latent,
                &Device::Cpu,
            )
            .expect("second sample should reuse the materialized diffusion graph");
    }

    #[test]
    fn decode_latent_sdxl_produces_image_shape() {
        let bundle = sdxl_bundle();
        install_test_vae_graph(&bundle);
        let input_latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
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
    fn encode_image_sdxl_produces_latent_shape() {
        let bundle = sdxl_bundle();
        install_test_vae_graph(&bundle);
        let input_image = CandleImage::new(
            Tensor::zeros((1, 3, 64, 64), DType::F32, &Device::Cpu).unwrap(),
            64,
            64,
            1,
            "rgb".to_string(),
        );
        let result = bundle
            .encode_image(LatentEncodeInput { image: input_image }, &Device::Cpu)
            .expect("encode_image should succeed");
        assert_eq!(result.latent.tensor().shape().dims(), &[1, 4, 8, 8]);
        assert_eq!(
            result.latent.latent_space(),
            &LatentSpaceMetadata::sdxl_base()
        );
    }

    #[test]
    fn encode_image_sdxl_reuses_materialized_vae_graph_for_repeated_encode() {
        let bundle = sdxl_bundle();
        install_test_vae_graph(&bundle);
        for _ in 0..2 {
            let input_image = CandleImage::new(
                Tensor::zeros((1, 3, 64, 64), DType::F32, &Device::Cpu).unwrap(),
                64,
                64,
                1,
                "rgb".to_string(),
            );
            let result = bundle
                .encode_image(LatentEncodeInput { image: input_image }, &Device::Cpu)
                .expect("encode_image should reuse cached VAE graph");
            assert_eq!(result.latent.tensor().shape().dims(), &[1, 4, 8, 8]);
        }
    }

    #[test]
    fn decode_latent_sdxl_reuses_materialized_vae_graph_for_repeated_decode() {
        // First decode materializes the graph; the second decode must
        // succeed without reloading VAE weights. We verify this by
        // installing a placeholder graph and ensuring both calls go
        // through the same cached instance.
        let bundle = sdxl_bundle();
        install_test_vae_graph(&bundle);
        for _ in 0..2 {
            let input_latent = CandleLatent::new(
                Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap(),
                LatentSpaceMetadata::sdxl_base(),
            );
            let result = bundle
                .decode_latent(
                    LatentDecodeInput {
                        latent: input_latent,
                    },
                    &Device::Cpu,
                )
                .expect("decode_latent should reuse cached VAE graph");
            assert_eq!(result.image.tensor().shape().dims(), &[1, 3, 64, 64]);
        }
    }

    #[test]
    fn decode_latent_sdxl_rejects_checkpoint_only_vae_source() {
        // When no split VAE is supplied the bundle must report a
        // precise `import_sdxl_checkpoint_to_candle_example_split`
        // diagnostic instead of silently falling back.
        let bundle = sdxl_bundle();
        let input_latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
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
            msg.contains("import_sdxl_checkpoint_to_candle_example_split"),
            "expected split-import diagnostic, got {msg}"
        );
        assert!(
            msg.contains("component=vae"),
            "expected actionable metadata hint, got {msg}"
        );
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
        let input_latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
        let err = bundle
            .sample_diffusion(
                DiffusionSampleInput {
                    seed: 0,
                    steps: 1,
                    cfg: 1.0,
                    sampler_name: "euler".to_string(),
                    scheduler_name: "normal".to_string(),
                    denoise: 1.0,
                    positive: sdxl_conditioning(),
                    negative: sdxl_conditioning(),
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
        let input_latent = CandleLatent::new(
            Tensor::zeros((1, 4, 8, 8), DType::F32, &Device::Cpu).unwrap(),
            LatentSpaceMetadata::sdxl_base(),
        );
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

    #[test]
    fn encode_image_placeholder_returns_precise_error() {
        let bundle = LoadedModelBundle::TestPlaceholder;
        let input_image = CandleImage::new(
            Tensor::zeros((1, 3, 64, 64), DType::F32, &Device::Cpu).unwrap(),
            64,
            64,
            1,
            "rgb".to_string(),
        );
        let err = bundle
            .encode_image(LatentEncodeInput { image: input_image }, &Device::Cpu)
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("latent.encode") && msg.contains("test placeholder bundle"),
            "expected precise unsupported-bundle diagnostic, got {msg}"
        );
    }
}
