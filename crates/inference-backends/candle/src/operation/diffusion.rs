//! `diffusion.sample` operation.
//!
//! Translates a [`DiffusionSampleRequest`] into a backend-local sampler
//! call through the loaded-bundle graph facade and returns a
//! [`DiffusionSampleResponse`] carrying a `RuntimeLatent` handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! Model-family-specific sampling and UNet work live behind the loaded-bundle
//! graph facade.
//!
use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_inference::{
    BackendPayloadKey, BackendTensorHandle, DiffusionSampleRequest, DiffusionSampleResponse,
    ExecutionConditioning, InferenceBackend, RuntimeLatent, RuntimeModelHandle, SamplerName,
    SchedulerName,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::graph::{DiffusionSampleInput, DiffusionSampleResult};

pub fn execute_diffusion_sample(
    request: DiffusionSampleRequest,
    backend: &CandleBackend,
) -> Result<DiffusionSampleResponse, CandleBackendError> {
    let model_handle = require_model_handle(request.model(), backend)?;
    let positive_handle = require_conditioning_handle(request.positive(), backend)?;
    let negative_handle = require_conditioning_handle(request.negative(), backend)?;
    let latent_handle = require_latent_handle(request.latent(), backend)?;
    require_backend_instance_affinity(&request, backend)?;
    let _denoise_mode = request
        .denoise_mode()
        .map_err(|err| CandleBackendError::InvalidRequest(err.to_string()))?;

    let bundle = backend
        .model_cache()
        .get_bundle(model_handle.model_id())
        .ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample `model` handle points to model `{}` with no loaded bundle; load the model first via model.load_bundle",
                model_handle.model_id().as_str()
            ))
        })?;

    bundle.validate_model_handle(&model_handle)?;

    let sampler_name = sampler_name_from(request.sampler());
    let scheduler_name = scheduler_name_from(request.scheduler());
    bundle.validate_sample_params(
        request.seed(),
        request.steps(),
        request.cfg(),
        &sampler_name,
        &scheduler_name,
        request.denoise(),
    )?;

    let positive = backend
        .store()
        .get_conditioning(positive_handle.text_embedding().payload_key())?;
    let negative = backend
        .store()
        .get_conditioning(negative_handle.text_embedding().payload_key())?;

    let input_latent = backend
        .store()
        .get_latent(latent_handle.payload().payload_key())?;

    if !input_latent
        .latent_space()
        .is_compatible(latent_handle.latent_space())
    {
        return Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample stored latent payload `{}` latent space `{}` disagrees with latent handle latent space `{}`",
            latent_handle.payload().payload_key().as_str(),
            input_latent.latent_space().id(),
            latent_handle.latent_space().id(),
        )));
    }

    // Validate the input latent space against the loaded bundle's
    // expected latent space before any sampling tensor work. This
    // is a precise request-time rejection: incompatible latents
    // never reach the UNet.
    let expected = bundle.expected_latent_space();
    if !latent_handle.latent_space().is_compatible(&expected) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample input latent space `{}` (channels={}, scale={}) is incompatible with loaded {} model expected latent space `{}` (channels={}, scale={})",
            latent_handle.latent_space().id(),
            latent_handle.latent_space().channels(),
            latent_handle.latent_space().spatial_scale_factor(),
            bundle.family_label(),
            expected.id(),
            expected.channels(),
            expected.spatial_scale_factor(),
        )));
    }
    if !input_latent.latent_space().is_compatible(&expected) {
        return Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample stored latent payload `{}` latent space `{}` disagrees with loaded {} model expected latent space `{}`",
            latent_handle.payload().payload_key().as_str(),
            input_latent.latent_space().id(),
            bundle.family_label(),
            expected.id(),
        )));
    }

    let sample_input = DiffusionSampleInput {
        seed: request.seed(),
        steps: request.steps(),
        cfg: request.cfg(),
        sampler_name,
        scheduler_name,
        denoise: request.denoise(),
        expected_batch: latent_handle.batch(),
        positive,
        negative,
    };

    // Validate model-family-specific sampler parameters before sampling so
    // request-level errors are reported before graph materialization.
    bundle.validate_sample_input(&sample_input)?;

    let DiffusionSampleResult { latent } =
        bundle.sample_diffusion(sample_input, input_latent, backend.device().as_ref())?;

    let payload_key = BackendPayloadKey::new(format!(
        "latent:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    let output_dims = latent.dims();

    backend.store().insert_latent(
        request.run_id().clone(),
        payload_key.clone(),
        latent.into_tensor(),
        expected.clone(),
    );

    let latent_width = latent_handle.width();
    let latent_height = latent_handle.height();
    let latent_batch = latent_handle.batch();
    let latent_channels = latent_handle.channels();

    let latent = RuntimeLatent::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            payload_key,
            TensorDType::F32,
            TensorShape::new(output_dims),
            backend.device_label(),
        ),
        latent_width,
        latent_height,
        latent_batch,
        latent_channels,
        expected,
        reimagine_inference::LatentContent::Sampled,
    );

    Ok(DiffusionSampleResponse::new(latent))
}

fn require_backend_instance_affinity(
    request: &DiffusionSampleRequest,
    backend: &CandleBackend,
) -> Result<(), CandleBackendError> {
    let expected = backend.backend_instance();
    for instance in request.backend_affinities() {
        if instance != expected {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample handle backend instance `{}` does not match executing backend instance `{}`",
                instance.as_str(),
                expected.as_str()
            )));
        }
    }
    Ok(())
}

fn sampler_name_from(name: &SamplerName) -> String {
    name.as_str().to_string()
}

fn scheduler_name_from(name: &SchedulerName) -> String {
    name.as_str().to_string()
}

fn require_model_handle(
    handle: &RuntimeModelHandle,
    backend: &CandleBackend,
) -> Result<RuntimeModelHandle, CandleBackendError> {
    if handle.backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample `model` handle belongs to backend `{}`, expected `{}`",
            handle.backend().as_str(),
            backend.backend_kind()
        )));
    }
    Ok(handle.clone())
}

fn require_conditioning_handle(
    handle: &ExecutionConditioning,
    backend: &CandleBackend,
) -> Result<ExecutionConditioning, CandleBackendError> {
    if handle.text_embedding().backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample conditioning handle belongs to backend `{}`, expected `{}`",
            handle.text_embedding().backend().as_str(),
            backend.backend_kind()
        )));
    }
    if let Some(pooled) = handle.pooled_embedding() {
        if pooled.backend() != backend.backend_kind() {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample pooled conditioning handle belongs to backend `{}`, expected `{}`",
                pooled.backend().as_str(),
                backend.backend_kind()
            )));
        }
        if pooled.payload_key() != handle.text_embedding().payload_key() {
            return Err(CandleBackendError::InvalidRequest(format!(
                "diffusion.sample pooled conditioning payload `{}` does not match text conditioning payload `{}`",
                pooled.payload_key().as_str(),
                handle.text_embedding().payload_key().as_str()
            )));
        }
    }
    Ok(handle.clone())
}

fn require_latent_handle(
    handle: &RuntimeLatent,
    backend: &CandleBackend,
) -> Result<RuntimeLatent, CandleBackendError> {
    if handle.payload().backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample `latent` handle belongs to backend `{}`, expected `{}`",
            handle.payload().backend().as_str(),
            backend.backend_kind()
        )));
    }
    Ok(handle.clone())
}
