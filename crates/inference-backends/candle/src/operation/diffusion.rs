//! `diffusion.sample` operation.
//!
//! Translates a [`DiffusionSampleRequest`] into a backend-local sampler
//! call through the loaded-bundle graph facade and returns a
//! [`DiffusionSampleResponse`] carrying a `RuntimeLatent` handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! SDXL-specific tokenization, sampling, and UNet work live behind the
//! facade in `models/stable_diffusion/sdxl/diffusion.rs`.
//!
//! V1 uses a deterministic placeholder sampler so the backend
//! contract is testable without real UNet weights. The
//! `diffusion.sample` call still produces a real latent tensor that
//! flows through the existing typed `CandleStore` accessors.

use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_inference_core::{
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

    let sample_input = DiffusionSampleInput {
        seed: request.seed(),
        steps: request.steps(),
        cfg: request.cfg(),
        sampler_name: sampler_name_from(request.sampler()),
        scheduler_name: scheduler_name_from(request.scheduler()),
        denoise: request.denoise(),
    };

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

    // Validate model-family-specific sampler parameters before touching
    // stored payloads so request-level errors are reported early.
    bundle.validate_sample_input(&sample_input)?;

    // The cached conditioning payloads only need to exist; we do not
    // need to read them in V1 because the sampler is a placeholder.
    // We still validate they resolve through the typed accessor so
    // the backend emits a useful error for stale or wrong-backend
    // handles instead of silently succeeding.
    backend
        .store()
        .get_conditioning(positive_handle.text_embedding().payload_key())?;
    backend
        .store()
        .get_conditioning(negative_handle.text_embedding().payload_key())?;

    let input_latent = backend
        .store()
        .get_latent(latent_handle.payload().payload_key())?;

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
    );

    let latent_width = latent_handle.width();
    let latent_height = latent_handle.height();
    let latent_batch = latent_handle.batch();
    let latent_channels = latent_handle.channels();

    let latent = RuntimeLatent::new(
        BackendTensorHandle::new(
            backend.backend_kind().clone(),
            payload_key,
            TensorDType::F32,
            TensorShape::new(output_dims),
            backend.device_label(),
        ),
        latent_width,
        latent_height,
        latent_batch,
        latent_channels,
    );

    Ok(DiffusionSampleResponse::new(latent))
}

fn sampler_name_from(name: &SamplerName) -> String {
    match name {
        SamplerName::Euler => "euler".to_string(),
        SamplerName::Other(name) => name.clone(),
    }
}

fn scheduler_name_from(name: &SchedulerName) -> String {
    match name {
        SchedulerName::Normal => "normal".to_string(),
        SchedulerName::Other(name) => name.clone(),
    }
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
