//! `diffusion.sample` operation.
//!
//! Translates a backend-neutral [`InferenceRequest`] for SDXL sampling
//! into a backend-local sampler pass and returns a
//! `RuntimeValue::Latent` handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! SDXL-specific tokenization, sampling, and UNet work live in
//! `models/stable_diffusion/sdxl/diffusion.rs`.
//!
//! V1 uses a deterministic placeholder sampler so the backend
//! contract is testable without real UNet weights. The
//! `diffusion.sample` call still produces a real latent tensor that
//! flows through the existing typed `CandleStore` accessors.

use std::sync::Arc;

use reimagine_core::model::SlotId;
use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_inference::InferenceBackend;
use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::{InferenceOutput, InferenceResponse};
use reimagine_runtime::{BackendKind, BackendPayloadKey, RuntimeLatent, RuntimeValue};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;
use crate::models::stable_diffusion::sdxl::diffusion::{SdxlSampleRequest, SdxlSampler};

pub fn execute_diffusion_sample(
    request: &InferenceRequest,
    backend: &CandleBackend,
) -> Result<InferenceResponse, CandleBackendError> {
    let model = require_input(request, "model")?;
    let positive = require_input(request, "positive")?;
    let negative = require_input(request, "negative")?;
    let latent = require_input(request, "latent")?;

    let model_handle = require_model_handle(model, backend)?;
    let positive_handle = require_conditioning_handle(positive, backend)?;
    let negative_handle = require_conditioning_handle(negative, backend)?;
    let latent_handle = require_latent_handle(latent, backend)?;

    let sample_request = SdxlSampleRequest::from_params(request.params())?;

    let bundle = backend
        .model_cache()
        .get_bundle(model_handle.model_id())
        .ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "diffusion.sample `model` handle points to model `{}` with no loaded bundle; load the model first via model.load_bundle",
                model_handle.model_id().as_str()
            ))
        })?;

    let bundle = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.clone(),
    };

    if model_handle.payload_key() != &bundle.model_payload_key {
        return Err(CandleBackendError::InvalidRequest(format!(
            "diffusion.sample `model` payload `{}` does not match loaded SDXL model payload `{}` for model `{}`",
            model_handle.payload_key().as_str(),
            bundle.model_payload_key.as_str(),
            bundle.model_id.as_str()
        )));
    }

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

    let sampler = SdxlSampler::new();
    let result = sampler.sample(input_latent, &sample_request, backend.device().as_ref())?;

    let payload_key = BackendPayloadKey::new(format!(
        "latent:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    let output_dims = result.latent.dims();

    backend.store().insert_latent(
        request.run_id().clone(),
        payload_key.clone(),
        result.latent.into_tensor(),
    );

    let latent_width = latent_handle.width();
    let latent_height = latent_handle.height();
    let latent_batch = latent_handle.batch();
    let latent_channels = latent_handle.channels();

    let latent = RuntimeValue::Latent(RuntimeLatent::new(
        make_tensor_handle(
            backend.backend_kind(),
            payload_key,
            output_dims,
            backend.device_label(),
        ),
        latent_width,
        latent_height,
        latent_batch,
        latent_channels,
    ));

    Ok(InferenceResponse::new(vec![InferenceOutput::new(
        "latent",
        Arc::new(latent),
    )]))
}

fn require_input<'a>(
    request: &'a InferenceRequest,
    slot: &str,
) -> Result<&'a Arc<RuntimeValue>, CandleBackendError> {
    request.inputs().get(&SlotId::new(slot)).ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!("diffusion.sample requires a `{slot}` input"))
    })
}

fn require_model_handle(
    value: &RuntimeValue,
    backend: &CandleBackend,
) -> Result<reimagine_runtime::RuntimeModelHandle, CandleBackendError> {
    match value {
        RuntimeValue::Model(handle) => {
            if handle.backend().as_str() != backend.backend_kind() {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample `model` handle belongs to backend `{}`, expected `{}`",
                    handle.backend().as_str(),
                    backend.backend_kind()
                )));
            }
            Ok(handle.clone())
        }
        _ => Err(CandleBackendError::InvalidRequest(
            "diffusion.sample `model` input must be a Model handle".to_string(),
        )),
    }
}

fn require_conditioning_handle(
    value: &RuntimeValue,
    backend: &CandleBackend,
) -> Result<reimagine_runtime::RuntimeConditioning, CandleBackendError> {
    match value {
        RuntimeValue::Conditioning(handle) => {
            if handle.text_embedding().backend().as_str() != backend.backend_kind() {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample conditioning handle belongs to backend `{}`, expected `{}`",
                    handle.text_embedding().backend().as_str(),
                    backend.backend_kind()
                )));
            }
            if let Some(pooled) = handle.pooled_embedding() {
                if pooled.backend().as_str() != backend.backend_kind() {
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
        _ => Err(CandleBackendError::InvalidRequest(
            "diffusion.sample conditioning input must be a Conditioning handle".to_string(),
        )),
    }
}

fn require_latent_handle(
    value: &RuntimeValue,
    backend: &CandleBackend,
) -> Result<reimagine_runtime::RuntimeLatent, CandleBackendError> {
    match value {
        RuntimeValue::Latent(handle) => {
            if handle.payload().backend().as_str() != backend.backend_kind() {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "diffusion.sample `latent` handle belongs to backend `{}`, expected `{}`",
                    handle.payload().backend().as_str(),
                    backend.backend_kind()
                )));
            }
            Ok(handle.clone())
        }
        _ => Err(CandleBackendError::InvalidRequest(
            "diffusion.sample `latent` input must be a Latent handle".to_string(),
        )),
    }
}

fn make_tensor_handle(
    backend_kind: &str,
    payload_key: BackendPayloadKey,
    shape: Vec<usize>,
    device_label: &str,
) -> reimagine_runtime::BackendTensorHandle {
    reimagine_runtime::BackendTensorHandle::new(
        BackendKind::from(backend_kind),
        payload_key,
        TensorDType::F32,
        TensorShape::new(shape),
        device_label,
    )
}
