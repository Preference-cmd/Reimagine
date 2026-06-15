//! `text.encode` operation.
//!
//! Translates a backend-neutral [`InferenceRequest`] into
//! backend-local SDXL text encoding calls and returns a
//! `RuntimeValue::Conditioning` handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! SDXL-specific tokenization and dual CLIP encoding live in
//! `models/stable_diffusion/sdxl/text.rs` and
//! `models/stable_diffusion/sdxl/tokenizer.rs`.

use std::sync::Arc;

use reimagine_inference::InferenceBackend;
use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::{InferenceOutput, InferenceResponse};
use reimagine_runtime::RuntimeValue;

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;
use crate::models::stable_diffusion::sdxl::LoadedSdxlBundle;
use crate::models::stable_diffusion::sdxl::text::{
    SdxlTextEncoder, build_conditioning_runtime_value,
};

pub fn execute_text_encode(
    request: &InferenceRequest,
    backend: &CandleBackend,
) -> Result<InferenceResponse, CandleBackendError> {
    // Extract the clip input — this is the loaded model bundle's clip handle
    let clip = request
        .inputs()
        .get(&reimagine_core::model::SlotId::new("clip"))
        .ok_or_else(|| {
            CandleBackendError::InvalidRequest("text.encode requires a `clip` input".to_string())
        })?;

    // Extract the text input — this is the prompt string
    let text = request
        .inputs()
        .get(&reimagine_core::model::SlotId::new("text"))
        .ok_or_else(|| {
            CandleBackendError::InvalidRequest("text.encode requires a `text` input".to_string())
        })?;

    // Get the prompt text from the RuntimeValue
    let prompt = match text.as_ref() {
        RuntimeValue::Param(reimagine_core::model::ParamValue::String(s)) => s.clone(),
        _ => {
            return Err(CandleBackendError::InvalidRequest(
                "text.encode `text` input must be a string parameter".to_string(),
            ));
        }
    };

    let clip_handle = match clip.as_ref() {
        RuntimeValue::Clip(handle) => handle,
        _ => {
            return Err(CandleBackendError::InvalidRequest(
                "text.encode `clip` input must be a Clip handle".to_string(),
            ));
        }
    };

    if clip_handle.backend().as_str() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "text.encode `clip` handle belongs to backend `{}`, expected `{}`",
            clip_handle.backend().as_str(),
            backend.backend_kind()
        )));
    }

    let bundle = backend
        .model_cache()
        .get_bundle(clip_handle.model_id())
        .ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!(
            "no loaded model bundle found for model `{}`; load the model first via model.load_bundle",
            clip_handle.model_id().as_str()
        ))
    })?;

    match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
            if clip_handle.payload_key() != &sdxl.clip_payload_key {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "text.encode `clip` payload `{}` does not match loaded SDXL CLIP payload `{}` for model `{}`",
                    clip_handle.payload_key().as_str(),
                    sdxl.clip_payload_key.as_str(),
                    sdxl.model_id.as_str()
                )));
            }
            encode_sdxl(request, backend, &prompt, sdxl)
        }
    }
}

fn encode_sdxl(
    request: &InferenceRequest,
    backend: &CandleBackend,
    prompt: &str,
    _bundle: &LoadedSdxlBundle,
) -> Result<InferenceResponse, CandleBackendError> {
    let encoder = SdxlTextEncoder::new();
    let device = backend.device();
    let backend_kind = backend.backend_kind();
    let device_label = backend.device_label();

    let (payload_key, text_emb, pooled_emb) = encoder.encode_and_store(
        prompt,
        device,
        backend.store(),
        request.run_id(),
        request.node_id(),
        backend_kind,
        device_label,
    )?;

    let text_shape = text_emb.shape().dims().to_vec();
    let pooled_shape = pooled_emb.shape().dims().to_vec();

    let conditioning = build_conditioning_runtime_value(
        payload_key,
        text_shape,
        pooled_shape,
        backend_kind,
        device_label,
    );

    Ok(InferenceResponse::new(vec![InferenceOutput::new(
        "conditioning",
        Arc::new(conditioning),
    )]))
}
