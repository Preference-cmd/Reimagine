//! `text.encode` operation.
//!
//! Translates a [`TextEncodeRequest`] into backend-local SDXL text
//! encoding calls and returns a [`TextEncodeResponse`] carrying an
//! [`ExecutionConditioning`] handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! SDXL-specific tokenization and dual CLIP encoding live in
//! `models/stable_diffusion/sdxl/text.rs` and
//! `models/stable_diffusion/sdxl/tokenizer.rs`.

use reimagine_inference_core::ExecutionValue;
use reimagine_inference_core::{InferenceBackend, TextEncodeRequest, TextEncodeResponse};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;
use crate::models::stable_diffusion::sdxl::text::{
    SdxlTextEncoder, build_conditioning_runtime_value,
};

pub fn execute_text_encode(
    request: TextEncodeRequest,
    backend: &CandleBackend,
) -> Result<TextEncodeResponse, CandleBackendError> {
    let run_id = request.run_id().clone();
    let node_id = request.node_id().clone();
    let (clip, text) = request.into_parts();
    let prompt = match text.as_ref() {
        ExecutionValue::Param(reimagine_core::model::ParamValue::String(s)) => s.clone(),
        _ => {
            return Err(CandleBackendError::InvalidRequest(
                "text.encode `text` input must be a string parameter".to_string(),
            ));
        }
    };

    if clip.backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "text.encode `clip` handle belongs to backend `{}`, expected `{}`",
            clip.backend().as_str(),
            backend.backend_kind()
        )));
    }

    let bundle = backend
        .model_cache()
        .get_bundle(clip.model_id())
        .ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "no loaded model bundle found for model `{}`; load the model first via model.load_bundle",
                clip.model_id().as_str()
            ))
        })?;

    match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => {
            if clip.payload_key() != &sdxl.clip_payload_key {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "text.encode `clip` payload `{}` does not match loaded SDXL CLIP payload `{}` for model `{}`",
                    clip.payload_key().as_str(),
                    sdxl.clip_payload_key.as_str(),
                    sdxl.model_id.as_str()
                )));
            }
            encode_sdxl(&prompt, backend, &run_id, &node_id)
        }
    }
}

fn encode_sdxl(
    prompt: &str,
    backend: &CandleBackend,
    run_id: &reimagine_core::model::RunId,
    node_id: &reimagine_core::model::NodeId,
) -> Result<TextEncodeResponse, CandleBackendError> {
    let encoder = SdxlTextEncoder::new();
    let device = backend.device();
    let backend_kind = backend.backend_kind().as_str();
    let device_label = backend.device_label();

    let (payload_key, text_emb, pooled_emb) = encoder.encode_and_store(
        prompt,
        device,
        backend.store(),
        run_id,
        node_id,
        backend_kind,
        device_label,
    )?;

    let text_shape = text_emb.shape().dims().to_vec();
    let pooled_shape = pooled_emb.shape().dims().to_vec();

    let conditioning_value = build_conditioning_runtime_value(
        payload_key,
        text_shape,
        pooled_shape,
        backend_kind,
        device_label,
    );

    let ExecutionValue::Conditioning(conditioning) = conditioning_value else {
        return Err(CandleBackendError::InvalidRequest(
            "text.encode conditioning builder returned a non-conditioning value".to_string(),
        ));
    };

    Ok(TextEncodeResponse::new(conditioning))
}
