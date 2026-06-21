//! `text.encode` operation.
//!
//! Translates a [`TextEncodeRequest`] into a backend-local text encoding
//! call through the loaded-bundle graph facade and returns a
//! [`TextEncodeResponse`] carrying an [`ExecutionConditioning`] handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! SDXL-specific tokenization and dual CLIP encoding live behind the
//! facade in `models/stable_diffusion/sdxl/text.rs` and
//! `models/stable_diffusion/sdxl/tokenizer.rs`.

use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_inference_core::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, ConditioningMetadata,
    ExecutionConditioning, ExecutionValue, InferenceBackend, TextEncodeRequest, TextEncodeResponse,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::graph::{TextEncodeInput, TextEncodeResult};
use crate::store::CandleConditioning;

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

    bundle.validate_clip_handle(&clip)?;

    let TextEncodeResult {
        text_embedding,
        pooled_embedding,
    } = bundle.encode_text(TextEncodeInput { prompt }, backend.device().as_ref())?;

    let text_shape = text_embedding.shape().dims().to_vec();
    let pooled_shape = pooled_embedding.shape().dims().to_vec();

    let payload_key = BackendPayloadKey::new(format!(
        "conditioning:{}:{}",
        run_id.as_str(),
        node_id.as_str()
    ));

    backend.store().insert_conditioning(
        run_id.clone(),
        payload_key.clone(),
        CandleConditioning::new(text_embedding, Some(pooled_embedding)),
    );

    let ExecutionValue::Conditioning(conditioning) = build_conditioning_runtime_value(
        payload_key,
        text_shape,
        pooled_shape,
        backend.backend_kind().as_str(),
        backend.device_label(),
    ) else {
        unreachable!("conditioning runtime builder always returns ExecutionValue::Conditioning")
    };

    Ok(TextEncodeResponse::new(conditioning))
}

/// Build a `ExecutionValue::Conditioning` from stored tensors.
///
/// This helper constructs the lightweight `BackendTensorHandle` values
/// that cross the backend boundary. The actual tensors remain in the
/// store; only the handles are returned to runtime.
fn build_conditioning_runtime_value(
    payload_key: BackendPayloadKey,
    text_embedding_shape: Vec<usize>,
    pooled_embedding_shape: Vec<usize>,
    backend_kind: &str,
    device_label: &str,
) -> ExecutionValue {
    let text_handle = BackendTensorHandle::new(
        BackendKind::from(backend_kind),
        payload_key.clone(),
        TensorDType::F32,
        TensorShape::new(text_embedding_shape),
        device_label,
    );

    let pooled_handle = BackendTensorHandle::new(
        BackendKind::from(backend_kind),
        payload_key,
        TensorDType::F32,
        TensorShape::new(pooled_embedding_shape),
        device_label,
    );

    let metadata = ConditioningMetadata::new(0, 0);

    ExecutionValue::Conditioning(
        ExecutionConditioning::new(text_handle, metadata).with_pooled_embedding(pooled_handle),
    )
}
