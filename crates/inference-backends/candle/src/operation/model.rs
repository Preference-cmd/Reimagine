use std::sync::Arc;

use reimagine_core::model::ModelRole;
use reimagine_inference::InferenceBackend;
use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::{InferenceOutput, InferenceResponse};
use reimagine_runtime::{
    BackendKind, BackendPayloadKey, RuntimeClipHandle, RuntimeModelHandle, RuntimeVaeHandle,
    RuntimeValue,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::LoadedSdxlBundle;

pub fn execute_model_load_bundle(
    request: &InferenceRequest,
    backend: &CandleBackend,
) -> Result<InferenceResponse, CandleBackendError> {
    let model_ref = request.models().first().ok_or_else(|| {
        CandleBackendError::InvalidRequest(
            "model.load_bundle requires a resolved model".to_string(),
        )
    })?;

    let bundle = backend
        .model_cache()
        .get_bundle(model_ref.model_id())
        .unwrap_or_else(|| {
            let bundle = LoadedSdxlBundle {
                model_payload_key: format!("bundle:{}:model", model_ref.model_id().as_str()),
                clip_payload_key: format!("bundle:{}:clip", model_ref.model_id().as_str()),
                vae_payload_key: format!("bundle:{}:vae", model_ref.model_id().as_str()),
            };
            backend
                .model_cache()
                .insert_bundle(model_ref.model_id().clone(), bundle.clone());
            bundle
        });

    let backend_kind = BackendKind::from(backend.backend_kind());
    let device_label = backend.device().label();
    let model = RuntimeValue::Model(
        RuntimeModelHandle::new(
            model_ref.model_id().clone(),
            ModelRole::CheckpointBundle,
            backend_kind.clone(),
            BackendPayloadKey::new(bundle.model_payload_key),
        )
        .with_device(device_label),
    );
    let clip = RuntimeValue::Clip(
        RuntimeClipHandle::new(
            model_ref.model_id().clone(),
            backend_kind.clone(),
            BackendPayloadKey::new(bundle.clip_payload_key),
        )
        .with_device(device_label),
    );
    let vae = RuntimeValue::Vae(
        RuntimeVaeHandle::new(
            model_ref.model_id().clone(),
            backend_kind,
            BackendPayloadKey::new(bundle.vae_payload_key),
        )
        .with_device(device_label),
    );
    Ok(InferenceResponse::new(vec![
        InferenceOutput::new("model", Arc::new(model)),
        InferenceOutput::new("clip", Arc::new(clip)),
        InferenceOutput::new("vae", Arc::new(vae)),
    ]))
}
