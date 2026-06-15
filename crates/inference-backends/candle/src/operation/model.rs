//! `model.load_bundle` operation.
//!
//! Translates a resolved [`ResolvedInferenceModel`] into a
//! family-aware [`LoadedModelBundle`] entry that lives in
//! [`CandleModelCache`], then emits three lightweight
//! `RuntimeValue` handles for the workflow: `model`, `clip`, `vae`.
//!
//! The first call for a given resolved model dispatches on
//! `series` + `variant` to the right family loader and caches the
//! result. Subsequent calls reuse the cached entry without
//! re-reading the file.

use std::sync::Arc;

use reimagine_core::model::ModelRole;
use reimagine_inference::InferenceBackend;
use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::{InferenceOutput, InferenceResponse};
use reimagine_runtime::{
    BackendKind, RuntimeClipHandle, RuntimeModelHandle, RuntimeVaeHandle, RuntimeValue,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;
use crate::models::LoadedSdxlBundle;

pub fn execute_model_load_bundle(
    request: &InferenceRequest,
    backend: &CandleBackend,
) -> Result<InferenceResponse, CandleBackendError> {
    let resolved = request.models().first().ok_or_else(|| {
        CandleBackendError::InvalidRequest(
            "model.load_bundle requires a resolved model".to_string(),
        )
    })?;

    // Concurrent first-time loads for the same model id may both miss
    // the cache and validate the file. The race is benign: the second
    // entry overwrites the first and every caller still receives a
    // valid `Arc<LoadedModelBundle>`. Contention-bounded fixes live
    // behind the cache lock.
    let bundle = match backend.model_cache().get_bundle(resolved.model_id()) {
        Some(bundle) => bundle,
        None => LoadedModelBundle::load(
            resolved.model_id().clone(),
            resolved.series(),
            resolved.variant(),
            resolved.source_path(),
            resolved.format(),
            backend.device().clone(),
        )
        .map(|bundle| {
            backend
                .model_cache()
                .insert_bundle(resolved.model_id().clone(), bundle.clone());
            bundle
        })?,
    };

    Ok(bundle_response(&bundle, backend))
}

fn bundle_response(bundle: &LoadedModelBundle, backend: &CandleBackend) -> InferenceResponse {
    match bundle {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl_response(sdxl, backend),
    }
}

fn sdxl_response(bundle: &LoadedSdxlBundle, backend: &CandleBackend) -> InferenceResponse {
    let backend_kind = BackendKind::from(backend.backend_kind());
    let device_label = backend.device_label();
    let model = RuntimeValue::Model(
        RuntimeModelHandle::new(
            bundle.model_id.clone(),
            ModelRole::CheckpointBundle,
            backend_kind.clone(),
            bundle.model_payload_key.clone(),
        )
        .with_device(device_label),
    );
    let clip = RuntimeValue::Clip(
        RuntimeClipHandle::new(
            bundle.model_id.clone(),
            backend_kind.clone(),
            bundle.clip_payload_key.clone(),
        )
        .with_device(device_label),
    );
    let vae = RuntimeValue::Vae(
        RuntimeVaeHandle::new(
            bundle.model_id.clone(),
            backend_kind,
            bundle.vae_payload_key.clone(),
        )
        .with_device(device_label),
    );
    InferenceResponse::new(vec![
        InferenceOutput::new("model", Arc::new(model)),
        InferenceOutput::new("clip", Arc::new(clip)),
        InferenceOutput::new("vae", Arc::new(vae)),
    ])
}
