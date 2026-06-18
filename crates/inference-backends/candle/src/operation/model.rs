//! `model.load_bundle` operation.
//!
//! Translates a [`LoadBundleRequest`] into a family-aware
//! [`LoadedModelBundle`] entry that lives in [`CandleModelCache`],
//! then emits three lightweight handles for the workflow: `model`,
//! `clip`, `vae`.
//!
//! The first call for a given resolved model dispatches on
//! `series` + `variant` to the right family loader and caches the
//! result. Subsequent calls reuse the cached entry without
//! re-reading the file.

use reimagine_core::model::ModelRole;
use reimagine_core::{RuntimeClipHandle, RuntimeModelHandle, RuntimeVaeHandle};
use reimagine_inference_core::{InferenceBackend, LoadBundleRequest, LoadBundleResponse};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;
use crate::models::LoadedSdxlBundle;

pub fn execute_model_load_bundle(
    request: LoadBundleRequest,
    backend: &CandleBackend,
) -> Result<LoadBundleResponse, CandleBackendError> {
    let resolved = request.resolved_model();

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

fn bundle_response(bundle: &LoadedModelBundle, backend: &CandleBackend) -> LoadBundleResponse {
    match bundle {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl_response(sdxl, backend),
    }
}

fn sdxl_response(bundle: &LoadedSdxlBundle, backend: &CandleBackend) -> LoadBundleResponse {
    let backend_kind = backend.backend_kind().clone();
    let device_label = backend.device_label();
    let model = RuntimeModelHandle::new(
        bundle.model_id.clone(),
        ModelRole::CheckpointBundle,
        backend_kind.clone(),
        bundle.model_payload_key.clone(),
    )
    .with_device(device_label);
    let clip = RuntimeClipHandle::new(
        bundle.model_id.clone(),
        backend_kind.clone(),
        bundle.clip_payload_key.clone(),
    )
    .with_device(device_label);
    let vae = RuntimeVaeHandle::new(
        bundle.model_id.clone(),
        backend_kind,
        bundle.vae_payload_key.clone(),
    )
    .with_device(device_label);
    LoadBundleResponse::new(model, clip, vae)
}
