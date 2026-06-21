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

use reimagine_inference::{InferenceBackend, LoadBundleRequest, LoadBundleResponse};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;

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

    bundle_response(&bundle, backend)
}

fn bundle_response(
    bundle: &LoadedModelBundle,
    backend: &CandleBackend,
) -> Result<LoadBundleResponse, CandleBackendError> {
    bundle.load_bundle_response(backend.backend_kind().clone(), backend.device_label())
}
