//! `model.load_bundle` operation.
//!
//! Translates a [`LoadBundleRequest`] into a family-aware
//! [`LoadedModelBundle`] entry that lives in [`CandleModelCache`],
//! then emits three lightweight handles for the workflow: `model`,
//! `clip`, `vae`.
//!
//! The first call for a given resolved model dispatches on
//! `series` + `variant` to the right family loader and caches the
//! result. Subsequent calls reuse the cached entry if the source
//! set is compatible; incompatible entries are evicted.

use reimagine_inference::{InferenceBackend, LoadBundleRequest, LoadBundleResponse};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;

pub fn execute_model_load_bundle(
    request: LoadBundleRequest,
    backend: &CandleBackend,
) -> Result<LoadBundleResponse, CandleBackendError> {
    let resolved = request.resolved_model();

    let bundle = {
        let source_set = resolved
            .source_set()
            .cloned()
            .unwrap_or_else(|| resolved.to_checkpoint_bundle_source_set());

        match backend
            .model_cache()
            .get_compatible_bundle(resolved.model_id(), &source_set)
        {
            Some(bundle) => bundle,
            None => LoadedModelBundle::load_from_source_set(
                resolved.model_id().clone(),
                resolved.series(),
                resolved.variant(),
                &source_set,
                resolved.format(),
                backend.device().clone(),
            )
            .map(|bundle| {
                backend
                    .model_cache()
                    .insert_bundle(resolved.model_id().clone(), bundle.clone());
                bundle
            })?,
        }
    };

    bundle_response(&bundle, backend)
}

fn bundle_response(
    bundle: &LoadedModelBundle,
    backend: &CandleBackend,
) -> Result<LoadBundleResponse, CandleBackendError> {
    bundle.load_bundle_response(
        backend.backend_kind().clone(),
        backend.backend_instance(),
        backend.device_label(),
    )
}
