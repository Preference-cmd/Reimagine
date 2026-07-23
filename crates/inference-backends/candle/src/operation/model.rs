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
            None => load_model_bundle(resolved, &source_set, backend).inspect(|bundle| {
                backend
                    .model_cache()
                    .insert_bundle(resolved.model_id().clone(), bundle.clone());
            })?,
        }
    };

    bundle_response(&bundle, backend)
}

fn load_model_bundle(
    resolved: &reimagine_inference::ResolvedInferenceModel,
    source_set: &reimagine_inference::ResolvedInferenceModelSourceSet,
    backend: &CandleBackend,
) -> Result<std::sync::Arc<LoadedModelBundle>, CandleBackendError> {
    let bundle = if backend.allow_test_text_projection() {
        LoadedModelBundle::load_from_source_set_with_test_text_projection(
            resolved.model_id().clone(),
            resolved.series(),
            resolved.variant(),
            source_set,
            resolved.format(),
            backend.device().clone(),
        )?
    } else {
        LoadedModelBundle::load_from_source_set(
            resolved.model_id().clone(),
            resolved.series(),
            resolved.variant(),
            source_set,
            resolved.format(),
            backend.device().clone(),
        )?
    };

    if backend.allow_test_vae_projection() {
        install_test_vae_graph(&bundle);
    }

    Ok(bundle)
}

/// Install a test-only VAE graph on a freshly loaded bundle.
/// Production code never reaches this path; only the
/// `with_test_vae_projection()` test helper enables it.
#[doc(hidden)]
#[allow(irrefutable_let_patterns)]
fn install_test_vae_graph(bundle: &std::sync::Arc<LoadedModelBundle>) {
    // `LoadedModelBundle::TestPlaceholder` is gated to unit tests in
    // the candle crate and is not reachable through the production
    // bundle-construction flow used here. Use an `if let` so the
    // compiler can warn if a future non-SDXL variant lands in
    // production builds.
    if let LoadedModelBundle::StableDiffusionSdxl(sdxl) = bundle.as_ref() {
        use crate::models::stable_diffusion::sdxl::vae::SdxlVaeGraph;
        sdxl.install_test_vae_graph_for_tests(
            std::sync::Arc::new(SdxlVaeGraph::test_placeholder()),
        );
    }
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
