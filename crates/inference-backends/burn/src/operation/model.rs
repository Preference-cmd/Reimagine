use std::sync::Arc;

use reimagine_inference::{InferenceBackend, LoadBundleRequest, LoadBundleResponse};

use crate::backend::BurnBackend;
use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::{BurnLoadedModelBundle, BurnLoadedSdxlBundle};

pub fn execute_model_load_bundle(
    request: LoadBundleRequest,
    backend: &BurnBackend,
) -> Result<LoadBundleResponse, BurnBackendError> {
    let resolved = request.resolved_model();
    let Some(source_set) = resolved.source_set() else {
        return Err(BurnBackendError::UnsupportedSourceLayout(format!(
            "Burn model `{}` requires burn/04 converted SplitComponent source set",
            resolved.model_id()
        )));
    };

    let candidate = BurnLoadedSdxlBundle::from_resolved(resolved, source_set)?;
    let signature = candidate.source_signature().clone();
    let bundle = match backend
        .model_cache()
        .get_compatible_bundle(resolved.model_id(), &signature)
    {
        Some(bundle) => bundle,
        None => {
            let bundle = Arc::new(BurnLoadedModelBundle::StableDiffusionSdxl(Arc::new(
                candidate,
            )));
            backend
                .model_cache()
                .insert_bundle(resolved.model_id().clone(), bundle.clone());
            bundle
        }
    };

    Ok(bundle.load_bundle_response(
        backend.backend_kind().clone(),
        backend.backend_instance(),
        backend.device_label(),
    ))
}
