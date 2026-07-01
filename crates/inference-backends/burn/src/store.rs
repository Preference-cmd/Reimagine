use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use reimagine_core::model::ModelId;

use crate::models::stable_diffusion::sdxl::{BurnLoadedModelBundle, BurnSdxlSourceSignature};

#[derive(Debug, Default)]
pub struct BurnStore;

impl BurnStore {
    pub fn new() -> Self {
        Self
    }

    pub fn payload_count(&self) -> usize {
        0
    }
}

#[derive(Debug, Default)]
pub struct BurnModelCache {
    bundles: Mutex<HashMap<ModelId, Arc<BurnLoadedModelBundle>>>,
}

impl BurnModelCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_compatible_bundle(
        &self,
        model_id: &ModelId,
        signature: &BurnSdxlSourceSignature,
    ) -> Option<Arc<BurnLoadedModelBundle>> {
        let bundles = self.bundles.lock().expect("model cache poisoned");
        let bundle = bundles.get(model_id)?;
        (bundle.source_signature() == signature).then(|| bundle.clone())
    }

    pub fn insert_bundle(&self, model_id: ModelId, bundle: Arc<BurnLoadedModelBundle>) {
        self.bundles
            .lock()
            .expect("model cache poisoned")
            .insert(model_id, bundle);
    }

    pub fn bundle_count(&self) -> usize {
        self.bundles.lock().expect("model cache poisoned").len()
    }
}
