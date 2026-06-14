use std::collections::HashMap;
use std::sync::Mutex;

use reimagine_core::model::ModelId;

use crate::models::LoadedSdxlBundle;

#[derive(Debug, Default, Clone)]
pub struct CandleStore;

impl CandleStore {
    pub fn new() -> Self {
        Self
    }
}

/// Cross-run cache for loaded model bundles.
///
/// V1 stores placeholder bundle descriptors keyed by model id. Real
/// loaded checkpoint / UNet / CLIP / VAE objects will live here once
/// the Candle kernels land in follow-up issues.
#[derive(Debug, Default)]
pub struct CandleModelCache {
    bundles: Mutex<HashMap<ModelId, LoadedSdxlBundle>>,
}

impl CandleModelCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_bundle(&self, model_id: &ModelId) -> Option<LoadedSdxlBundle> {
        self.bundles
            .lock()
            .expect("model cache poisoned")
            .get(model_id)
            .cloned()
    }

    pub fn insert_bundle(&self, model_id: ModelId, bundle: LoadedSdxlBundle) {
        self.bundles
            .lock()
            .expect("model cache poisoned")
            .insert(model_id, bundle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bundle(model_id: &str) -> LoadedSdxlBundle {
        LoadedSdxlBundle {
            model_payload_key: format!("bundle:{model_id}:model"),
            clip_payload_key: format!("bundle:{model_id}:clip"),
            vae_payload_key: format!("bundle:{model_id}:vae"),
        }
    }

    #[test]
    fn cache_returns_none_for_unknown_model() {
        let cache = CandleModelCache::new();
        assert!(cache.get_bundle(&ModelId::new("unknown")).is_none());
    }

    #[test]
    fn cache_round_trips_bundle() {
        let cache = CandleModelCache::new();
        let model_id = ModelId::new("sdxl-base-1.0");
        let bundle = sample_bundle("sdxl-base-1.0");
        cache.insert_bundle(model_id.clone(), bundle.clone());
        let retrieved = cache.get_bundle(&model_id).expect("cached bundle");
        assert_eq!(retrieved.model_payload_key, bundle.model_payload_key);
        assert_eq!(retrieved.clip_payload_key, bundle.clip_payload_key);
        assert_eq!(retrieved.vae_payload_key, bundle.vae_payload_key);
    }

    #[test]
    fn cache_overwrites_bundle() {
        let cache = CandleModelCache::new();
        let model_id = ModelId::new("sdxl-base-1.0");
        cache.insert_bundle(model_id.clone(), sample_bundle("first"));
        cache.insert_bundle(model_id.clone(), sample_bundle("second"));
        let retrieved = cache.get_bundle(&model_id).expect("cached bundle");
        assert!(retrieved.model_payload_key.contains("second"));
    }
}
