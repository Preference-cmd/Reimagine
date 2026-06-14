use std::collections::HashMap;
use std::sync::Mutex;

use reimagine_core::model::ModelId;
use reimagine_core::model::RunId;
use reimagine_runtime::BackendPayloadKey;

use crate::models::LoadedSdxlBundle;

/// Placeholder enum for backend-owned payloads.
///
/// V1 holds only lightweight descriptors; real Candle tensors will land
/// in follow-up issues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandlePayload {
    LatentPlaceholder { dims: Vec<usize>, dtype: String },
}

/// Per-backend store that owns run-scoped payloads.
///
/// Cross-run model cache lives in [`CandleModelCache`].
#[derive(Debug, Default)]
pub struct CandleStore {
    inner: Mutex<CandleStoreInner>,
}

#[derive(Debug, Default)]
struct CandleStoreInner {
    payloads: HashMap<BackendPayloadKey, CandlePayload>,
    run_index: HashMap<RunId, Vec<BackendPayloadKey>>,
}

impl CandleStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a run-scoped payload key.
    pub fn register_run_payload(
        &self,
        run_id: RunId,
        key: BackendPayloadKey,
        payload: CandlePayload,
    ) {
        let mut inner = self.inner.lock().expect("store poisoned");
        inner.payloads.insert(key.clone(), payload);
        inner.run_index.entry(run_id).or_default().push(key);
    }

    /// Remove all payloads and run pins for the given run id.
    pub fn cleanup_run(&self, run_id: &RunId) {
        let mut inner = self.inner.lock().expect("store poisoned");
        if let Some(keys) = inner.run_index.remove(run_id) {
            for key in keys {
                inner.payloads.remove(&key);
            }
        }
    }

    /// Release a single payload by key, if present.
    pub fn release_payload(&self, key: &BackendPayloadKey) -> bool {
        let mut inner = self.inner.lock().expect("store poisoned");
        let removed = inner.payloads.remove(key).is_some();
        for keys in inner.run_index.values_mut() {
            keys.retain(|k| k != key);
        }
        removed
    }

    /// Total number of payloads currently stored.
    pub fn payload_count(&self) -> usize {
        self.inner.lock().expect("store poisoned").payloads.len()
    }

    /// Number of payloads registered for a specific run.
    pub fn run_payload_count(&self, run_id: &RunId) -> usize {
        self.inner
            .lock()
            .expect("store poisoned")
            .run_index
            .get(run_id)
            .map(|keys| keys.len())
            .unwrap_or(0)
    }

    /// Check if a payload key exists in the store.
    pub fn contains_payload(&self, key: &BackendPayloadKey) -> bool {
        self.inner
            .lock()
            .expect("store poisoned")
            .payloads
            .contains_key(key)
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

    pub fn bundle_count(&self) -> usize {
        self.bundles.lock().expect("model cache poisoned").len()
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

    #[test]
    fn store_registers_and_cleans_run_payloads() {
        let store = CandleStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.register_run_payload(
            run_id.clone(),
            key.clone(),
            CandlePayload::LatentPlaceholder {
                dims: vec![1, 4, 64, 64],
                dtype: "f32".to_string(),
            },
        );
        assert_eq!(store.payload_count(), 1);
        assert_eq!(store.run_payload_count(&run_id), 1);
        assert!(store.contains_payload(&key));

        store.cleanup_run(&run_id);
        assert_eq!(store.payload_count(), 0);
        assert_eq!(store.run_payload_count(&run_id), 0);
        assert!(!store.contains_payload(&key));
    }

    #[test]
    fn store_cleanup_run_does_not_affect_other_runs() {
        let store = CandleStore::new();
        let run_a = RunId::new("run-a");
        let run_b = RunId::new("run-b");
        let key_a = BackendPayloadKey::new("latent:run-a:node-a");
        let key_b = BackendPayloadKey::new("latent:run-b:node-a");

        store.register_run_payload(
            run_a.clone(),
            key_a.clone(),
            CandlePayload::LatentPlaceholder {
                dims: vec![1, 4, 64, 64],
                dtype: "f32".to_string(),
            },
        );
        store.register_run_payload(
            run_b.clone(),
            key_b.clone(),
            CandlePayload::LatentPlaceholder {
                dims: vec![1, 4, 64, 64],
                dtype: "f32".to_string(),
            },
        );

        store.cleanup_run(&run_a);
        assert_eq!(store.payload_count(), 1);
        assert!(!store.contains_payload(&key_a));
        assert!(store.contains_payload(&key_b));
    }

    #[test]
    fn store_release_payload_removes_single_entry() {
        let store = CandleStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.register_run_payload(
            run_id.clone(),
            key.clone(),
            CandlePayload::LatentPlaceholder {
                dims: vec![1, 4, 64, 64],
                dtype: "f32".to_string(),
            },
        );
        store.release_payload(&key);
        assert_eq!(store.payload_count(), 0);
        assert_eq!(store.run_payload_count(&run_id), 0);
    }
}
