use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Tensor};
use reimagine_core::model::{ModelId, RunId};
use reimagine_runtime::BackendPayloadKey;

use crate::error::CandleBackendError;
use crate::models::LoadedModelBundle;

/// Backend-owned latent tensor wrapper.
///
/// `candle_core::Tensor` is reference-counted internally, so cloning
/// a `CandleLatent` is cheap. The wrapper is the typed handle that
/// operation modules hand around; the raw `Tensor` never crosses the
/// backend boundary.
#[derive(Debug, Clone)]
pub struct CandleLatent {
    tensor: Tensor,
}

impl CandleLatent {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    pub fn dims(&self) -> Vec<usize> {
        self.tensor.shape().dims().to_vec()
    }

    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    /// Approximate byte size of the latent payload.
    pub fn byte_size(&self) -> usize {
        let dtype_size = match self.tensor.dtype() {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::I16 | DType::F8E8M0 => 2,
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::U8 => 1,
            // Future DType variants default to 1 byte; byte_size is
            // approximate and only feeds the memory snapshot.
            _ => 1,
        };
        self.tensor.elem_count() * dtype_size
    }
}

/// Backend-owned payload enum.
///
/// V1 carries a single real latent variant; later milestones add image
/// and conditioning payloads behind the same typed-accessor boundary.
#[derive(Debug, Clone)]
pub enum CandlePayload {
    Latent(CandleLatent),
}

impl CandlePayload {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Latent(_) => "latent",
        }
    }
}

/// Errors returned by the typed [`CandleStore`] accessors.
///
/// The variants are deliberately small so operation modules can map
/// them to [`CandleBackendError::InvalidRequest`] without losing the
/// useful message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    PayloadNotFound {
        key: BackendPayloadKey,
        expected: &'static str,
    },
    WrongPayloadKind {
        key: BackendPayloadKey,
        expected: &'static str,
        actual: &'static str,
    },
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PayloadNotFound { key, expected } => {
                write!(f, "no `{expected}` payload registered for key `{key}`")
            }
            Self::WrongPayloadKind {
                key,
                expected,
                actual,
            } => write!(f, "payload `{key}` is `{actual}`, expected `{expected}`"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<StoreError> for CandleBackendError {
    fn from(err: StoreError) -> Self {
        CandleBackendError::InvalidRequest(err.to_string())
    }
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

    /// Insert a real latent tensor payload and pin it to the run.
    pub fn insert_latent(&self, run_id: RunId, key: BackendPayloadKey, tensor: Tensor) {
        let mut inner = self.inner.lock().expect("store poisoned");
        inner.payloads.insert(
            key.clone(),
            CandlePayload::Latent(CandleLatent::new(tensor)),
        );
        inner.run_index.entry(run_id).or_default().push(key);
    }

    /// Borrow a latent payload behind `key` by cloning the underlying
    /// reference-counted tensor. The lock is released as soon as the
    /// cheap Arc-internal clone completes; no work is done outside the
    /// lock beyond the return.
    #[allow(unreachable_patterns)] // reserved for future image/conditioning kinds
    pub fn get_latent(&self, key: &BackendPayloadKey) -> Result<CandleLatent, StoreError> {
        let inner = self.inner.lock().expect("store poisoned");
        let cloned = inner.payloads.get(key).cloned();
        drop(inner);
        match cloned {
            Some(CandlePayload::Latent(latent)) => Ok(latent),
            // No other payload kinds in V1; this arm is reserved for
            // future image/conditioning variants so wrong-kind
            // lookups still produce a useful error.
            Some(other) => Err(StoreError::WrongPayloadKind {
                key: key.clone(),
                expected: "latent",
                actual: other.kind(),
            }),
            None => Err(StoreError::PayloadNotFound {
                key: key.clone(),
                expected: "latent",
            }),
        }
    }

    /// Take ownership of a latent payload, removing it from the store
    /// and unpinning it from any run index.
    #[allow(unreachable_patterns)] // reserved for future image/conditioning kinds
    pub fn take_latent(&self, key: &BackendPayloadKey) -> Result<Tensor, StoreError> {
        let mut inner = self.inner.lock().expect("store poisoned");
        match inner.payloads.remove(key) {
            Some(CandlePayload::Latent(latent)) => {
                for keys in inner.run_index.values_mut() {
                    keys.retain(|k| k != key);
                }
                Ok(latent.into_tensor())
            }
            Some(other) => Err(StoreError::WrongPayloadKind {
                key: key.clone(),
                expected: "latent",
                actual: other.kind(),
            }),
            None => Err(StoreError::PayloadNotFound {
                key: key.clone(),
                expected: "latent",
            }),
        }
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

    /// Release a single payload by key, if present. Used by the
    /// resource backend when a runtime value drops and the payload
    /// type is not statically known.
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

    /// Approximate total byte size of all stored latent payloads.
    pub fn latent_byte_size(&self) -> usize {
        let inner = self.inner.lock().expect("store poisoned");
        inner
            .payloads
            .values()
            .map(|payload| match payload {
                CandlePayload::Latent(latent) => latent.byte_size(),
            })
            .sum()
    }
}

/// Cross-run cache for loaded model bundles.
///
/// Stores the family-aware [`LoadedModelBundle`] wrapper so future
/// model families can be added without changing the cache shape.
#[derive(Debug, Default)]
pub struct CandleModelCache {
    bundles: Mutex<HashMap<ModelId, Arc<LoadedModelBundle>>>,
}

impl CandleModelCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a cached bundle by model id.
    pub fn get_bundle(&self, model_id: &ModelId) -> Option<Arc<LoadedModelBundle>> {
        self.bundles
            .lock()
            .expect("model cache poisoned")
            .get(model_id)
            .cloned()
    }

    /// Insert or replace a cached bundle entry.
    pub fn insert_bundle(&self, model_id: ModelId, bundle: Arc<LoadedModelBundle>) {
        self.bundles
            .lock()
            .expect("model cache poisoned")
            .insert(model_id, bundle);
    }

    /// Drop a cached bundle entry, returning the previous handle.
    pub fn remove_bundle(&self, model_id: &ModelId) -> Option<Arc<LoadedModelBundle>> {
        self.bundles
            .lock()
            .expect("model cache poisoned")
            .remove(model_id)
    }

    /// Number of cached bundles.
    pub fn bundle_count(&self) -> usize {
        self.bundles.lock().expect("model cache poisoned").len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CandleDevice;
    use reimagine_inference::ModelFormat;
    use std::fs;

    fn unique_temp_dir() -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-store-test-{nonce}"))
    }

    fn build_bundle(model_id: &str, dir: &std::path::Path) -> Arc<LoadedModelBundle> {
        fs::create_dir_all(dir).unwrap();
        let path = dir.join(format!("{model_id}.safetensors"));
        fs::write(&path, b"placeholder").unwrap();
        let device = Arc::new(CandleDevice::new("cpu").try_build_device().unwrap());
        let model_id_typed = ModelId::new(model_id);
        let sdxl = crate::models::LoadedSdxlBundle::from_resolved(
            model_id_typed.clone(),
            path,
            ModelFormat::SafeTensors,
            device,
        )
        .expect("sdxl bundle");
        Arc::new(LoadedModelBundle::StableDiffusionSdxl(sdxl))
    }

    fn build_latent_tensor(shape: &[usize]) -> Tensor {
        Tensor::zeros(
            shape,
            DType::F32,
            &CandleDevice::new("cpu").try_build_device().unwrap(),
        )
        .expect("zeros tensor")
    }

    #[test]
    fn cache_returns_none_for_unknown_model() {
        let cache = CandleModelCache::new();
        assert!(cache.get_bundle(&ModelId::new("unknown")).is_none());
    }

    #[test]
    fn cache_round_trips_bundle() {
        let cache = CandleModelCache::new();
        let dir = unique_temp_dir();
        let model_id = ModelId::new("sdxl-base-1.0");
        let bundle = build_bundle("sdxl-base-1.0", &dir);
        cache.insert_bundle(model_id.clone(), bundle.clone());
        let retrieved = cache.get_bundle(&model_id).expect("cached bundle");
        assert_eq!(retrieved.family_label(), bundle.family_label());
        match (retrieved.as_ref(), bundle.as_ref()) {
            (
                LoadedModelBundle::StableDiffusionSdxl(a),
                LoadedModelBundle::StableDiffusionSdxl(b),
            ) => {
                assert_eq!(a.model_payload_key, b.model_payload_key);
                assert_eq!(a.clip_payload_key, b.clip_payload_key);
                assert_eq!(a.vae_payload_key, b.vae_payload_key);
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_overwrites_bundle() {
        let cache = CandleModelCache::new();
        let dir = unique_temp_dir();
        let model_id = ModelId::new("sdxl-base-1.0");
        cache.insert_bundle(model_id.clone(), build_bundle("sdxl-base-1.0", &dir));
        cache.insert_bundle(model_id.clone(), build_bundle("sdxl-base-1.0", &dir));
        let retrieved = cache.get_bundle(&model_id).expect("cached bundle");
        assert_eq!(retrieved.family_label(), "stable_diffusion/sdxl");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_remove_drops_entry() {
        let cache = CandleModelCache::new();
        let dir = unique_temp_dir();
        let model_id = ModelId::new("sdxl-base-1.0");
        cache.insert_bundle(model_id.clone(), build_bundle("sdxl-base-1.0", &dir));
        assert_eq!(cache.bundle_count(), 1);
        cache.remove_bundle(&model_id);
        assert_eq!(cache.bundle_count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_insert_latent_registers_payload_and_run_pin() {
        let store = CandleStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        let tensor = build_latent_tensor(&[1, 4, 8, 8]);
        store.insert_latent(run_id.clone(), key.clone(), tensor);
        assert_eq!(store.payload_count(), 1);
        assert_eq!(store.run_payload_count(&run_id), 1);
        assert!(store.contains_payload(&key));
    }

    #[test]
    fn store_get_latent_clones_handle_with_useful_metadata() {
        let store = CandleStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        let tensor = build_latent_tensor(&[1, 4, 8, 8]);
        store.insert_latent(run_id, key.clone(), tensor);

        let latent = store.get_latent(&key).expect("latent payload");
        assert_eq!(latent.dims(), vec![1, 4, 8, 8]);
        assert_eq!(latent.dtype(), DType::F32);
        assert_eq!(latent.byte_size(), 1 * 4 * 8 * 8 * 4);
    }

    #[test]
    fn store_get_latent_reports_missing_key() {
        let store = CandleStore::new();
        let key = BackendPayloadKey::new("latent:missing");
        let err = store.get_latent(&key).unwrap_err();
        assert!(matches!(err, StoreError::PayloadNotFound { .. }));
        let msg = err.to_string();
        assert!(msg.contains("latent:missing"), "msg: {msg}");
        assert!(msg.contains("latent"), "msg: {msg}");
    }

    #[test]
    fn store_take_latent_removes_entry_and_returns_tensor() {
        let store = CandleStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        let tensor = build_latent_tensor(&[1, 4, 8, 8]);
        store.insert_latent(run_id.clone(), key.clone(), tensor);

        let taken = store.take_latent(&key).expect("latent tensor");
        assert_eq!(taken.shape().dims(), &[1, 4, 8, 8]);
        assert!(!store.contains_payload(&key));
        assert_eq!(store.run_payload_count(&run_id), 0);
    }

    #[test]
    fn store_take_latent_on_missing_key_returns_error() {
        let store = CandleStore::new();
        let key = BackendPayloadKey::new("latent:absent");
        let err = store.take_latent(&key).unwrap_err();
        assert!(matches!(err, StoreError::PayloadNotFound { .. }));
    }

    #[test]
    fn store_release_payload_removes_by_key() {
        let store = CandleStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.insert_latent(
            run_id.clone(),
            key.clone(),
            build_latent_tensor(&[1, 4, 8, 8]),
        );
        assert!(store.release_payload(&key));
        assert!(!store.contains_payload(&key));
        assert_eq!(store.run_payload_count(&run_id), 0);
    }

    #[test]
    fn store_cleanup_run_removes_latent_payloads() {
        let store = CandleStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.insert_latent(
            run_id.clone(),
            key.clone(),
            build_latent_tensor(&[1, 4, 8, 8]),
        );
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
        store.insert_latent(
            run_a.clone(),
            key_a.clone(),
            build_latent_tensor(&[1, 4, 8, 8]),
        );
        store.insert_latent(
            run_b.clone(),
            key_b.clone(),
            build_latent_tensor(&[1, 4, 8, 8]),
        );

        store.cleanup_run(&run_a);
        assert_eq!(store.payload_count(), 1);
        assert!(!store.contains_payload(&key_a));
        assert!(store.contains_payload(&key_b));
    }

    #[test]
    fn store_latent_byte_size_sums_payloads() {
        let store = CandleStore::new();
        let run_id = RunId::new("run-bytes");
        let first = BackendPayloadKey::new("latent:run-bytes:a");
        let second = BackendPayloadKey::new("latent:run-bytes:b");
        // 1 * 4 * 8 * 8 = 256 f32 elements = 1024 bytes per payload
        store.insert_latent(run_id.clone(), first, build_latent_tensor(&[1, 4, 8, 8]));
        store.insert_latent(run_id, second, build_latent_tensor(&[1, 4, 8, 8]));
        assert_eq!(store.latent_byte_size(), 2048);
    }

    #[test]
    fn store_error_display_messages_are_useful() {
        let not_found = StoreError::PayloadNotFound {
            key: BackendPayloadKey::new("latent:missing"),
            expected: "latent",
        };
        let msg = not_found.to_string();
        assert!(msg.contains("latent:missing"), "msg: {msg}");
        assert!(msg.contains("latent"), "msg: {msg}");

        let wrong_kind = StoreError::WrongPayloadKind {
            key: BackendPayloadKey::new("latent:misuse"),
            expected: "latent",
            actual: "image",
        };
        let msg = wrong_kind.to_string();
        assert!(msg.contains("latent:misuse"), "{msg}");
        assert!(msg.contains("image"), "{msg}");
        assert!(msg.contains("latent"), "{msg}");
    }
}
