use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData};
use reimagine_core::model::{ModelId, RunId};
use reimagine_inference::{BackendPayloadKey, LatentSpaceMetadata};

use crate::models::stable_diffusion::sdxl::{BurnLoadedModelBundle, BurnSdxlSourceSignature};

/// Backend-owned latent payload wrapper.
///
/// V1 only carries the burn-ndarray CPU tensor variant. The payload
/// holds the latent-space metadata the tensor was allocated for so
/// the store can validate that a stored latent matches the metadata
/// the caller requested. Later issues (burn/13 wgpu, etc.) can
/// extend this enum without changing the operation/store seams.
#[derive(Debug, Clone)]
pub struct BurnLatentPayload {
    tensor: Tensor<NdArray, 4>,
    latent_space: LatentSpaceMetadata,
    width: u32,
    height: u32,
    batch: u32,
}

impl BurnLatentPayload {
    /// Build a backend-owned latent payload from a concrete
    /// burn-ndarray 4D tensor.
    pub fn new_ndarray(
        tensor: Tensor<NdArray, 4>,
        latent_space: LatentSpaceMetadata,
        width: u32,
        height: u32,
        batch: u32,
    ) -> Self {
        Self {
            tensor,
            latent_space,
            width,
            height,
            batch,
        }
    }

    /// Borrow the underlying Burn tensor. The tensor never crosses
    /// the backend boundary; only Burn-private callers can use this.
    pub fn tensor(&self) -> &Tensor<NdArray, 4> {
        &self.tensor
    }

    /// Consume the payload and return the underlying tensor.
    pub fn into_tensor(self) -> Tensor<NdArray, 4> {
        self.tensor
    }

    /// Shape of the stored tensor as a `[batch, channels, h, w]` slice.
    pub fn dims(&self) -> [usize; 4] {
        self.tensor.shape().dims()
    }

    /// Latent-space metadata this payload belongs to.
    pub fn latent_space(&self) -> &LatentSpaceMetadata {
        &self.latent_space
    }

    /// Pixel width of the original image the latent was sized for.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Pixel height of the original image the latent was sized for.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Batch size recorded on the payload.
    pub fn batch(&self) -> u32 {
        self.batch
    }

    /// Approximate byte size of the payload. Burn tensors are
    /// float32 by default in V1.
    pub fn byte_size(&self) -> usize {
        self.tensor.shape().num_elements() * std::mem::size_of::<f32>()
    }

    /// Pull the data buffer out of the tensor. Used by tests and
    /// future sampling/decode paths that need a contiguous view.
    pub fn to_data(&self) -> TensorData {
        self.tensor.to_data()
    }
}

/// Errors returned by typed [`BurnStore`] accessors.
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

/// Per-backend store that owns run-scoped payloads.
///
/// V1 carries latent payloads only. Conditioning (burn/08) and
/// image (burn/11) payloads will extend [`BurnStorePayload`]
/// without changing the operation/hook seams. The store is shared
/// between [`BurnBackend`](crate::backend::BurnBackend) and the
/// [`BurnBackendInstanceRuntimeHooks`](crate::resource::BurnBackendInstanceRuntimeHooks)
/// so runtime snapshots observe the same payload state as the
/// backend instance.
///
/// Cross-run model cache lives in [`BurnModelCache`].
#[derive(Debug, Default)]
pub struct BurnStore {
    inner: Mutex<BurnStoreInner>,
}

#[derive(Debug, Default)]
struct BurnStoreInner {
    payloads: HashMap<BackendPayloadKey, BurnStorePayload>,
    run_index: HashMap<RunId, Vec<BackendPayloadKey>>,
}

#[derive(Debug, Clone)]
enum BurnStorePayload {
    Latent(BurnLatentPayload),
}

impl BurnStorePayload {
    fn kind(&self) -> &'static str {
        match self {
            Self::Latent(_) => "latent",
        }
    }
}

impl BurnStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a real Burn-private latent payload and pin it to the
    /// run. The caller must supply the [`LatentSpaceMetadata`] this
    /// latent belongs to so the store and the operation layer agree
    /// on metadata.
    pub fn insert_latent(&self, run_id: RunId, key: BackendPayloadKey, payload: BurnLatentPayload) {
        let mut inner = self.inner.lock().expect("store poisoned");
        inner
            .payloads
            .insert(key.clone(), BurnStorePayload::Latent(payload));
        inner.run_index.entry(run_id).or_default().push(key);
    }

    /// Borrow a latent payload by key. The lock is released as soon
    /// as the cheap Burn tensor clone completes; no work happens
    /// outside the lock beyond the return.
    #[allow(unreachable_patterns)] // Future wgpu/flex variants will reuse this arm.
    pub fn get_latent(&self, key: &BackendPayloadKey) -> Result<BurnLatentPayload, StoreError> {
        let inner = self.inner.lock().expect("store poisoned");
        let cloned = inner.payloads.get(key).cloned();
        drop(inner);
        match cloned {
            Some(BurnStorePayload::Latent(latent)) => Ok(latent),
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

    /// Take ownership of a latent payload, removing it from the
    /// store and unpinning it from any run index.
    #[allow(unreachable_patterns)] // Future wgpu/flex variants will reuse this arm.
    pub fn take_latent(&self, key: &BackendPayloadKey) -> Result<BurnLatentPayload, StoreError> {
        let mut inner = self.inner.lock().expect("store poisoned");
        match inner.payloads.remove(key) {
            Some(BurnStorePayload::Latent(latent)) => {
                for keys in inner.run_index.values_mut() {
                    keys.retain(|k| k != key);
                }
                Ok(latent)
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
    ///
    /// Returns the number of payloads evicted so the runtime hooks
    /// can include the count in their lifecycle report.
    pub fn cleanup_run(&self, run_id: &RunId) -> usize {
        let mut inner = self.inner.lock().expect("store poisoned");
        let Some(keys) = inner.run_index.remove(run_id) else {
            return 0;
        };
        let mut removed = 0usize;
        for key in keys {
            if inner.payloads.remove(&key).is_some() {
                removed += 1;
            }
        }
        removed
    }

    /// Release a single payload by key, if present. Used by the
    /// backend-instance runtime hooks when a runtime value drops
    /// and the payload type is not statically known.
    pub fn release_payload(&self, key: &BackendPayloadKey) -> bool {
        let mut inner = self.inner.lock().expect("store poisoned");
        let removed = inner.payloads.remove(key).is_some();
        for keys in inner.run_index.values_mut() {
            keys.retain(|k| k != key);
        }
        removed
    }

    /// Check if a payload key exists in the store.
    pub fn contains_payload(&self, key: &BackendPayloadKey) -> bool {
        self.inner
            .lock()
            .expect("store poisoned")
            .payloads
            .contains_key(key)
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

    /// Approximate total byte size of all stored payloads.
    pub fn payload_byte_size(&self) -> usize {
        let inner = self.inner.lock().expect("store poisoned");
        inner
            .payloads
            .values()
            .map(|payload| match payload {
                BurnStorePayload::Latent(latent) => latent.byte_size(),
            })
            .sum()
    }
}

impl From<StoreError> for crate::error::BurnBackendError {
    fn from(err: StoreError) -> Self {
        Self::InvalidRequest(err.to_string())
    }
}

/// Cross-run cache for loaded model bundles.
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArrayDevice;

    fn sdxl_latent_space() -> LatentSpaceMetadata {
        LatentSpaceMetadata::sdxl_base()
    }

    fn zero_latent(batch: usize, channels: usize, h: usize, w: usize) -> Tensor<NdArray, 4> {
        Tensor::<NdArray, 4>::zeros([batch, channels, h, w], &NdArrayDevice::Cpu)
    }

    fn build_payload(batch: u32, h: u32, w: u32) -> BurnLatentPayload {
        BurnLatentPayload::new_ndarray(
            zero_latent(batch as usize, 4, (h / 8) as usize, (w / 8) as usize),
            sdxl_latent_space(),
            w,
            h,
            batch,
        )
    }

    #[test]
    fn store_insert_latent_registers_payload_and_run_pin() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.insert_latent(run_id.clone(), key.clone(), build_payload(1, 64, 64));
        assert_eq!(store.payload_count(), 1);
        assert_eq!(store.run_payload_count(&run_id), 1);
        assert!(store.contains_payload(&key));
    }

    #[test]
    fn store_get_latent_clones_handle_with_useful_metadata() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.insert_latent(run_id, key.clone(), build_payload(1, 64, 64));

        let latent = store.get_latent(&key).expect("latent payload");
        assert_eq!(latent.dims(), [1, 4, 8, 8]);
        assert_eq!(latent.width(), 64);
        assert_eq!(latent.height(), 64);
        assert_eq!(latent.batch(), 1);
        assert_eq!(latent.latent_space(), &LatentSpaceMetadata::sdxl_base());
        // 1 * 4 * 8 * 8 = 256 f32 elements = 1024 bytes
        assert_eq!(latent.byte_size(), 1024);
    }

    #[test]
    fn store_get_latent_reports_missing_key() {
        let store = BurnStore::new();
        let key = BackendPayloadKey::new("latent:missing");
        let err = store.get_latent(&key).unwrap_err();
        assert!(matches!(err, StoreError::PayloadNotFound { .. }));
        let msg = err.to_string();
        assert!(msg.contains("latent:missing"), "msg: {msg}");
        assert!(msg.contains("latent"), "msg: {msg}");
    }

    #[test]
    fn store_take_latent_removes_entry_and_returns_payload() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.insert_latent(run_id.clone(), key.clone(), build_payload(1, 64, 64));

        let taken = store.take_latent(&key).expect("latent payload");
        assert_eq!(taken.dims(), [1, 4, 8, 8]);
        assert!(!store.contains_payload(&key));
        assert_eq!(store.run_payload_count(&run_id), 0);
    }

    #[test]
    fn store_take_latent_on_missing_key_returns_error() {
        let store = BurnStore::new();
        let key = BackendPayloadKey::new("latent:absent");
        let err = store.take_latent(&key).unwrap_err();
        assert!(matches!(err, StoreError::PayloadNotFound { .. }));
    }

    #[test]
    fn store_release_payload_removes_by_key() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.insert_latent(run_id.clone(), key.clone(), build_payload(1, 64, 64));
        assert!(store.release_payload(&key));
        assert!(!store.contains_payload(&key));
        assert_eq!(store.run_payload_count(&run_id), 0);
    }

    #[test]
    fn store_cleanup_run_removes_latent_payloads() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-1");
        let key = BackendPayloadKey::new("latent:run-1:node-a");
        store.insert_latent(run_id.clone(), key.clone(), build_payload(1, 64, 64));
        store.cleanup_run(&run_id);
        assert_eq!(store.payload_count(), 0);
        assert_eq!(store.run_payload_count(&run_id), 0);
        assert!(!store.contains_payload(&key));
    }

    #[test]
    fn store_cleanup_run_does_not_affect_other_runs() {
        let store = BurnStore::new();
        let run_a = RunId::new("run-a");
        let run_b = RunId::new("run-b");
        let key_a = BackendPayloadKey::new("latent:run-a:node-a");
        let key_b = BackendPayloadKey::new("latent:run-b:node-a");
        store.insert_latent(run_a.clone(), key_a.clone(), build_payload(1, 64, 64));
        store.insert_latent(run_b.clone(), key_b.clone(), build_payload(1, 64, 64));

        store.cleanup_run(&run_a);
        assert_eq!(store.payload_count(), 1);
        assert!(!store.contains_payload(&key_a));
        assert!(store.contains_payload(&key_b));
    }

    #[test]
    fn store_latent_byte_size_sums_payloads() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-bytes");
        let first = BackendPayloadKey::new("latent:run-bytes:a");
        let second = BackendPayloadKey::new("latent:run-bytes:b");
        store.insert_latent(run_id.clone(), first, build_payload(1, 64, 64));
        store.insert_latent(run_id, second, build_payload(1, 64, 64));
        assert_eq!(store.payload_byte_size(), 2048);
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
            key: BackendPayloadKey::new("payload:misuse"),
            expected: "latent",
            actual: "image",
        };
        let msg = wrong_kind.to_string();
        assert!(msg.contains("payload:misuse"), "{msg}");
        assert!(msg.contains("image"), "{msg}");
        assert!(msg.contains("latent"), "{msg}");
    }

    #[test]
    fn store_cleanup_run_reports_removed_count() {
        let store = BurnStore::new();
        let run_a = RunId::new("run-a");
        let run_b = RunId::new("run-b");
        store.insert_latent(
            run_a.clone(),
            BackendPayloadKey::new("latent:run-a:a"),
            build_payload(1, 64, 64),
        );
        store.insert_latent(
            run_a.clone(),
            BackendPayloadKey::new("latent:run-a:b"),
            build_payload(1, 64, 64),
        );
        store.insert_latent(
            run_b.clone(),
            BackendPayloadKey::new("latent:run-b:a"),
            build_payload(1, 64, 64),
        );
        assert_eq!(store.payload_count(), 3);

        // Two payloads pinned to run_a are evicted; run_b is
        // untouched. The returned count is the per-run eviction
        // tally, not the residual store size.
        assert_eq!(store.cleanup_run(&run_a), 2);
        assert_eq!(store.payload_count(), 1);
        assert!(store.contains_payload(&BackendPayloadKey::new("latent:run-b:a")));

        // Unknown run id returns 0 and is a no-op.
        assert_eq!(store.cleanup_run(&RunId::new("run-missing")), 0);
    }

    #[test]
    fn store_payload_count_reflects_inserted_latents() {
        let store = BurnStore::new();
        assert_eq!(store.payload_count(), 0);
        store.insert_latent(
            RunId::new("run-1"),
            BackendPayloadKey::new("latent:run-1:a"),
            build_payload(1, 64, 64),
        );
        assert_eq!(store.payload_count(), 1);
        store.insert_latent(
            RunId::new("run-2"),
            BackendPayloadKey::new("latent:run-2:b"),
            build_payload(1, 64, 64),
        );
        assert_eq!(store.payload_count(), 2);
    }

    #[test]
    fn burn_latent_payload_preserves_supplied_latent_space_metadata() {
        let custom = LatentSpaceMetadata::new(
            reimagine_inference::LatentSpaceId::new("custom/test"),
            4,
            8,
            reimagine_core::model::TensorDType::F32,
            reimagine_inference::TensorLayout::Nchw,
        );
        let tensor = zero_latent(1, 4, 8, 8);
        let payload = BurnLatentPayload::new_ndarray(tensor, custom.clone(), 64, 64, 1);
        assert_eq!(payload.latent_space(), &custom);
        assert_eq!(payload.latent_space().id().as_str(), "custom/test");
    }
}
