use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use burn_tensor::{Tensor, TensorData};
use reimagine_core::model::{ModelId, RunId};
use reimagine_inference::{BackendPayloadKey, LatentSpaceMetadata};

use crate::active_backend::ActiveBurnBackend;
use crate::models::stable_diffusion::sdxl::{
    BurnLoadedModelBundle, BurnSdxlSourceSignature, BurnSdxlTokenizedPromptPair,
};

/// Backend-owned latent payload wrapper.
#[derive(Debug, Clone)]
pub struct BurnLatentPayload {
    tensor: BurnLatentTensor,
    latent_space: LatentSpaceMetadata,
    width: u32,
    height: u32,
    batch: u32,
}

#[derive(Debug, Clone)]
pub struct BurnLatentTensor(Box<Tensor<ActiveBurnBackend, 4>>);

impl BurnLatentPayload {
    /// Build a backend-owned latent payload from an active WGPU/Flex tensor.
    pub fn new_active(
        tensor: Tensor<ActiveBurnBackend, 4>,
        latent_space: LatentSpaceMetadata,
        width: u32,
        height: u32,
        batch: u32,
    ) -> Self {
        Self {
            tensor: BurnLatentTensor(Box::new(tensor)),
            latent_space,
            width,
            height,
            batch,
        }
    }

    pub fn into_active_tensor(
        self,
    ) -> Result<Tensor<ActiveBurnBackend, 4>, crate::error::BurnBackendError> {
        Ok(*self.tensor.0)
    }

    pub fn active_tensor(&self) -> Option<&Tensor<ActiveBurnBackend, 4>> {
        Some(self.tensor.0.as_ref())
    }

    #[cfg(test)]
    pub(crate) fn is_active_backend(&self) -> bool {
        true
    }

    /// Shape of the stored tensor as a `[batch, channels, h, w]` slice.
    pub fn dims(&self) -> [usize; 4] {
        self.tensor.0.shape().dims()
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

    /// Approximate byte size of the payload.
    pub fn byte_size(&self) -> usize {
        self.dims().iter().product::<usize>() * std::mem::size_of::<f32>()
    }

    /// Pull the data buffer out of the tensor.
    pub fn to_data(&self) -> TensorData {
        self.tensor.0.to_data()
    }
}

/// Backend-private metadata captured for a text-encode preflight.
///
/// This type lives only inside the Burn backend. It is richer than
/// the public inference-layer `ExecutionConditioning` because the
/// store must also remember facts that are only relevant inside
/// Burn (per-role tokenizer identity, source signature, model
/// series/variant, and the requested sequence length). Keeping this
/// metadata attached to the payload lets downstream Burn operations
/// resolve the conditioning without having to re-run the preflight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnConditioningMetadata {
    model_id: ModelId,
    series: String,
    variant: String,
    sequence_length: u32,
    pooled_embedding_available: bool,
    primary_tokenizer_id: String,
    secondary_tokenizer_id: String,
    source_signature: BurnSdxlSourceSignature,
}

impl BurnConditioningMetadata {
    /// Build a metadata record from a loaded bundle and the
    /// resolved tokenizer resources. The tokenizer ids are the
    /// resolved file paths so the record is reproducible from the
    /// preflight inputs alone.
    pub fn from_bundle(
        bundle: &BurnLoadedModelBundle,
        sequence_length: u32,
        primary_tokenizer_id: String,
        secondary_tokenizer_id: String,
    ) -> Self {
        Self {
            model_id: bundle_model_id(bundle).clone(),
            series: "stable_diffusion".to_owned(),
            variant: "sdxl".to_owned(),
            sequence_length,
            pooled_embedding_available: true,
            primary_tokenizer_id,
            secondary_tokenizer_id,
            source_signature: bundle.source_signature().clone(),
        }
    }

    /// Test-only constructor used by conditioning-payload shape
    /// tests. Production code must not assemble fake metadata
    /// records.
    #[cfg(test)]
    pub(crate) fn test_only(
        model_id: ModelId,
        sequence_length: u32,
        primary_tokenizer_id: String,
        secondary_tokenizer_id: String,
    ) -> Self {
        Self {
            model_id,
            series: "stable_diffusion".to_owned(),
            variant: "sdxl".to_owned(),
            sequence_length,
            pooled_embedding_available: true,
            primary_tokenizer_id,
            secondary_tokenizer_id,
            source_signature: BurnSdxlSourceSignature::empty(),
        }
    }

    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    pub fn series(&self) -> &str {
        &self.series
    }

    pub fn variant(&self) -> &str {
        &self.variant
    }

    pub fn sequence_length(&self) -> u32 {
        self.sequence_length
    }

    pub fn pooled_embedding_available(&self) -> bool {
        self.pooled_embedding_available
    }

    pub fn primary_tokenizer_id(&self) -> &str {
        &self.primary_tokenizer_id
    }

    pub fn secondary_tokenizer_id(&self) -> &str {
        &self.secondary_tokenizer_id
    }

    pub fn source_signature(&self) -> BurnSdxlSourceSignature {
        self.source_signature.clone()
    }
}

/// Output from a CLIP text encoder forward pass.
#[derive(Debug, Clone)]
pub enum ClipOutputs {
    /// Active WGPU/Flex backend path for Burn-native Module graph outputs.
    Active {
        text_embeddings: Box<burn_tensor::Tensor<ActiveBurnBackend, 3>>,
        pooled_embeddings: Option<Box<burn_tensor::Tensor<ActiveBurnBackend, 2>>>,
    },
}

impl ClipOutputs {
    pub(crate) fn active(
        text_embeddings: burn_tensor::Tensor<ActiveBurnBackend, 3>,
        pooled_embeddings: Option<burn_tensor::Tensor<ActiveBurnBackend, 2>>,
    ) -> Self {
        Self::Active {
            text_embeddings: Box::new(text_embeddings),
            pooled_embeddings: pooled_embeddings.map(Box::new),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_active_backend(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    pub(crate) fn text_dims(&self) -> [usize; 3] {
        match self {
            Self::Active {
                text_embeddings, ..
            } => text_embeddings.shape().dims(),
        }
    }

    pub(crate) fn active_text_embeddings(
        &self,
    ) -> Result<Tensor<ActiveBurnBackend, 3>, crate::error::BurnBackendError> {
        match self {
            Self::Active {
                text_embeddings, ..
            } => Ok(text_embeddings.as_ref().clone()),
        }
    }

    pub(crate) fn active_pooled_embeddings(
        &self,
    ) -> Result<Tensor<ActiveBurnBackend, 2>, crate::error::BurnBackendError> {
        match self {
            Self::Active {
                pooled_embeddings, ..
            } => pooled_embeddings
                .as_ref()
                .map(|tensor| tensor.as_ref().clone())
                .ok_or_else(|| {
                    crate::error::BurnBackendError::InvalidRequest(
                        "diffusion.sample requires stored pooled text encoder embeddings"
                            .to_owned(),
                    )
                }),
        }
    }

    pub(crate) fn pooled_dims(&self) -> Option<[usize; 2]> {
        match self {
            Self::Active {
                pooled_embeddings, ..
            } => pooled_embeddings
                .as_ref()
                .map(|tensor| tensor.shape().dims()),
        }
    }

    pub(crate) fn byte_size(&self) -> usize {
        let text_bytes = self.text_dims().iter().product::<usize>() * std::mem::size_of::<f32>();
        let pooled_bytes = self.pooled_dims().map_or(0, |dims| {
            dims.iter().product::<usize>() * std::mem::size_of::<f32>()
        });
        text_bytes + pooled_bytes
    }
}

/// Backend-owned conditioning payload wrapper.
///
/// V1 carries deterministic tokenization output plus optional CLIP
/// embeddings from the active Burn backend. Component-backed
/// `text.encode` fills the embeddings; no-component test bundles may
/// use synthetic shape-correct placeholders.
#[derive(Debug, Clone)]
pub struct BurnConditioningPayload {
    pub(crate) metadata: BurnConditioningMetadata,
    pub(crate) tokenized_prompts: BurnSdxlTokenizedPromptPair,
    pub(crate) embeddings: Option<ClipOutputs>,
}

impl BurnConditioningPayload {
    /// Build a payload from a tokenized prompt pair and metadata.
    /// Visible to the burn/08a preflight so the preconditioned
    /// record can be assembled for the future store insertion; the
    /// preflight deliberately does not insert the payload into the
    /// store — CLIP forward is not yet wired.
    pub(crate) fn from_tokenized(
        metadata: BurnConditioningMetadata,
        tokenized_prompts: BurnSdxlTokenizedPromptPair,
    ) -> Self {
        Self {
            metadata,
            tokenized_prompts,
            embeddings: None,
        }
    }

    /// Attach real CLIP forward outputs to the payload.
    pub(crate) fn with_embeddings(mut self, embeddings: ClipOutputs) -> Self {
        self.embeddings = Some(embeddings);
        self
    }

    /// Test-only constructor. Production code paths must not build
    /// fake successful conditioning payloads; this seam exists so
    /// store-shape tests can exercise the conditioning variant
    /// without depending on a real CLIP forward pass.
    #[cfg(test)]
    pub(crate) fn test_only(
        metadata: BurnConditioningMetadata,
        tokenized_prompts: BurnSdxlTokenizedPromptPair,
    ) -> Self {
        Self::from_tokenized(metadata, tokenized_prompts)
    }

    pub fn metadata(&self) -> &BurnConditioningMetadata {
        &self.metadata
    }

    pub fn tokenized_prompts(&self) -> &BurnSdxlTokenizedPromptPair {
        &self.tokenized_prompts
    }

    #[cfg(test)]
    pub(crate) fn embeddings(&self) -> Option<&ClipOutputs> {
        self.embeddings.as_ref()
    }

    /// Approximate byte size of the payload: tokenized prompts plus
    /// any stored embedding tensors.
    pub fn byte_size(&self) -> usize {
        let tokens = self.tokenized_prompts.clip_l.token_ids.len()
            + self.tokenized_prompts.clip_g.token_ids.len();
        let masks = self.tokenized_prompts.clip_l.attention_mask.len()
            + self.tokenized_prompts.clip_g.attention_mask.len();
        let token_size = (tokens + masks) * std::mem::size_of::<u32>();
        let embed_size = self.embeddings.as_ref().map_or(0, ClipOutputs::byte_size);
        token_size + embed_size
    }

    pub(crate) fn active_text_embeddings(
        &self,
    ) -> Result<Tensor<ActiveBurnBackend, 3>, crate::error::BurnBackendError> {
        self.embeddings
            .as_ref()
            .ok_or_else(|| {
                crate::error::BurnBackendError::InvalidRequest(
                    "diffusion.sample requires stored text encoder embeddings".to_owned(),
                )
            })?
            .active_text_embeddings()
    }

    pub(crate) fn active_pooled_embeddings(
        &self,
    ) -> Result<Tensor<ActiveBurnBackend, 2>, crate::error::BurnBackendError> {
        self.embeddings
            .as_ref()
            .ok_or_else(|| {
                crate::error::BurnBackendError::InvalidRequest(
                    "diffusion.sample requires stored text encoder embeddings".to_owned(),
                )
            })?
            .active_pooled_embeddings()
    }
}

/// Backend-owned image payload wrapper.
///
/// Holds a decoded VAE output tensor in NCHW float32 format with
/// metadata about the image dimensions and color space.
#[derive(Debug, Clone)]
pub struct BurnImagePayload {
    tensor: BurnImageTensor,
    width: u32,
    height: u32,
    batch: u32,
    color_space: String,
}

#[derive(Debug, Clone)]
pub struct BurnImageTensor(Box<Tensor<ActiveBurnBackend, 4>>);

impl BurnImagePayload {
    pub fn new_active(
        tensor: Tensor<ActiveBurnBackend, 4>,
        width: u32,
        height: u32,
        batch: u32,
        color_space: impl Into<String>,
    ) -> Self {
        Self {
            tensor: BurnImageTensor(Box::new(tensor)),
            width,
            height,
            batch,
            color_space: color_space.into(),
        }
    }

    pub fn dims(&self) -> [usize; 4] {
        self.tensor.0.shape().dims()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn color_space(&self) -> &str {
        &self.color_space
    }

    pub fn byte_size(&self) -> usize {
        self.dims().iter().product::<usize>() * std::mem::size_of::<f32>()
    }

    pub fn to_data(&self) -> TensorData {
        self.tensor.0.to_data()
    }

    #[cfg(test)]
    pub(crate) fn is_active_backend(&self) -> bool {
        true
    }
}

/// Pull the model id out of an [`BurnLoadedModelBundle`] without
/// re-exporting the inner SDXL enum variant outside the burn
/// crate. Both currently supported variants are SDXL bundles, so
/// this accessor is exhaustive.
fn bundle_model_id(bundle: &BurnLoadedModelBundle) -> &ModelId {
    match bundle {
        BurnLoadedModelBundle::StableDiffusionSdxl(bundle) => bundle.as_ref().model_id(),
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
#[allow(clippy::large_enum_variant)]
enum BurnStorePayload {
    Latent(BurnLatentPayload),
    Conditioning(BurnConditioningPayload),
    Image(BurnImagePayload),
}

impl BurnStorePayload {
    fn kind(&self) -> &'static str {
        match self {
            Self::Latent(_) => "latent",
            Self::Conditioning(_) => "conditioning",
            Self::Image(_) => "image",
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

    /// Insert a real Burn-private conditioning payload and pin it
    /// to the run. The payload shape is defined by burn/08a and
    /// will be populated with real CLIP forward outputs by
    /// burn/08f. Production `text.encode` does not call this
    /// method until burn/08f lands.
    pub fn insert_conditioning(
        &self,
        run_id: RunId,
        key: BackendPayloadKey,
        payload: BurnConditioningPayload,
    ) {
        let mut inner = self.inner.lock().expect("store poisoned");
        inner
            .payloads
            .insert(key.clone(), BurnStorePayload::Conditioning(payload));
        inner.run_index.entry(run_id).or_default().push(key);
    }

    /// Borrow a conditioning payload by key. The lock is released
    /// as soon as the cheap payload clone completes; no work
    /// happens outside the lock beyond the return.
    #[allow(unreachable_patterns)]
    pub fn get_conditioning(
        &self,
        key: &BackendPayloadKey,
    ) -> Result<BurnConditioningPayload, StoreError> {
        let inner = self.inner.lock().expect("store poisoned");
        let cloned = inner.payloads.get(key).cloned();
        drop(inner);
        match cloned {
            Some(BurnStorePayload::Conditioning(payload)) => Ok(payload),
            Some(other) => Err(StoreError::WrongPayloadKind {
                key: key.clone(),
                expected: "conditioning",
                actual: other.kind(),
            }),
            None => Err(StoreError::PayloadNotFound {
                key: key.clone(),
                expected: "conditioning",
            }),
        }
    }

    /// Take ownership of a conditioning payload, removing it from
    /// the store and unpinning it from any run index.
    #[allow(unreachable_patterns)]
    pub fn take_conditioning(
        &self,
        key: &BackendPayloadKey,
    ) -> Result<BurnConditioningPayload, StoreError> {
        let mut inner = self.inner.lock().expect("store poisoned");
        match inner.payloads.remove(key) {
            Some(BurnStorePayload::Conditioning(payload)) => {
                for keys in inner.run_index.values_mut() {
                    keys.retain(|k| k != key);
                }
                Ok(payload)
            }
            Some(other) => Err(StoreError::WrongPayloadKind {
                key: key.clone(),
                expected: "conditioning",
                actual: other.kind(),
            }),
            None => Err(StoreError::PayloadNotFound {
                key: key.clone(),
                expected: "conditioning",
            }),
        }
    }

    /// Insert an image payload and pin it to the run.
    pub fn insert_image(&self, run_id: RunId, key: BackendPayloadKey, payload: BurnImagePayload) {
        let mut inner = self.inner.lock().expect("store poisoned");
        inner
            .payloads
            .insert(key.clone(), BurnStorePayload::Image(payload));
        inner.run_index.entry(run_id).or_default().push(key);
    }

    /// Borrow an image payload by key.
    #[allow(unreachable_patterns)]
    pub fn get_image(&self, key: &BackendPayloadKey) -> Result<BurnImagePayload, StoreError> {
        let inner = self.inner.lock().expect("store poisoned");
        let cloned = inner.payloads.get(key).cloned();
        drop(inner);
        match cloned {
            Some(BurnStorePayload::Image(payload)) => Ok(payload),
            Some(other) => Err(StoreError::WrongPayloadKind {
                key: key.clone(),
                expected: "image",
                actual: other.kind(),
            }),
            None => Err(StoreError::PayloadNotFound {
                key: key.clone(),
                expected: "image",
            }),
        }
    }

    /// Take ownership of an image payload.
    #[allow(unreachable_patterns)]
    pub fn take_image(&self, key: &BackendPayloadKey) -> Result<BurnImagePayload, StoreError> {
        let mut inner = self.inner.lock().expect("store poisoned");
        match inner.payloads.remove(key) {
            Some(BurnStorePayload::Image(payload)) => {
                for keys in inner.run_index.values_mut() {
                    keys.retain(|k| k != key);
                }
                Ok(payload)
            }
            Some(other) => Err(StoreError::WrongPayloadKind {
                key: key.clone(),
                expected: "image",
                actual: other.kind(),
            }),
            None => Err(StoreError::PayloadNotFound {
                key: key.clone(),
                expected: "image",
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

    /// Remove every payload and run pin owned by this backend process.
    ///
    /// Worker shutdown uses this process-scoped operation because no payload
    /// authority may survive the worker incarnation.
    pub fn cleanup_all(&self) -> usize {
        let mut inner = self.inner.lock().expect("store poisoned");
        let removed = inner.payloads.len();
        inner.payloads.clear();
        inner.run_index.clear();
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
                BurnStorePayload::Conditioning(conditioning) => conditioning.byte_size(),
                BurnStorePayload::Image(image) => image.byte_size(),
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

    /// Return the bundle currently registered for a model id,
    /// without checking the source signature. Used by the
    /// burn/08a text.encode preflight to confirm a bundle is
    /// loaded before the CLIP forward pass becomes real.
    pub fn get_bundle(&self, model_id: &ModelId) -> Option<Arc<BurnLoadedModelBundle>> {
        self.bundles
            .lock()
            .expect("model cache poisoned")
            .get(model_id)
            .cloned()
    }

    /// Check whether a bundle is currently registered for a
    /// model id. Used by the burn/08a preflight before the
    /// signature-bearing lookup.
    pub fn contains(&self, model_id: &ModelId) -> bool {
        self.bundles
            .lock()
            .expect("model cache poisoned")
            .contains_key(model_id)
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

    /// Remove all model bundles cached by this backend process.
    pub fn clear(&self) -> usize {
        let mut bundles = self.bundles.lock().expect("model cache poisoned");
        let removed = bundles.len();
        bundles.clear();
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;
    use burn_tensor::Tensor;

    fn sdxl_latent_space() -> LatentSpaceMetadata {
        LatentSpaceMetadata::sdxl_base()
    }

    fn active_device_for_test() -> burn_tensor::Device<ActiveBurnBackend> {
        let config = BurnBackendConfig::new("/models", "/output");
        active_device(config.device())
    }

    fn active_zero_tensor(
        batch: usize,
        channels: usize,
        h: usize,
        w: usize,
    ) -> Tensor<ActiveBurnBackend, 4> {
        Tensor::<ActiveBurnBackend, 4>::zeros([batch, channels, h, w], &active_device_for_test())
    }

    fn build_payload(batch: u32, h: u32, w: u32) -> BurnLatentPayload {
        BurnLatentPayload::new_active(
            active_zero_tensor(batch as usize, 4, (h / 8) as usize, (w / 8) as usize),
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
    fn store_cleanup_all_removes_payloads_from_every_run() {
        let store = BurnStore::new();
        store.insert_latent(
            RunId::new("run-a"),
            BackendPayloadKey::new("a"),
            build_payload(1, 64, 64),
        );
        store.insert_latent(
            RunId::new("run-b"),
            BackendPayloadKey::new("b"),
            build_payload(1, 64, 64),
        );
        assert_eq!(store.cleanup_all(), 2);
        assert_eq!(store.payload_count(), 0);
        assert_eq!(store.cleanup_all(), 0);
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
        let tensor = active_zero_tensor(1, 4, 8, 8);
        let payload = BurnLatentPayload::new_active(tensor, custom.clone(), 64, 64, 1);
        assert_eq!(payload.latent_space(), &custom);
        assert_eq!(payload.latent_space().id().as_str(), "custom/test");
    }

    #[test]
    fn burn_latent_payload_can_store_active_backend_tensor() {
        let tensor = Tensor::<ActiveBurnBackend, 4>::zeros([1, 4, 8, 8], &active_device_for_test());
        let payload =
            BurnLatentPayload::new_active(tensor, LatentSpaceMetadata::sdxl_base(), 64, 64, 1);

        assert!(payload.is_active_backend());
        assert_eq!(payload.dims(), [1, 4, 8, 8]);
        assert_eq!(payload.byte_size(), 1024);
    }

    #[test]
    fn burn_image_payload_can_store_active_backend_tensor() {
        let tensor =
            Tensor::<ActiveBurnBackend, 4>::zeros([1, 3, 64, 64], &active_device_for_test());
        let payload = BurnImagePayload::new_active(tensor, 64, 64, 1, "rgb");

        assert!(payload.is_active_backend());
        assert_eq!(payload.dims(), [1, 3, 64, 64]);
        assert_eq!(payload.byte_size(), 3 * 64 * 64 * 4);
    }

    fn build_conditioning(model_id: &str, primary: u32, secondary: u32) -> BurnConditioningPayload {
        let metadata = BurnConditioningMetadata::test_only(
            ModelId::new(model_id),
            77,
            format!("primary://{primary}"),
            format!("secondary://{secondary}"),
        );
        use crate::models::stable_diffusion::sdxl::BurnSdxlTokenizedPrompt;
        let pair = BurnSdxlTokenizedPromptPair {
            clip_l: BurnSdxlTokenizedPrompt {
                token_ids: vec![primary; 77],
                attention_mask: vec![1; 77],
            },
            clip_g: BurnSdxlTokenizedPrompt {
                token_ids: vec![secondary; 77],
                attention_mask: vec![1; 77],
            },
        };
        BurnConditioningPayload::test_only(metadata, pair)
    }

    #[test]
    fn store_insert_conditioning_registers_payload_and_run_pin() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-cond");
        let key = BackendPayloadKey::new("conditioning:run-cond:node-a");
        store.insert_conditioning(
            run_id.clone(),
            key.clone(),
            build_conditioning("sdxl-base", 1, 2),
        );
        assert_eq!(store.payload_count(), 1);
        assert_eq!(store.run_payload_count(&run_id), 1);
        assert!(store.contains_payload(&key));
    }

    #[test]
    fn store_get_conditioning_returns_inserted_payload() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-cond");
        let key = BackendPayloadKey::new("conditioning:run-cond:node-a");
        let original = build_conditioning("sdxl-base", 11, 22);
        store.insert_conditioning(run_id, key.clone(), original.clone());

        let got = store.get_conditioning(&key).expect("conditioning payload");
        assert_eq!(got.metadata().model_id().as_str(), "sdxl-base");
        assert_eq!(got.tokenized_prompts().clip_l.token_ids[0], 11);
        assert_eq!(got.tokenized_prompts().clip_g.token_ids[0], 22);
        assert_eq!(got.byte_size(), 4 * 77 * std::mem::size_of::<u32>());
    }

    #[test]
    fn conditioning_payload_can_store_active_backend_embeddings() {
        let device = active_device_for_test();
        let text = Tensor::<ActiveBurnBackend, 3>::zeros([1, 77, 2048], &device);
        let pooled = Tensor::<ActiveBurnBackend, 2>::zeros([1, 1280], &device);
        let payload = build_conditioning("sdxl-base", 11, 22)
            .with_embeddings(ClipOutputs::active(text, Some(pooled)));

        let embeddings = payload.embeddings().expect("active embeddings");

        assert!(embeddings.is_active_backend());
        assert_eq!(embeddings.text_dims(), [1, 77, 2048]);
        assert_eq!(embeddings.pooled_dims(), Some([1, 1280]));
    }

    #[test]
    fn conditioning_payload_exposes_active_pooled_embeddings() {
        let device = active_device_for_test();
        let text = Tensor::<ActiveBurnBackend, 3>::zeros([1, 77, 2048], &device);
        let pooled = Tensor::<ActiveBurnBackend, 2>::ones([1, 1280], &device);
        let payload = build_conditioning("sdxl-base", 11, 22)
            .with_embeddings(ClipOutputs::active(text, Some(pooled)));

        let pooled = payload
            .active_pooled_embeddings()
            .expect("active pooled embeddings");

        assert_eq!(pooled.dims(), [1, 1280]);
    }

    #[test]
    fn conditioning_payload_reports_missing_active_pooled_embeddings() {
        let device = active_device_for_test();
        let text = Tensor::<ActiveBurnBackend, 3>::zeros([1, 77, 2048], &device);
        let payload = build_conditioning("sdxl-base", 11, 22)
            .with_embeddings(ClipOutputs::active(text, None));

        let err = payload
            .active_pooled_embeddings()
            .expect_err("missing pooled embeddings should be rejected");
        let msg = err.to_string();

        assert!(msg.contains("pooled"), "msg: {msg}");
    }

    #[test]
    fn store_get_conditioning_reports_missing_key() {
        let store = BurnStore::new();
        let key = BackendPayloadKey::new("conditioning:missing");
        let err = store.get_conditioning(&key).unwrap_err();
        assert!(matches!(err, StoreError::PayloadNotFound { .. }));
        let msg = err.to_string();
        assert!(msg.contains("conditioning"), "msg: {msg}");
        assert!(msg.contains("conditioning:missing"), "msg: {msg}");
    }

    #[test]
    fn store_take_conditioning_removes_entry_and_returns_payload() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-cond");
        let key = BackendPayloadKey::new("conditioning:run-cond:node-a");
        store.insert_conditioning(
            run_id.clone(),
            key.clone(),
            build_conditioning("sdxl-base", 3, 4),
        );

        let taken = store.take_conditioning(&key).expect("conditioning payload");
        assert_eq!(taken.metadata().model_id().as_str(), "sdxl-base");
        assert!(!store.contains_payload(&key));
        assert_eq!(store.run_payload_count(&run_id), 0);
    }

    #[test]
    fn store_get_conditioning_on_latent_key_reports_wrong_kind() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-cond");
        let key = BackendPayloadKey::new("latent:run-cond:node-a");
        store.insert_latent(run_id, key.clone(), build_payload(1, 64, 64));

        let err = store.get_conditioning(&key).unwrap_err();
        assert!(matches!(err, StoreError::WrongPayloadKind { .. }));
        let msg = err.to_string();
        assert!(msg.contains("latent"), "msg: {msg}");
        assert!(msg.contains("conditioning"), "msg: {msg}");
    }

    #[test]
    fn store_release_payload_evicts_conditioning() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-cond");
        let key = BackendPayloadKey::new("conditioning:run-cond:node-a");
        store.insert_conditioning(run_id.clone(), key.clone(), build_conditioning("m", 1, 2));
        assert!(store.release_payload(&key));
        assert!(!store.contains_payload(&key));
        assert_eq!(store.run_payload_count(&run_id), 0);
    }

    #[test]
    fn store_cleanup_run_evicts_conditioning_payloads() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-cond");
        let key = BackendPayloadKey::new("conditioning:run-cond:node-a");
        store.insert_conditioning(run_id.clone(), key.clone(), build_conditioning("m", 1, 2));
        assert_eq!(store.cleanup_run(&run_id), 1);
        assert!(!store.contains_payload(&key));
    }

    #[test]
    fn store_payload_byte_size_sums_conditioning_payloads() {
        let store = BurnStore::new();
        let run_id = RunId::new("run-cond");
        let key_a = BackendPayloadKey::new("conditioning:run-cond:a");
        let key_b = BackendPayloadKey::new("conditioning:run-cond:b");
        store.insert_conditioning(run_id.clone(), key_a, build_conditioning("m-a", 1, 2));
        store.insert_conditioning(run_id, key_b, build_conditioning("m-b", 3, 4));
        // Each conditioning payload is 4 * 77 u32 values.
        assert_eq!(
            store.payload_byte_size(),
            2 * 4 * 77 * std::mem::size_of::<u32>()
        );
    }

    #[test]
    fn conditioning_metadata_test_only_captures_inputs() {
        let metadata = BurnConditioningMetadata::test_only(
            ModelId::new("sdxl-base"),
            77,
            "primary://p".to_owned(),
            "secondary://s".to_owned(),
        );
        assert_eq!(metadata.model_id().as_str(), "sdxl-base");
        assert_eq!(metadata.series(), "stable_diffusion");
        assert_eq!(metadata.variant(), "sdxl");
        assert_eq!(metadata.sequence_length(), 77);
        assert!(metadata.pooled_embedding_available());
        assert_eq!(metadata.primary_tokenizer_id(), "primary://p");
        assert_eq!(metadata.secondary_tokenizer_id(), "secondary://s");
        // The test-only signature is empty — verifying it via
        // Debug output avoids leaking the private component
        // signature type through a public accessor.
        let debug = format!("{:?}", metadata.source_signature());
        assert!(debug.contains("components"), "debug: {debug}");
    }
}
