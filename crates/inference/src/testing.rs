//! Fake / stub backend for tests.
//!
//! [`FakeBackend`] is a minimal [`InferenceBackend`] implementation
//! that returns canned typed responses. The module is only compiled
//! for crate unit tests or when the explicit `testing` feature is
//! enabled.
//!
//! Each capability is registered with a [`CannedCapabilityResponse`]
//! helper that constructs a typed response from a callback. Tests
//! can either build a response up front (when inputs are simple) or
//! capture the typed request and return a response derived from it.
//!
//! Downstream crates that need `FakeBackend` in integration tests
//! should enable the `testing` feature. Production code must not
//! depend on `FakeBackend`.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_core::model::{ArtifactId, ArtifactRef, SlotId};
use reimagine_inference_core::{
    BackendKind, CreateEmptyLatentRequest, CreateEmptyLatentResponse, DiffusionSampleRequest,
    DiffusionSampleResponse, ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest,
    ImageSaveResponse, InferenceBackend, InferenceBackendCapabilities, InferenceCapability,
    InferenceCapabilitySupport, InferenceError, LatentDecodeRequest, LatentDecodeResponse,
    LoadBundleRequest, LoadBundleResponse, TextEncodeRequest, TextEncodeResponse,
};

use crate::artifact_publisher::{ArtifactEventKind, ArtifactPublisher};
use crate::cancellation::NodeCancellation;

/// A canned response factory for a single capability.
pub struct CannedCapabilityResponse<Req, Resp> {
    factory: Box<dyn Fn(Req) -> Result<Resp, InferenceError> + Send + Sync>,
}

impl<Req, Resp> std::fmt::Debug for CannedCapabilityResponse<Req, Resp> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CannedCapabilityResponse").finish()
    }
}

impl<Req, Resp> CannedCapabilityResponse<Req, Resp> {
    /// Build a canned response that returns `response` for every call.
    pub fn always(response: Resp) -> Self
    where
        Resp: Clone + Send + Sync + 'static,
    {
        Self {
            factory: Box::new(move |_| Ok(response.clone())),
        }
    }

    /// Build a canned response that derives its response from the
    /// incoming request.
    pub fn from_request<F>(f: F) -> Self
    where
        F: Fn(Req) -> Result<Resp, InferenceError> + Send + Sync + 'static,
    {
        Self {
            factory: Box::new(f),
        }
    }

    fn run(&self, request: Req) -> Result<Resp, InferenceError> {
        (self.factory)(request)
    }
}

/// A minimal fake backend for tests.
///
/// Capabilities are registered via [`FakeBackend::load_bundle`],
/// [`FakeBackend::text_encode`], etc. Unregistered capabilities
/// return [`InferenceError::BackendNotImplemented`].
///
/// # Example
///
/// ```
/// use reimagine_inference::{FakeBackend, CannedCapabilityResponse};
/// use reimagine_inference::operation::InferenceCapability;
/// use reimagine_inference::{
///     BackendKind, BackendPayloadKey, BackendTensorHandle, RuntimeLatent,
/// };
/// use reimagine_core::model::{TensorDType, TensorShape};
/// use reimagine_inference_core::CreateEmptyLatentResponse;
///
/// let backend = FakeBackend::new("fake")
///     .create_empty_latent(CannedCapabilityResponse::always(
///         CreateEmptyLatentResponse::new(RuntimeLatent::new(
///             BackendTensorHandle::new(
///                 BackendKind::new("fake"),
///                 BackendPayloadKey::new("k"),
///                 TensorDType::F32,
///                 TensorShape::new(vec![1, 4, 8, 8]),
///                 "cpu",
///             ),
///             64,
///             64,
///             1,
///             4,
///         ))
///     ));
/// ```
pub struct FakeBackend {
    kind: BackendKind,
    canned: Mutex<CannedCapabilities>,
}

#[derive(Default)]
struct CannedCapabilities {
    load_bundle: Option<CannedCapabilityResponse<LoadBundleRequest, LoadBundleResponse>>,
    text_encode: Option<CannedCapabilityResponse<TextEncodeRequest, TextEncodeResponse>>,
    create_empty_latent:
        Option<CannedCapabilityResponse<CreateEmptyLatentRequest, CreateEmptyLatentResponse>>,
    diffusion_sample:
        Option<CannedCapabilityResponse<DiffusionSampleRequest, DiffusionSampleResponse>>,
    latent_decode: Option<CannedCapabilityResponse<LatentDecodeRequest, LatentDecodeResponse>>,
    image_save: Option<CannedCapabilityResponse<ImageSaveRequest, ImageSaveResponse>>,
    image_preview: Option<CannedCapabilityResponse<ImagePreviewRequest, ImagePreviewResponse>>,
}

impl FakeBackend {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: BackendKind::new(kind),
            canned: Mutex::new(CannedCapabilities::default()),
        }
    }

    pub fn load_bundle(
        self,
        response: CannedCapabilityResponse<LoadBundleRequest, LoadBundleResponse>,
    ) -> Self {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .load_bundle = Some(response);
        self
    }

    pub fn text_encode(
        self,
        response: CannedCapabilityResponse<TextEncodeRequest, TextEncodeResponse>,
    ) -> Self {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .text_encode = Some(response);
        self
    }

    pub fn create_empty_latent(
        self,
        response: CannedCapabilityResponse<CreateEmptyLatentRequest, CreateEmptyLatentResponse>,
    ) -> Self {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .create_empty_latent = Some(response);
        self
    }

    pub fn diffusion_sample(
        self,
        response: CannedCapabilityResponse<DiffusionSampleRequest, DiffusionSampleResponse>,
    ) -> Self {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .diffusion_sample = Some(response);
        self
    }

    pub fn latent_decode(
        self,
        response: CannedCapabilityResponse<LatentDecodeRequest, LatentDecodeResponse>,
    ) -> Self {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .latent_decode = Some(response);
        self
    }

    pub fn image_save(
        self,
        response: CannedCapabilityResponse<ImageSaveRequest, ImageSaveResponse>,
    ) -> Self {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .image_save = Some(response);
        self
    }

    pub fn image_preview(
        self,
        response: CannedCapabilityResponse<ImagePreviewRequest, ImagePreviewResponse>,
    ) -> Self {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .image_preview = Some(response);
        self
    }

    /// Insert a canned response at runtime (takes `&self`).
    pub fn insert_create_empty_latent(
        &self,
        response: CannedCapabilityResponse<CreateEmptyLatentRequest, CreateEmptyLatentResponse>,
    ) {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .create_empty_latent = Some(response);
    }

    pub fn insert_load_bundle(
        &self,
        response: CannedCapabilityResponse<LoadBundleRequest, LoadBundleResponse>,
    ) {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .load_bundle = Some(response);
    }

    pub fn insert_image_save(
        &self,
        response: CannedCapabilityResponse<ImageSaveRequest, ImageSaveResponse>,
    ) {
        self.canned
            .lock()
            .expect("fake backend poisoned")
            .image_save = Some(response);
    }

    /// Build a capability support entry for a given capability.
    fn capability_support(capability: InferenceCapability) -> InferenceCapabilitySupport {
        InferenceCapabilitySupport::new(capability)
    }

    fn supports_interned(&self) -> Vec<InferenceCapability> {
        let canned = self.canned.lock().expect("fake backend poisoned");
        let mut caps = Vec::new();
        if canned.load_bundle.is_some() {
            caps.push(InferenceCapability::LoadBundle);
        }
        if canned.text_encode.is_some() {
            caps.push(InferenceCapability::TextEncode);
        }
        if canned.create_empty_latent.is_some() {
            caps.push(InferenceCapability::CreateEmptyLatent);
        }
        if canned.diffusion_sample.is_some() {
            caps.push(InferenceCapability::DiffusionSample);
        }
        if canned.latent_decode.is_some() {
            caps.push(InferenceCapability::LatentDecode);
        }
        if canned.image_save.is_some() {
            caps.push(InferenceCapability::ImageSave);
        }
        if canned.image_preview.is_some() {
            caps.push(InferenceCapability::ImagePreview);
        }
        caps
    }
}

fn not_implemented(backend: &FakeBackend, capability: InferenceCapability) -> InferenceError {
    InferenceError::BackendNotImplemented {
        capability,
        backend_kind: backend.kind.to_string(),
        message: None,
    }
}

#[async_trait::async_trait]
impl InferenceBackend for FakeBackend {
    fn backend_kind(&self) -> &BackendKind {
        &self.kind
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        let mut caps = InferenceBackendCapabilities::new(self.kind.clone());
        for cap in self.supports_interned() {
            caps = caps.with_support(FakeBackend::capability_support(cap));
        }
        caps
    }

    async fn load_bundle(
        &self,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        let canned = self.canned.lock().expect("fake backend poisoned");
        match canned.load_bundle.as_ref() {
            Some(c) => c.run(request),
            None => Err(not_implemented(self, InferenceCapability::LoadBundle)),
        }
    }

    async fn text_encode(
        &self,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        let canned = self.canned.lock().expect("fake backend poisoned");
        match canned.text_encode.as_ref() {
            Some(c) => c.run(request),
            None => Err(not_implemented(self, InferenceCapability::TextEncode)),
        }
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        let canned = self.canned.lock().expect("fake backend poisoned");
        match canned.create_empty_latent.as_ref() {
            Some(c) => c.run(request),
            None => Err(not_implemented(
                self,
                InferenceCapability::CreateEmptyLatent,
            )),
        }
    }

    async fn diffusion_sample(
        &self,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        let canned = self.canned.lock().expect("fake backend poisoned");
        match canned.diffusion_sample.as_ref() {
            Some(c) => c.run(request),
            None => Err(not_implemented(self, InferenceCapability::DiffusionSample)),
        }
    }

    async fn latent_decode(
        &self,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        let canned = self.canned.lock().expect("fake backend poisoned");
        match canned.latent_decode.as_ref() {
            Some(c) => c.run(request),
            None => Err(not_implemented(self, InferenceCapability::LatentDecode)),
        }
    }

    async fn image_save(
        &self,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        let canned = self.canned.lock().expect("fake backend poisoned");
        match canned.image_save.as_ref() {
            Some(c) => c.run(request),
            None => Err(not_implemented(self, InferenceCapability::ImageSave)),
        }
    }

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        let canned = self.canned.lock().expect("fake backend poisoned");
        match canned.image_preview.as_ref() {
            Some(c) => c.run(request),
            None => Err(not_implemented(self, InferenceCapability::ImagePreview)),
        }
    }
}

impl std::fmt::Debug for FakeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeBackend")
            .field("kind", &self.kind)
            .field("registered_capabilities", &self.supports_interned().len())
            .finish()
    }
}

// Helper: stand-alone canned response builders for common cases.
impl CannedCapabilities {
    #[allow(dead_code)]
    fn _referenced(&self) -> bool {
        self.load_bundle.is_some() || self.text_encode.is_some()
    }
}

/// No-op [`ArtifactPublisher`] for tests.
///
/// Records nothing, returns a deterministic `ArtifactId` so executors
/// that observe the returned id don't panic. Tests that need to assert
/// on recorded artifacts should wrap a different publisher.
#[derive(Debug, Default, Clone)]
pub struct NoopArtifactPublisher {
    counter: Arc<AtomicU64>,
}

impl NoopArtifactPublisher {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ArtifactPublisher for NoopArtifactPublisher {
    async fn record(
        &self,
        _slot_id: SlotId,
        _reference: ArtifactRef,
        _kind: ArtifactEventKind,
    ) -> Option<ArtifactId> {
        let index = self.counter.fetch_add(1, Ordering::Relaxed);
        Some(ArtifactId::new(format!("noop-{index}")))
    }
}

/// One artifact observation captured by [`RecordingArtifactPublisher`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedArtifact {
    pub slot_id: SlotId,
    pub reference: ArtifactRef,
    pub kind: ArtifactEventKind,
}

/// [`ArtifactPublisher`] implementation that records every call into
/// an in-memory `Vec` so tests can assert on what executors published.
#[derive(Debug, Default)]
pub struct RecordingArtifactPublisher {
    records: Arc<Mutex<Vec<RecordedArtifact>>>,
    counter: Arc<AtomicU64>,
}

impl RecordingArtifactPublisher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> Vec<RecordedArtifact> {
        self.records.lock().expect("records poisoned").clone()
    }
}

#[async_trait]
impl ArtifactPublisher for RecordingArtifactPublisher {
    async fn record(
        &self,
        slot_id: SlotId,
        reference: ArtifactRef,
        kind: ArtifactEventKind,
    ) -> Option<ArtifactId> {
        self.records
            .lock()
            .expect("records poisoned")
            .push(RecordedArtifact {
                slot_id,
                reference,
                kind,
            });
        let index = self.counter.fetch_add(1, Ordering::Relaxed);
        Some(ArtifactId::new(format!("rec-{index}")))
    }
}

/// No-op [`NodeCancellation`] for tests.
///
/// Polls the cancelled flag with a short `tokio::time::sleep` so
/// executors that `await cancellation().cancelled()` yield back to the
/// runtime. Tests that need a more responsive signal can call
/// [`cancel`](Self::cancel) before the await.
#[derive(Debug, Clone)]
pub struct NoopNodeCancellation {
    cancelled: Arc<AtomicBool>,
}

impl Default for NoopNodeCancellation {
    fn default() -> Self {
        Self::new()
    }
}

impl NoopNodeCancellation {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

#[async_trait]
impl NodeCancellation for NoopNodeCancellation {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    async fn cancelled(&self) {
        while !self.is_cancelled() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
}
