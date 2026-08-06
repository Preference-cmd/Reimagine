//! Scripted backend for integration testing.
//!
//! [`ScriptedBackend`] extends [`super::FakeBackend`]'s canned-response
//! model with a **per-capability script**: a `Vec` of
//! `Result<Response, InferenceError>` steps that are consumed strictly in
//! order, one per capability call. This lets tests script error
//! propagation (e.g. "step 2 of 3 fails with `OutOfMemory`") and
//! mid-pipeline failure without writing a bespoke backend per test.
//!
//! Two test-oriented knobs are available:
//!
//! - [`ScriptedBackend::with_step_delay`] pauses every scripted step for a
//!   fixed [`Duration`] before it returns. The pause is cancellation-aware:
//!   when the invocation's [`NodeCancellation`] trips while a step is
//!   waiting (or held), the backend returns [`InferenceError::Cancelled`]
//!   immediately instead of finishing the step.
//! - [`ScriptedBackend::with_hold`] installs a [`ScriptedHold`] gate. The
//!   next scripted step signals `entered` and then blocks until the test
//!   calls [`ScriptedHold::release`] or the invocation is cancelled. This
//!   is the primitive used to interleave a cancellation with a slow
//!   backend operation.
//!
//! The invocation's cancellation handle is captured in
//! [`InferenceBackend::admit_invocation`], which the default
//! `*_with_invocation` trait methods call before the capability method.
//! Executors and the router always go through `*_with_invocation`, so the
//! delay/hold machinery observes the real run-level and node-level
//! cancellation tokens.
//!
//! # Exhaustion
//!
//! When a script runs out of steps the call either returns a deterministic
//! [`InferenceError::BackendExecutionFailed`] (default) or panics with a
//! descriptive message when [`ScriptExhaustion::Panic`] is configured.
//!
//! # API deviation from the roadmap sketch
//!
//! The roadmap sketched `ScriptedBackend::new(vec![Ok(response),
//! Err(error), ...])` with a flat step list. The typed [`InferenceBackend`]
//! contract makes a flat list impossible in Rust — each capability has its
//! own request/response pair — so the script is declared per capability
//! with the consuming builders (`.load_bundle(...)`, `.text_encode(...)`,
//! ...). Semantics (ordered consumption, mid-script errors, exhaustion)
//! are unchanged.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use crate::cancellation::NodeCancellation;
use crate::{
    Backend, CreateEmptyLatentRequest, CreateEmptyLatentResponse, DiffusionSampleRequest,
    DiffusionSampleResponse, ImageImportRequest, ImageImportResponse, ImagePreviewRequest,
    ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse, InferenceBackend,
    InferenceBackendCapabilities, InferenceCapability, InferenceCapabilitySupport, InferenceError,
    InferenceInvocation, LatentDecodeRequest, LatentDecodeResponse, LatentEncodeRequest,
    LatentEncodeResponse, LoadBundleRequest, LoadBundleResponse, TextEncodeRequest,
    TextEncodeResponse,
};

/// How a [`ScriptedBackend`] behaves when a capability's script is
/// exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptExhaustion {
    /// Return a deterministic
    /// [`InferenceError::BackendExecutionFailed`] naming the capability.
    /// This is the default.
    Error,
    /// Panic with a descriptive message so over-consumption is loud in
    /// tests that want strict consumption assertions.
    Panic,
}

/// A queue of canned results for one capability, consumed in order.
struct Script<Resp> {
    steps: VecDeque<Result<Resp, InferenceError>>,
}

impl<Resp> Script<Resp> {
    fn new(steps: Vec<Result<Resp, InferenceError>>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
        }
    }

    fn remaining(&self) -> usize {
        self.steps.len()
    }

    fn pop(&mut self) -> Option<Result<Resp, InferenceError>> {
        self.steps.pop_front()
    }
}

#[derive(Default)]
struct ScriptedCapabilities {
    load_bundle: Option<Script<LoadBundleResponse>>,
    text_encode: Option<Script<TextEncodeResponse>>,
    create_empty_latent: Option<Script<CreateEmptyLatentResponse>>,
    diffusion_sample: Option<Script<DiffusionSampleResponse>>,
    latent_decode: Option<Script<LatentDecodeResponse>>,
    latent_encode: Option<Script<LatentEncodeResponse>>,
    image_import: Option<Script<ImageImportResponse>>,
    image_save: Option<Script<ImageSaveResponse>>,
    image_preview: Option<Script<ImagePreviewResponse>>,
}

impl ScriptedCapabilities {
    /// Capabilities whose script still has at least one step left.
    fn supported(&self) -> Vec<InferenceCapability> {
        let mut caps = Vec::new();
        if self.load_bundle.as_ref().is_some_and(|s| s.remaining() > 0) {
            caps.push(InferenceCapability::LoadBundle);
        }
        if self.text_encode.as_ref().is_some_and(|s| s.remaining() > 0) {
            caps.push(InferenceCapability::TextEncode);
        }
        if self
            .create_empty_latent
            .as_ref()
            .is_some_and(|s| s.remaining() > 0)
        {
            caps.push(InferenceCapability::CreateEmptyLatent);
        }
        if self
            .diffusion_sample
            .as_ref()
            .is_some_and(|s| s.remaining() > 0)
        {
            caps.push(InferenceCapability::DiffusionSample);
        }
        if self
            .latent_decode
            .as_ref()
            .is_some_and(|s| s.remaining() > 0)
        {
            caps.push(InferenceCapability::LatentDecode);
        }
        if self
            .latent_encode
            .as_ref()
            .is_some_and(|s| s.remaining() > 0)
        {
            caps.push(InferenceCapability::LatentEncode);
        }
        if self
            .image_import
            .as_ref()
            .is_some_and(|s| s.remaining() > 0)
        {
            caps.push(InferenceCapability::ImageImport);
        }
        if self.image_save.as_ref().is_some_and(|s| s.remaining() > 0) {
            caps.push(InferenceCapability::ImageSave);
        }
        if self
            .image_preview
            .as_ref()
            .is_some_and(|s| s.remaining() > 0)
        {
            caps.push(InferenceCapability::ImagePreview);
        }
        caps
    }
}

/// Shared state behind a [`ScriptedHold`] gate.
struct HoldState {
    /// Set when a scripted step reaches the hold.
    entered: AtomicBool,
    /// Set when the test releases the hold.
    released: AtomicBool,
}

/// Handle for pausing a [`ScriptedBackend`] step mid-flight.
///
/// The gate is created with [`ScriptedBackend::with_hold`]. The next
/// scripted step (of any capability) signals `entered` and then blocks
/// until the test calls [`release`](Self::release) **or** the invocation
/// is cancelled — cancellation wins, and the backend returns
/// [`InferenceError::Cancelled`]. This is the primitive that lets tests
/// interleave a cancel with an in-flight backend operation.
///
/// Waiting is poll-based (1ms) rather than channel-based so the backend
/// works with plain `std::sync` primitives; tests assert on eventual state
/// only, so the polls are CI-stable.
#[derive(Clone)]
pub struct ScriptedHold {
    state: Arc<HoldState>,
}

impl ScriptedHold {
    fn new() -> Self {
        Self {
            state: Arc::new(HoldState {
                entered: AtomicBool::new(false),
                released: AtomicBool::new(false),
            }),
        }
    }

    /// Returns once the held scripted step has entered the gate.
    pub async fn wait_entered(&self) {
        while !self.state.entered.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Release the held step; it proceeds (or returns
    /// [`InferenceError::Cancelled`] if the invocation was cancelled
    /// while held).
    pub fn release(&self) {
        self.state.released.store(true, Ordering::SeqCst);
    }

    /// `true` once a scripted step has entered the gate.
    pub fn is_entered(&self) -> bool {
        self.state.entered.load(Ordering::SeqCst)
    }

    /// `true` once the gate has been released.
    pub fn is_released(&self) -> bool {
        self.state.released.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for ScriptedHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedHold")
            .field("entered", &self.is_entered())
            .field("released", &self.is_released())
            .finish()
    }
}

/// A backend that replays a per-capability script of
/// `Result<Response, InferenceError>` steps in order.
///
/// # Example
///
/// ```
/// use reimagine_core::model::{TensorDType, TensorShape};
/// use reimagine_inference::{
///     Backend, BackendPayloadKey, BackendTensorHandle, ConditioningMetadata,
///     ExecutionConditioning, InferenceBackend, InferenceError, ScriptedBackend,
///     ScriptExhaustion, TextEncodeResponse,
/// };
///
/// // Script one success then an OutOfMemory.
/// let handle = BackendTensorHandle::new(
///     Backend::new("fake"),
///     BackendPayloadKey::new("embedding"),
///     TensorDType::F32,
///     TensorShape::new(vec![1, 4, 8, 8]),
///     "cpu",
/// );
/// let backend = ScriptedBackend::new("scripted")
///     .text_encode(vec![
///         Ok(TextEncodeResponse::new(ExecutionConditioning::new(
///             handle,
///             ConditioningMetadata::new(64, 64),
///         ))),
///         Err(InferenceError::OutOfMemory {
///             requested: Some(8_000_000),
///             available: None,
///         }),
///     ])
///     .on_exhaustion(ScriptExhaustion::Error);
///
/// let caps = backend.capabilities();
/// assert!(caps.supports_capability(reimagine_inference::operation::InferenceCapability::TextEncode));
/// assert_eq!(backend.total_calls(), 0);
/// ```
pub struct ScriptedBackend {
    kind: Backend,
    scripts: Mutex<ScriptedCapabilities>,
    /// Optional per-step pause applied before each scripted result.
    step_delay: Option<Duration>,
    /// Optional gate that pauses a step until released or cancelled.
    hold: Option<ScriptedHold>,
    /// Behavior when a script is exhausted.
    exhaustion: ScriptExhaustion,
    /// Cancellation captured from the most recent `admit_invocation`.
    latest_cancellation: Mutex<Option<Arc<dyn NodeCancellation>>>,
    /// Total capability calls observed (admitted steps), for assertions.
    total_calls: AtomicU64,
}

impl ScriptedBackend {
    /// Create a scripted backend with the given backend kind label.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: Backend::new(kind),
            scripts: Mutex::new(ScriptedCapabilities::default()),
            step_delay: None,
            hold: None,
            exhaustion: ScriptExhaustion::Error,
            latest_cancellation: Mutex::new(None),
            total_calls: AtomicU64::new(0),
        }
    }

    /// Pause every scripted step for `delay` before it returns. The pause
    /// is cancellation-aware: a cancelled invocation aborts the wait and
    /// the step returns [`InferenceError::Cancelled`].
    #[must_use]
    pub fn with_step_delay(mut self, delay: Duration) -> Self {
        self.step_delay = Some(delay);
        self
    }

    /// Configure the behavior when a script is exhausted (default:
    /// [`ScriptExhaustion::Error`]).
    #[must_use]
    pub fn on_exhaustion(mut self, behavior: ScriptExhaustion) -> Self {
        self.exhaustion = behavior;
        self
    }

    /// Install a hold gate. Returns the backend together with the gate
    /// handle; the next scripted step blocks inside the gate until the
    /// handle is released or the invocation is cancelled.
    #[must_use]
    pub fn with_hold(mut self) -> (Self, ScriptedHold) {
        let hold = ScriptedHold::new();
        self.hold = Some(hold.clone());
        (self, hold)
    }

    /// Total number of capability calls the backend has served (every
    /// call that passed the delay/hold gate, including exhausted ones).
    #[must_use]
    pub fn total_calls(&self) -> u64 {
        self.total_calls.load(Ordering::SeqCst)
    }

    /// Script the `load.bundle` capability.
    #[must_use]
    pub fn load_bundle(self, steps: Vec<Result<LoadBundleResponse, InferenceError>>) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .load_bundle = Some(Script::new(steps));
        self
    }

    /// Script the `text.encode` capability.
    #[must_use]
    pub fn text_encode(self, steps: Vec<Result<TextEncodeResponse, InferenceError>>) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .text_encode = Some(Script::new(steps));
        self
    }

    /// Script the `latent.create_empty` capability.
    #[must_use]
    pub fn create_empty_latent(
        self,
        steps: Vec<Result<CreateEmptyLatentResponse, InferenceError>>,
    ) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .create_empty_latent = Some(Script::new(steps));
        self
    }

    /// Script the `diffusion.sample` capability.
    #[must_use]
    pub fn diffusion_sample(
        self,
        steps: Vec<Result<DiffusionSampleResponse, InferenceError>>,
    ) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .diffusion_sample = Some(Script::new(steps));
        self
    }

    /// Script the `latent.decode` capability.
    #[must_use]
    pub fn latent_decode(self, steps: Vec<Result<LatentDecodeResponse, InferenceError>>) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .latent_decode = Some(Script::new(steps));
        self
    }

    /// Script the `latent.encode` capability.
    #[must_use]
    pub fn latent_encode(self, steps: Vec<Result<LatentEncodeResponse, InferenceError>>) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .latent_encode = Some(Script::new(steps));
        self
    }

    /// Script the `image.import` capability.
    #[must_use]
    pub fn image_import(self, steps: Vec<Result<ImageImportResponse, InferenceError>>) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .image_import = Some(Script::new(steps));
        self
    }

    /// Script the `image.save` capability.
    #[must_use]
    pub fn image_save(self, steps: Vec<Result<ImageSaveResponse, InferenceError>>) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .image_save = Some(Script::new(steps));
        self
    }

    /// Script the `image.preview` capability.
    #[must_use]
    pub fn image_preview(self, steps: Vec<Result<ImagePreviewResponse, InferenceError>>) -> Self {
        self.scripts
            .lock()
            .expect("scripted backend poisoned")
            .image_preview = Some(Script::new(steps));
        self
    }

    /// Wait for the hold gate (if armed) and the per-step delay (if
    /// configured), aborting early with [`InferenceError::Cancelled`]
    /// when the invocation cancellation trips.
    async fn gate(&self) -> Result<(), InferenceError> {
        let cancellation = self
            .latest_cancellation
            .lock()
            .expect("scripted backend poisoned")
            .clone();
        if let Some(hold) = &self.hold {
            hold.state.entered.store(true, Ordering::SeqCst);
            while !hold.state.released.load(Ordering::SeqCst) {
                if let Some(c) = &cancellation
                    && c.is_cancelled()
                {
                    return Err(InferenceError::Cancelled);
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
        if let Some(delay) = self.step_delay {
            match &cancellation {
                Some(c) => tokio::select! {
                    _ = tokio::time::sleep(delay) => {}
                    _ = c.cancelled() => return Err(InferenceError::Cancelled),
                },
                None => tokio::time::sleep(delay).await,
            }
        }
        if let Some(c) = &cancellation
            && c.is_cancelled()
        {
            return Err(InferenceError::Cancelled);
        }
        Ok(())
    }
}

fn exhausted_message(capability: InferenceCapability) -> String {
    format!("scripted backend exhausted: no more scripted steps for capability `{capability}`")
}

fn not_scripted_message(capability: InferenceCapability) -> String {
    format!(
        "scripted backend has no script for capability `{capability}`; register one with the builder methods"
    )
}

/// Pop and return the next step of `script`, or produce the configured
/// exhaustion/not-scripted outcome.
fn pop_step<Resp>(
    capability: InferenceCapability,
    script: Option<&mut Script<Resp>>,
    exhaustion: ScriptExhaustion,
    total_calls: &AtomicU64,
    kind: &Backend,
) -> Result<Resp, InferenceError> {
    total_calls.fetch_add(1, Ordering::Relaxed);
    match script {
        Some(script) => match script.pop() {
            Some(result) => result,
            None => match exhaustion {
                ScriptExhaustion::Error => Err(InferenceError::BackendExecutionFailed {
                    message: exhausted_message(capability),
                }),
                ScriptExhaustion::Panic => panic!("{}", exhausted_message(capability)),
            },
        },
        None => Err(InferenceError::BackendNotImplemented {
            capability,
            backend_kind: kind.to_string(),
            message: Some(not_scripted_message(capability)),
        }),
    }
}

#[async_trait]
impl InferenceBackend for ScriptedBackend {
    fn backend_kind(&self) -> &Backend {
        &self.kind
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        let scripts = self.scripts.lock().expect("scripted backend poisoned");
        let mut caps = InferenceBackendCapabilities::new(self.kind.clone());
        for capability in scripts.supported() {
            caps = caps.with_support(InferenceCapabilitySupport::new(capability));
        }
        caps
    }

    fn admit_invocation(&self, invocation: &InferenceInvocation) -> Result<(), InferenceError> {
        *self
            .latest_cancellation
            .lock()
            .expect("scripted backend poisoned") = Some(invocation.cancellation().clone());
        Ok(())
    }

    async fn load_bundle(
        &self,
        _request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::LoadBundle,
            scripts.load_bundle.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }

    async fn text_encode(
        &self,
        _request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::TextEncode,
            scripts.text_encode.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }

    async fn create_empty_latent(
        &self,
        _request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::CreateEmptyLatent,
            scripts.create_empty_latent.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }

    async fn diffusion_sample(
        &self,
        _request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::DiffusionSample,
            scripts.diffusion_sample.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }

    async fn latent_decode(
        &self,
        _request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::LatentDecode,
            scripts.latent_decode.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }

    async fn latent_encode(
        &self,
        _request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::LatentEncode,
            scripts.latent_encode.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }

    async fn image_import(
        &self,
        _request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::ImageImport,
            scripts.image_import.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }

    async fn image_save(
        &self,
        _request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::ImageSave,
            scripts.image_save.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }

    async fn image_preview(
        &self,
        _request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        self.gate().await?;
        let mut scripts = self.scripts.lock().expect("scripted backend poisoned");
        pop_step(
            InferenceCapability::ImagePreview,
            scripts.image_preview.as_mut(),
            self.exhaustion,
            &self.total_calls,
            &self.kind,
        )
    }
}

impl std::fmt::Debug for ScriptedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedBackend")
            .field("kind", &self.kind)
            .field("step_delay", &self.step_delay)
            .field("exhaustion", &self.exhaustion)
            .field("total_calls", &self.total_calls())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::NoopNodeCancellation;
    use crate::{
        BackendPayloadKey, BackendTensorHandle, ConditioningMetadata, ExecutionConditioning,
        ExecutionValue, RuntimeClipHandle,
    };
    use reimagine_core::model::{
        ModelId, NodeId, ParamValue, RunId, TensorDType, TensorShape, WorkflowId, WorkflowVersion,
    };

    fn conditioning(label: &str) -> ExecutionConditioning {
        ExecutionConditioning::new(
            BackendTensorHandle::new(
                Backend::new("fake"),
                BackendPayloadKey::new(label),
                TensorDType::F32,
                TensorShape::new(vec![1, 4, 8, 8]),
                "cpu",
            ),
            ConditioningMetadata::new(64, 64),
        )
    }

    fn text_request() -> TextEncodeRequest {
        TextEncodeRequest::new(
            RuntimeClipHandle::new(
                ModelId::new("sdxl-base-1.0"),
                Backend::new("fake"),
                "clip-1",
            ),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                "a prompt".to_owned(),
            ))),
            RunId::new("run-1"),
            WorkflowId::new("wf-1"),
            WorkflowVersion::new(1),
            NodeId::new("node-1"),
        )
    }

    fn run_blocking<F>(future: F) -> F::Output
    where
        F: std::future::Future,
    {
        tokio::runtime::Runtime::new().unwrap().block_on(future)
    }

    #[test]
    fn scripted_steps_are_consumed_in_order() {
        let backend = ScriptedBackend::new("fake").text_encode(vec![
            Ok(TextEncodeResponse::new(conditioning("one"))),
            Err(InferenceError::OutOfMemory {
                requested: Some(1024),
                available: None,
            }),
            Ok(TextEncodeResponse::new(conditioning("three"))),
        ]);

        let first =
            run_blocking(InferenceBackend::text_encode(&backend, text_request())).expect("step 1");
        assert_eq!(
            first.conditioning().text_embedding().payload_key().as_str(),
            "one"
        );

        let second = run_blocking(InferenceBackend::text_encode(&backend, text_request()))
            .expect_err("step 2 must fail with the scripted variant");
        match second {
            InferenceError::OutOfMemory {
                requested: Some(1024),
                ..
            } => {}
            other => panic!("expected OutOfMemory, got {other:?}"),
        }

        let third =
            run_blocking(InferenceBackend::text_encode(&backend, text_request())).expect("step 3");
        assert_eq!(
            third.conditioning().text_embedding().payload_key().as_str(),
            "three"
        );

        assert_eq!(backend.total_calls(), 3);
    }

    #[test]
    fn exhausted_script_returns_deterministic_error() {
        let backend = ScriptedBackend::new("fake")
            .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("one")))]);
        run_blocking(InferenceBackend::text_encode(&backend, text_request()))
            .expect("first step succeeds");
        let error = run_blocking(InferenceBackend::text_encode(&backend, text_request()))
            .expect_err("second call must report exhaustion");
        match error {
            InferenceError::BackendExecutionFailed { message } => {
                assert!(message.contains("exhausted"), "{message}");
                assert!(message.contains("text.encode"), "{message}");
            }
            other => panic!("expected BackendExecutionFailed, got {other:?}"),
        }
    }

    #[test]
    fn exhaustion_can_be_configured_to_panic() {
        let backend = ScriptedBackend::new("fake")
            .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("one")))])
            .on_exhaustion(ScriptExhaustion::Panic);
        run_blocking(InferenceBackend::text_encode(&backend, text_request()))
            .expect("first step succeeds");
        let panic = std::panic::catch_unwind(|| {
            run_blocking(InferenceBackend::text_encode(&backend, text_request()))
        });
        assert!(panic.is_err(), "exhausted script must panic in Panic mode");
    }

    #[test]
    fn capabilities_reflect_remaining_script_steps() {
        let backend = ScriptedBackend::new("fake")
            .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("one")))]);
        let caps = backend.capabilities();
        assert!(caps.supports_capability(InferenceCapability::TextEncode));
        assert!(!caps.supports_capability(InferenceCapability::LoadBundle));

        run_blocking(InferenceBackend::text_encode(&backend, text_request())).unwrap();
        let caps = backend.capabilities();
        assert!(
            !caps.supports_capability(InferenceCapability::TextEncode),
            "consumed scripts must stop being advertised"
        );
    }

    #[test]
    fn unscripted_capability_returns_backend_not_implemented() {
        let backend = ScriptedBackend::new("fake");
        let error = run_blocking(InferenceBackend::text_encode(&backend, text_request()))
            .expect_err("unscripted capability must fail");
        match error {
            InferenceError::BackendNotImplemented { capability, .. } => {
                assert_eq!(capability, InferenceCapability::TextEncode);
            }
            other => panic!("expected BackendNotImplemented, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn hold_gates_a_step_until_released() {
        let (backend, hold) = ScriptedBackend::new("fake")
            .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("held")))])
            .with_hold();
        let cancellation: Arc<dyn NodeCancellation> = Arc::new(NoopNodeCancellation::new());
        let invocation = InferenceInvocation::new(
            RunId::new("run-1"),
            NodeId::new("node-1"),
            None,
            cancellation,
            Arc::new(crate::NoopInferenceProgressSink),
        );

        let task = tokio::spawn(async move {
            backend
                .text_encode_with_invocation(&invocation, text_request())
                .await
        });
        hold.wait_entered().await;
        assert!(hold.is_entered());
        assert!(!hold.is_released());
        hold.release();
        let response = task.await.expect("step must finish after release");
        assert!(response.is_ok());
        assert!(hold.is_released());
    }

    #[tokio::test]
    async fn cancelling_while_held_returns_cancelled() {
        let (backend, hold) = ScriptedBackend::new("fake")
            .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("held")))])
            .with_hold();
        let cancellation = Arc::new(NoopNodeCancellation::new());
        let invocation = InferenceInvocation::new(
            RunId::new("run-1"),
            NodeId::new("node-1"),
            None,
            cancellation.clone(),
            Arc::new(crate::NoopInferenceProgressSink),
        );

        let task = tokio::spawn(async move {
            backend
                .text_encode_with_invocation(&invocation, text_request())
                .await
        });
        hold.wait_entered().await;
        cancellation.cancel();
        hold.release();
        let result = task.await.expect("held step must return after cancel");
        assert!(matches!(result, Err(InferenceError::Cancelled)));
    }

    #[tokio::test]
    async fn step_delay_is_cancellation_aware() {
        let backend = ScriptedBackend::new("fake")
            .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("slow")))])
            .with_step_delay(Duration::from_secs(60));
        let cancellation = Arc::new(NoopNodeCancellation::new());
        let invocation = InferenceInvocation::new(
            RunId::new("run-1"),
            NodeId::new("node-1"),
            None,
            cancellation.clone(),
            Arc::new(crate::NoopInferenceProgressSink),
        );

        let task = tokio::spawn(async move {
            backend
                .text_encode_with_invocation(&invocation, text_request())
                .await
        });
        tokio::task::yield_now().await;
        cancellation.cancel();
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("cancelled delayed step must not wait out the full delay")
            .expect("task must not panic");
        assert!(matches!(result, Err(InferenceError::Cancelled)));
    }
}
