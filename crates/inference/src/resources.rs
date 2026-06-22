//! Backend-instance resource mechanism contract.
//!
//! This module lives in `reimagine-inference` because concrete
//! backends implement [`BackendResourceMechanism`] and must do so
//! without depending on `reimagine-runtime`. The runtime composes a
//! concrete implementation (e.g.
//! `candle::CandleResourceMechanism`) at startup and calls into it
//! during run lifecycle; the backend decides what to load, offload,
//! evict, or move between devices.
//!
//! The mechanism idiom is deliberately split into two traits
//! ([`BackendRunLifecycle`] and [`BackendResourceObservation`])
//! unified under [`BackendResourceMechanism`] so that consumers can
//! depend on the narrowest contract they need.
//!
//! ## V1 limitation
//!
//! Multi-instance composition (broadcasting lifecycle calls across
//! multiple `BackendResourceMechanism` instances and merging
//! snapshots) is **deferred**. The runtime currently receives a
//! single `Arc<dyn BackendResourceMechanism>` trait object. A
//! composite wrapper belongs in an inference helper or app-host
//! fixture, not inside runtime or concrete backend crates.
//!
//! TODO(inference/05-composite-resource-mechanism): add a composite
//! multi-instance resource mechanism that delegates to every registered
//! backend and merges per-instance snapshots.

use std::collections::BTreeMap;

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::model::RunId;
use reimagine_plugin::{Extension, Plugin};

use crate::backend_selection::{Backend, BackendInstance, DeviceProfile};
use crate::inference_error::InferenceError;

/// Request for the run-lifecycle calls.
#[derive(Debug, Clone)]
pub struct BackendRunLifecycleRequest {
    pub run_id: RunId,
}

/// Report returned after a lifecycle transition.
#[derive(Debug, Clone)]
pub struct BackendRunLifecycleReport {
    pub backend_instance: BackendInstance,
    pub diagnostics: Vec<Diagnostic>,
}

/// Run-scoped lifecycle hooks invoked by the runtime.
///
/// The runtime signals intent (`begin_run`, `cleanup_run`) and lets
/// the backend implementation decide what resources to pin or
/// release. Lifecycle calls are now fallible; callers should collect
/// diagnostics from the report.
#[async_trait::async_trait]
pub trait BackendRunLifecycle: Send + Sync + 'static {
    /// The concrete backend instance this lifecycle governs.
    fn backend_instance(&self) -> &BackendInstance;

    /// Called once when a run starts. The backend may pin loaded
    /// models or create run-scoped allocation state.
    async fn begin_run(
        &self,
        request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError>;

    /// Called once after the run finishes (success/failure/cancel).
    /// The backend may release run-pinned resources. Cached models
    /// remain under backend policy.
    async fn cleanup_run(
        &self,
        request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError>;
}

/// Snapshot of backend resource state, returned for diagnostics.
#[derive(Debug, Clone)]
pub struct BackendResourceSnapshot {
    pub backend_instance: BackendInstance,
    pub backend: Backend,
    pub plugin: Option<Plugin>,
    pub extension: Option<Extension>,
    pub device: Option<DeviceProfile>,
    /// Backend-specific observations, e.g. `bytes_by_device`, `cached_models`.
    pub observations: BTreeMap<String, String>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Observation contract for backend resource introspection.
#[async_trait::async_trait]
pub trait BackendResourceObservation: Send + Sync + 'static {
    /// The concrete backend instance this observation source
    /// describes.
    fn backend_instance(&self) -> &BackendInstance;

    /// Produce a resource snapshot for diagnostics.
    async fn resource_snapshot(&self) -> BackendResourceSnapshot;
}

/// Supertrait combining run lifecycle and resource observation.
///
/// Anything that implements both [`BackendRunLifecycle`] and
/// [`BackendResourceObservation`] automatically satisfies
/// [`BackendResourceMechanism`].
pub trait BackendResourceMechanism: BackendRunLifecycle + BackendResourceObservation {}

impl<T: BackendRunLifecycle + BackendResourceObservation> BackendResourceMechanism for T {}
