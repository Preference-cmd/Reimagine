//! Backend-instance runtime hook contract.
//!
//! This module lives in `reimagine-inference` because concrete
//! backends implement [`BackendInstanceRuntimeHooks`] and must do so
//! without depending on `reimagine-runtime`. The runtime composes a
//! concrete implementation (e.g.
//! `candle::CandleBackendInstanceRuntimeHooks`) at startup and calls
//! into it during run lifecycle. The hook surface is lifecycle and
//! observation only, not a resource manager.
//!
//! The hook idiom is deliberately split into two traits
//! ([`BackendRunLifecycle`] and [`BackendInstanceObservation`])
//! unified under [`BackendInstanceRuntimeHooks`] so that consumers can
//! depend on the narrowest contract they need.

use std::collections::BTreeMap;
use std::sync::Arc;

use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::{DiagnosticId, RunId};
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

/// Snapshot of one backend instance, returned for diagnostics.
#[derive(Debug, Clone)]
pub struct BackendInstanceSnapshot {
    pub backend_instance: BackendInstance,
    pub backend: Backend,
    pub plugin: Option<Plugin>,
    pub extension: Option<Extension>,
    pub device: Option<DeviceProfile>,
    /// Backend-specific observations, e.g. `bytes_by_device`, `cached_models`.
    pub observations: BTreeMap<String, String>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Observation contract for backend-instance introspection.
#[async_trait::async_trait]
pub trait BackendInstanceObservation: Send + Sync + 'static {
    /// The concrete backend instance this observation source
    /// describes.
    fn backend_instance(&self) -> &BackendInstance;

    /// Produce a backend-instance snapshot for diagnostics.
    async fn snapshot(&self) -> BackendInstanceSnapshot;

    /// Produce one or more backend-instance snapshots.
    ///
    /// The default implementation returns a single snapshot from
    /// [`Self::snapshot`], which is correct for concrete backend
    /// instances. Composite implementations override this to fan out
    /// one snapshot per contained concrete instance.
    async fn snapshots(&self) -> Vec<BackendInstanceSnapshot> {
        vec![self.snapshot().await]
    }
}

/// Supertrait combining run lifecycle and backend-instance observation.
///
/// Anything that implements both [`BackendRunLifecycle`] and
/// [`BackendInstanceObservation`] automatically satisfies
/// [`BackendInstanceRuntimeHooks`].
pub trait BackendInstanceRuntimeHooks: BackendRunLifecycle + BackendInstanceObservation {}

impl<T: BackendRunLifecycle + BackendInstanceObservation> BackendInstanceRuntimeHooks for T {}

/// Composite hooks that let runtime consume one hook object while app-host
/// composes many backend instances.
pub struct CompositeBackendInstanceRuntimeHooks {
    backend_instance: BackendInstance,
    backend: Backend,
    hooks: Vec<Arc<dyn BackendInstanceRuntimeHooks>>,
}

impl std::fmt::Debug for CompositeBackendInstanceRuntimeHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeBackendInstanceRuntimeHooks")
            .field("backend_instance", &self.backend_instance)
            .field("backend", &self.backend)
            .field("hook_count", &self.hooks.len())
            .finish()
    }
}

impl CompositeBackendInstanceRuntimeHooks {
    pub fn new(hooks: Vec<Arc<dyn BackendInstanceRuntimeHooks>>) -> Self {
        Self {
            backend_instance: BackendInstance::new("composite"),
            backend: Backend::new("composite"),
            hooks,
        }
    }

    pub fn hooks(&self) -> &[Arc<dyn BackendInstanceRuntimeHooks>] {
        &self.hooks
    }
}

#[async_trait::async_trait]
impl BackendRunLifecycle for CompositeBackendInstanceRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
    }

    async fn begin_run(
        &self,
        request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError> {
        let mut diagnostics = Vec::new();
        for hook in &self.hooks {
            match hook.begin_run(request.clone()).await {
                Ok(report) => diagnostics.extend(report.diagnostics),
                Err(error) => diagnostics.push(hook_error_diagnostic(
                    "begin_run",
                    BackendRunLifecycle::backend_instance(hook.as_ref()),
                    error,
                )),
            }
        }
        Ok(BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics,
        })
    }

    async fn cleanup_run(
        &self,
        request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError> {
        let mut diagnostics = Vec::new();
        for hook in &self.hooks {
            match hook.cleanup_run(request.clone()).await {
                Ok(report) => diagnostics.extend(report.diagnostics),
                Err(error) => diagnostics.push(hook_error_diagnostic(
                    "cleanup_run",
                    BackendRunLifecycle::backend_instance(hook.as_ref()),
                    error,
                )),
            }
        }
        Ok(BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics,
        })
    }
}

#[async_trait::async_trait]
impl BackendInstanceObservation for CompositeBackendInstanceRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        // Aggregate the per-instance snapshots into a coarse composite view.
        let snapshots = self.snapshots().await;
        let mut observations = BTreeMap::new();
        observations.insert("backend_instances".to_owned(), snapshots.len().to_string());
        let diagnostics = snapshots
            .into_iter()
            .flat_map(|snapshot| snapshot.diagnostics)
            .collect();
        BackendInstanceSnapshot {
            backend_instance: self.backend_instance.clone(),
            backend: self.backend.clone(),
            plugin: None,
            extension: None,
            device: None,
            observations,
            diagnostics,
        }
    }

    async fn snapshots(&self) -> Vec<BackendInstanceSnapshot> {
        let mut snapshots = Vec::new();
        for hook in &self.hooks {
            // Recursively fan out so a composite of composites still returns
            // one snapshot per concrete backend instance.
            snapshots.extend(hook.snapshots().await);
        }
        snapshots
    }
}

fn hook_error_diagnostic(
    operation: &str,
    backend_instance: &BackendInstance,
    error: InferenceError,
) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!(
            "inference:backend_instance_hooks:{operation}:{backend_instance}"
        )),
        DiagnosticCode::new("INFERENCE/BACKEND_INSTANCE_HOOK_FAILED"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("inference"),
        format!("backend instance `{backend_instance}` {operation} hook failed: {error}"),
        DiagnosticTarget::new(DiagnosticTargetDomain::new("inference.backend_instance"))
            .with_id(backend_instance.to_string()),
    )
}

/// Historical alias. New code should use [`BackendInstanceSnapshot`].
pub type BackendResourceSnapshot = BackendInstanceSnapshot;

/// Historical alias. New code should use [`BackendInstanceObservation`].
pub trait BackendResourceObservation: BackendInstanceObservation {}

impl<T: BackendInstanceObservation> BackendResourceObservation for T {}

/// Historical alias. New code should use [`BackendInstanceRuntimeHooks`].
pub trait BackendResourceMechanism: BackendInstanceRuntimeHooks {}

impl<T: BackendInstanceRuntimeHooks> BackendResourceMechanism for T {}
