//! Backend lifecycle capability the runtime calls to communicate resource
//! intent without owning the underlying memory strategy.

use std::collections::HashMap;
use std::sync::Arc;

use reimagine_core::model::RunId;

use crate::value::ExecutionValue;

/// Snapshot of backend memory / cache state, returned for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Backend-specific observations, e.g. `bytes_by_device`, `cached_models`.
    pub observations: HashMap<String, String>,
}

/// Resource lifecycle hooks invoked by the runtime.
///
/// Runtime never decides what gets loaded, offloaded, evicted, or moved
/// between devices. It only signals intent (`begin_run`,
/// `release_runtime_value`, `cleanup_run`, `memory_snapshot`) and lets the
/// backend implementation decide what to do.
#[async_trait::async_trait]
pub trait RunResourceBackend: Send + Sync + 'static {
    /// Called once when a run starts. The backend may pin loaded models or
    /// create run-scoped allocation state.
    async fn begin_run(&self, run_id: &RunId);

    /// Called when the runtime drops a value (V1: at run cleanup). The
    /// backend may decrement a refcount, free a tensor, or keep it pooled.
    async fn release_runtime_value(&self, run_id: &RunId, value: Arc<ExecutionValue>);

    /// Called once after the run finishes (success/failure/cancel). The
    /// backend may release run-pinned resources. Cached models remain under
    /// backend policy.
    async fn cleanup_run(&self, run_id: &RunId);

    /// Backend-specific memory observations for diagnostics.
    async fn memory_snapshot(&self) -> MemorySnapshot;
}

/// Default no-op backend used in tests and when no backend is wired.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopRunResourceBackend;

#[async_trait::async_trait]
impl RunResourceBackend for NoopRunResourceBackend {
    async fn begin_run(&self, _run_id: &RunId) {}
    async fn release_runtime_value(&self, _run_id: &RunId, _value: Arc<ExecutionValue>) {}
    async fn cleanup_run(&self, _run_id: &RunId) {}
    async fn memory_snapshot(&self) -> MemorySnapshot {
        MemorySnapshot::default()
    }
}
