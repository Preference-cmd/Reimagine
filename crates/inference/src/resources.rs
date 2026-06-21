//! Backend lifecycle capability the runtime calls to communicate resource
//! intent without owning the underlying memory strategy.
//!
//! This module lives in `reimagine-inference` because concrete
//! backends implement [`RunResourceBackend`] and must do so without
//! depending on `reimagine-runtime`. The runtime composes a concrete
//! implementation (e.g. `candle::CandleRunResourceBackend`) at startup
//! and calls into it during run lifecycle; the backend decides what
//! to load, offload, evict, or move between devices.

use std::collections::HashMap;

use reimagine_core::model::RunId;

/// Snapshot of backend memory / cache state, returned for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Backend-specific observations, e.g. `bytes_by_device`, `cached_models`.
    pub observations: HashMap<String, String>,
}

/// Resource lifecycle hooks invoked by the runtime.
///
/// Runtime never decides what gets loaded, offloaded, evicted, or moved
/// between devices. It only signals intent (`begin_run`, `cleanup_run`,
/// `memory_snapshot`) and lets the backend implementation decide what
/// to do. The earlier per-value release hook was removed because
/// ordinary value lifecycle is driven by `Arc<ExecutionValue>` ownership
/// in the runtime and producer-declared retention; a per-value release
/// callback could be confused with a backend resource mechanism contract,
/// so it is intentionally not part of the trait anymore.
#[async_trait::async_trait]
pub trait RunResourceBackend: Send + Sync + 'static {
    /// Called once when a run starts. The backend may pin loaded models or
    /// create run-scoped allocation state.
    async fn begin_run(&self, run_id: &RunId);

    /// Called once after the run finishes (success/failure/cancel). The
    /// backend may release run-pinned resources. Cached models remain under
    /// backend policy.
    async fn cleanup_run(&self, run_id: &RunId);

    /// Backend-specific memory observations for diagnostics.
    async fn memory_snapshot(&self) -> MemorySnapshot;
}
