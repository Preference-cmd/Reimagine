//! Internal store of active runs, latest snapshots, and completed summaries.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::model::RunId;

use crate::handle::RunHandle;
use crate::snapshot::{RunSnapshot, RunSummary};

/// Inner state behind [`RunStore`].
///
/// Fields are `pub(crate)` so only the runner task within this crate can
/// mutate them. External hosts query through the narrow [`RunStore`]
/// surface.
#[derive(Debug, Default)]
pub struct RunStoreInner {
    /// Currently active (non-terminal) runs keyed by `RunId`.
    pub(crate) active: HashMap<RunId, RunHandle>,
    /// Latest snapshot per run id.
    pub(crate) snapshots: HashMap<RunId, RunSnapshot>,
    /// Completed run summaries per run id.
    pub(crate) summaries: HashMap<RunId, RunSummary>,
}

/// V1 store: simple `Arc<RwLock<RunStoreInner>>` lock model.
///
/// Hosts query through `RuntimeService::snapshot` / `summary`. The runner
/// task updates the inner state as the run progresses.
#[derive(Debug, Clone, Default)]
pub struct RunStore {
    inner: Arc<RwLock<RunStoreInner>>,
}

impl RunStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a snapshot.
    pub(crate) fn put_snapshot(&self, snapshot: RunSnapshot) {
        let mut guard = self.inner.write().expect("run store poisoned");
        guard.snapshots.insert(snapshot.run_id.clone(), snapshot);
    }

    /// Move a handle from active to summary and drop it from active.
    pub(crate) fn finalize(&self, run_id: &RunId) {
        let mut guard = self.inner.write().expect("run store poisoned");
        guard.active.remove(run_id);
    }

    /// Insert a summary directly.
    pub(crate) fn put_summary(&self, summary: RunSummary) {
        let mut guard = self.inner.write().expect("run store poisoned");
        guard.summaries.insert(summary.run_id.clone(), summary);
    }

    /// Append diagnostics to the latest snapshot and terminal summary for
    /// a run. Used after backend lifecycle hooks finish, because cleanup
    /// diagnostics are only available after the runner has published its
    /// terminal state.
    pub(crate) fn append_diagnostics(&self, run_id: &RunId, diagnostics: &[Diagnostic]) {
        if diagnostics.is_empty() {
            return;
        }

        let mut guard = self.inner.write().expect("run store poisoned");
        if let Some(snapshot) = guard.snapshots.get_mut(run_id) {
            snapshot.diagnostics.extend(diagnostics.iter().cloned());
        }
        if let Some(summary) = guard.summaries.get_mut(run_id) {
            summary.diagnostics.extend(diagnostics.iter().cloned());
        }
    }

    /// Borrow the inner store mutably; crate-internal only.
    #[allow(dead_code)]
    pub(crate) fn inner_mut(&self) -> std::sync::RwLockWriteGuard<'_, RunStoreInner> {
        self.inner.write().expect("run store poisoned")
    }

    /// Read a snapshot by run id.
    pub fn snapshot(&self, run_id: &RunId) -> Option<RunSnapshot> {
        let guard = self.inner.read().expect("run store poisoned");
        guard.snapshots.get(run_id).cloned()
    }

    /// Read a summary by run id.
    pub fn summary(&self, run_id: &RunId) -> Option<RunSummary> {
        let guard = self.inner.read().expect("run store poisoned");
        guard.summaries.get(run_id).cloned()
    }

    /// Number of active runs.
    pub fn active_count(&self) -> usize {
        let guard = self.inner.read().expect("run store poisoned");
        guard.active.len()
    }

    /// Number of stored summaries.
    pub fn summary_count(&self) -> usize {
        let guard = self.inner.read().expect("run store poisoned");
        guard.summaries.len()
    }

    /// Read the cancellation token for an active run.
    pub(crate) fn active_cancellation(
        &self,
        run_id: &RunId,
    ) -> Option<crate::cancellation::CancellationToken> {
        let guard = self.inner.read().expect("run store poisoned");
        guard.active.get(run_id).map(|h| h.cancellation())
    }

    /// Register an active run with its handle.
    pub(crate) fn register_active(&self, handle: RunHandle) {
        let mut guard = self.inner.write().expect("run store poisoned");
        let run_id = handle.run_id().clone();
        guard.active.insert(run_id, handle);
    }
}
