//! Internal store of active runs, latest snapshots, and completed summaries.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::model::{ArtifactId, NodeId, RunId};
use tokio::sync::{mpsc, watch};

use crate::handle::RunHandle;
use crate::scheduler::NodeState;
use crate::snapshot::{RunArtifactRef, RunSnapshot, RunSnapshotUpdate, RunSummary};

/// Per-run snapshot broadcast channel.
///
/// Holds a keep-alive `Receiver` alongside the `Sender`: tokio's
/// `watch::Sender::send` fails when the channel has no receivers, which
/// would leave the stored value stale for unsubscribed runs.
#[derive(Debug)]
pub(crate) struct RunWatchChannel {
    sender: watch::Sender<Arc<RunSnapshot>>,
    _keep_alive: watch::Receiver<Arc<RunSnapshot>>,
}

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
    /// Latest snapshot broadcast channel per run id. Every store mutation
    /// publishes the full snapshot here; hosts subscribe instead of polling.
    pub(crate) watches: HashMap<RunId, RunWatchChannel>,
    /// Delta-stream fan-out per run id. Each `delta_stream` subscription
    /// registers its own unbounded mpsc sender; the store forwards
    /// [`RunSnapshotUpdate::Delta`]s to every live sender.
    pub(crate) delta_senders: HashMap<RunId, Vec<mpsc::UnboundedSender<RunSnapshotUpdate>>>,
}

/// V1 store: simple `Arc<RwLock<RunStoreInner>>` lock model.
///
/// Hosts query through `RuntimeService::snapshot` / `summary`, or subscribe
/// to push updates via [`RunStore::subscribe`] (full snapshots over a watch
/// channel) and [`RunStore::delta_stream`] (incremental deltas over mpsc).
/// The runner task updates the inner state as the run progresses.
#[derive(Debug, Clone, Default)]
pub struct RunStore {
    inner: Arc<RwLock<RunStoreInner>>,
}

impl RunStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a snapshot.
    ///
    /// Publishes the full snapshot to the run's watch channel and fans out
    /// the incremental [`RunSnapshotUpdate::Delta`] (nodes/artifacts that
    /// changed since the previous snapshot) to subscribed delta streams.
    /// Both publishes happen while holding the store lock so that the delta
    /// stream sees the baseline `Full` (enqueued at subscribe time) strictly
    /// before any later delta for the same run.
    pub fn put_snapshot(&self, snapshot: RunSnapshot) {
        let mut guard = self.inner.write().expect("run store poisoned");
        let previous = guard
            .snapshots
            .insert(snapshot.run_id.clone(), snapshot.clone());
        let (changed_nodes, new_artifacts) = delta_between(previous.as_ref(), &snapshot);
        let sender = guard
            .watches
            .entry(snapshot.run_id.clone())
            .or_insert_with(|| {
                let (sender, receiver) = watch::channel(Arc::new(snapshot.clone()));
                RunWatchChannel {
                    sender,
                    _keep_alive: receiver,
                }
            })
            .sender
            .clone();
        let _ = sender.send(Arc::new(snapshot.clone()));
        if changed_nodes.is_empty() && new_artifacts.is_empty() {
            return;
        }
        let update = RunSnapshotUpdate::Delta {
            run_id: snapshot.run_id.clone(),
            changed_nodes,
            new_artifacts,
            timestamp: snapshot.updated_at.clone(),
        };
        if let Some(senders) = guard.delta_senders.get_mut(&snapshot.run_id) {
            senders.retain(|s| s.send(update.clone()).is_ok());
        }
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
    ///
    /// Re-publishes the updated snapshot to the run's watch channel so watch
    /// subscribers observe the appended diagnostics. Deltas do not carry
    /// diagnostics, so no delta is emitted here.
    pub fn append_diagnostics(&self, run_id: &RunId, diagnostics: &[Diagnostic]) {
        if diagnostics.is_empty() {
            return;
        }

        let mut guard = self.inner.write().expect("run store poisoned");
        if let Some(snapshot) = guard.snapshots.get_mut(run_id) {
            snapshot.diagnostics.extend(diagnostics.iter().cloned());
            let latest = Arc::new(snapshot.clone());
            if let Some(channel) = guard.watches.get(run_id) {
                let _ = channel.sender.send(latest);
            }
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

    /// Subscribe to full snapshots for a run.
    ///
    /// Returns `None` if the run has no snapshot in the store yet (subscribe
    /// after the run is registered — the runtime publishes an initial
    /// snapshot when a run starts). The receiver immediately observes the
    /// current snapshot via [`watch::Receiver::borrow`] and is notified on
    /// every subsequent store mutation, with no per-read cloning.
    pub fn subscribe(&self, run_id: &RunId) -> Option<watch::Receiver<Arc<RunSnapshot>>> {
        let guard = self.inner.read().expect("run store poisoned");
        guard
            .watches
            .get(run_id)
            .map(|channel| channel.sender.subscribe())
    }

    /// Subscribe to incremental snapshot updates for a run.
    ///
    /// The stream begins with a [`RunSnapshotUpdate::Full`] baseline (the
    /// current snapshot, for late joiners) followed by
    /// [`RunSnapshotUpdate::Delta`]s for every subsequent change. Returns
    /// `None` if the run has no snapshot in the store yet.
    ///
    /// Deliberately `mpsc::UnboundedSender` under the hood: deltas must never
    /// apply backpressure to the run loop, matching the roadmap design
    /// (B2-9: "deltas via mpsc, full via watch"). The store retains a sender
    /// per subscription; dropped receivers are pruned on the next publish.
    pub fn delta_stream(
        &self,
        run_id: &RunId,
    ) -> Option<mpsc::UnboundedReceiver<RunSnapshotUpdate>> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let mut guard = self.inner.write().expect("run store poisoned");
        let baseline = guard.snapshots.get(run_id)?.clone();
        // Enqueue the baseline before registering the sender while holding
        // the store lock, so no delta produced by a later put_snapshot can
        // overtake the baseline.
        let _ = sender.send(RunSnapshotUpdate::Full(baseline));
        guard
            .delta_senders
            .entry(run_id.clone())
            .or_default()
            .push(sender);
        Some(receiver)
    }

    /// Read a snapshot by run id.
    ///
    /// Reads the current value from the run's watch channel (the latest
    /// published snapshot) with a fallback to the stored map for runs that
    /// predate the watch. Signature is unchanged from the polling era for
    /// backward compatibility.
    pub fn snapshot(&self, run_id: &RunId) -> Option<RunSnapshot> {
        let guard = self.inner.read().expect("run store poisoned");
        match guard.watches.get(run_id) {
            Some(channel) => Some((*channel.sender.borrow()).as_ref().clone()),
            None => guard.snapshots.get(run_id).cloned(),
        }
    }

    /// Read a summary by run id.
    pub fn summary(&self, run_id: &RunId) -> Option<RunSummary> {
        let guard = self.inner.read().expect("run store poisoned");
        guard.summaries.get(run_id).cloned()
    }

    /// Search all active snapshots and terminal summaries for an artifact by id.
    // TODO: Replace linear scan with a HashMap<ArtifactId, RunArtifactRef> index
    // once artifact volume grows beyond a handful per session.
    pub fn find_artifact(&self, artifact_id: &ArtifactId) -> Option<RunArtifactRef> {
        let guard = self.inner.read().expect("run store poisoned");

        // Search active snapshots
        for snapshot in guard.snapshots.values() {
            for artifact in &snapshot.artifacts {
                if artifact.id == *artifact_id {
                    return Some(artifact.clone());
                }
            }
        }

        // Search terminal summaries
        for summary in guard.summaries.values() {
            for artifact in &summary.artifacts {
                if artifact.id == *artifact_id {
                    return Some(artifact.clone());
                }
            }
        }

        None
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

/// Compute the incremental difference between the previous snapshot (if any)
/// and the current one: node states that changed or appeared, and artifacts
/// that appeared since the previous snapshot.
fn delta_between(
    previous: Option<&RunSnapshot>,
    current: &RunSnapshot,
) -> (HashMap<NodeId, NodeState>, Vec<RunArtifactRef>) {
    let mut changed_nodes = HashMap::new();
    for (node_id, state) in &current.node_states {
        if previous.and_then(|p| p.node_states.get(node_id)) != Some(state) {
            changed_nodes.insert(node_id.clone(), *state);
        }
    }

    let previous_artifact_ids: HashSet<&ArtifactId> = previous
        .map(|p| p.artifacts.iter().map(|a| &a.id).collect())
        .unwrap_or_default();
    let new_artifacts = current
        .artifacts
        .iter()
        .filter(|a| !previous_artifact_ids.contains(&a.id))
        .cloned()
        .collect();

    (changed_nodes, new_artifacts)
}
