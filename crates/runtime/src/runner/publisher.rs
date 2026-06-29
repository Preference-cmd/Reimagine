use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::{RunEvent, RunEventId, RunEventKind, Timestamp};
use reimagine_core::model::NodeId;
use reimagine_core::readiness::ExecutionNode;
use tokio::sync::Mutex;

use super::diagnostics::make_diagnostic;
use super::orchestrator::Runner;
use crate::artifacts::ArtifactStore;
use crate::handle::RunState;
use crate::run_session::{NodeOutcome, RunSession};
use crate::scheduler::NodeState;
use crate::snapshot::{RunArtifactRef, RunSnapshot, RunSummary};

impl Runner {
    pub(super) async fn handle_cancellation(
        &self,
        session: &mut RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        for (node_id, outcome) in session.node_outcomes_mut() {
            if matches!(outcome, NodeOutcome::Queued | NodeOutcome::Running) {
                *outcome = NodeOutcome::Cancelled;
                let event =
                    self.build_event(RunEventKind::NodeCancelled, Some(node_id.clone()), &[]);
                self.safe_emit(event);
            }
        }
        let visited: HashSet<NodeId> = session.node_outcomes().keys().cloned().collect();
        for plan_node in self.plan.nodes() {
            if !visited.contains(plan_node.node_id()) {
                session.record_outcome(plan_node.node_id().clone(), NodeOutcome::Cancelled);
                let event = self.build_event(
                    RunEventKind::NodeCancelled,
                    Some(plan_node.node_id().clone()),
                    &[],
                );
                self.safe_emit(event);
            }
        }
        self.emit_lifecycle_event(RunEventKind::RunCancelled, None, &[]);
        self.publish_summary(
            session,
            RunState::Cancelled,
            started_at.clone(),
            self.clock.now(),
            artifact_store,
            &[],
        )
        .await;
        self.publish_snapshot_with_state(session, RunState::Cancelled, started_at, artifact_store)
            .await;
        self.store.finalize(&self.run_id);
    }

    pub(super) async fn publish_snapshot(
        &self,
        session: &RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        self.publish_snapshot_with_state(session, RunState::Running, started_at, artifact_store)
            .await;
    }

    pub(super) async fn publish_snapshot_with_state(
        &self,
        session: &RunSession,
        state: RunState,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        self.publish_snapshot_for_outcomes(
            session.node_outcomes(),
            state,
            started_at,
            artifact_store,
        )
        .await;
    }

    async fn publish_snapshot_for_outcomes(
        &self,
        node_outcomes: &HashMap<NodeId, NodeOutcome>,
        state: RunState,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        let node_states: HashMap<NodeId, NodeState> = node_outcomes
            .iter()
            .map(|(node_id, outcome)| {
                let state = match outcome {
                    NodeOutcome::Queued => NodeState::Queued,
                    NodeOutcome::Running => NodeState::Running,
                    NodeOutcome::Completed => NodeState::Completed,
                    NodeOutcome::Failed { .. } => NodeState::Failed,
                    NodeOutcome::Skipped { .. } => NodeState::Skipped,
                    NodeOutcome::Cancelled => NodeState::Cancelled,
                };
                (node_id.clone(), state)
            })
            .collect();
        let artifacts: Vec<RunArtifactRef> = {
            let store = artifact_store.lock().await;
            store
                .iter_ordered()
                .map(|a| RunArtifactRef::new(a.id.clone(), a.node_id.clone(), a.reference.clone()))
                .collect()
        };
        let diagnostics: Vec<Diagnostic> = node_outcomes
            .iter()
            .filter_map(|(node_id, outcome)| match outcome {
                NodeOutcome::Failed { message } => {
                    Some(make_diagnostic(&self.run_id, node_id, message))
                }
                _ => None,
            })
            .collect();
        let snapshot = RunSnapshot::new(
            self.run_id.clone(),
            self.plan.workflow_id().clone(),
            self.workflow_version(),
            state,
            node_states,
            diagnostics,
            artifacts,
            started_at.clone(),
            self.clock.now(),
        );
        self.store.put_snapshot(snapshot);
    }

    pub(super) async fn publish_summary(
        &self,
        _session: &RunSession,
        state: RunState,
        started_at: Timestamp,
        finished_at: Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        diagnostics: &[Diagnostic],
    ) {
        let artifacts: Vec<RunArtifactRef> = {
            let store = artifact_store.lock().await;
            store
                .iter_ordered()
                .map(|a| RunArtifactRef::new(a.id.clone(), a.node_id.clone(), a.reference.clone()))
                .collect()
        };
        let summary = RunSummary::new(
            self.run_id.clone(),
            self.plan.workflow_id().clone(),
            self.workflow_version(),
            state,
            diagnostics.to_vec(),
            artifacts,
            started_at,
            finished_at,
        );
        self.store.put_summary(summary);
    }

    pub(super) fn emit_node_event(
        &self,
        node: &ExecutionNode,
        kind: RunEventKind,
        diagnostics: &[Diagnostic],
    ) {
        let event = self.build_event(kind, Some(node.node_id().clone()), diagnostics);
        self.safe_emit(event);
    }

    pub(super) fn emit_lifecycle_event(
        &self,
        kind: RunEventKind,
        node_id: Option<NodeId>,
        diagnostics: &[Diagnostic],
    ) {
        let event = self.build_event(kind, node_id, diagnostics);
        self.safe_emit(event);
    }

    pub(super) fn emit_node_skipped(
        &self,
        node_id: &NodeId,
        type_id: &reimagine_core::model::NodeTypeId,
        reason: &str,
    ) {
        let diagnostic = make_diagnostic(&self.run_id, node_id, reason);
        let placeholder_node =
            ExecutionNode::new(node_id.clone(), type_id.clone(), Vec::new(), Vec::new());
        self.emit_node_event(
            &placeholder_node,
            RunEventKind::NodeSkipped,
            std::slice::from_ref(&diagnostic),
        );
    }

    fn safe_emit(&self, event: RunEvent) {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.sink.emit(event))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(
                    target: "reimagine_runtime",
                    run_id = %self.run_id.as_str(),
                    error = %error,
                    "run event sink failed; continuing without failing the run"
                );
            }
            Err(_) => {
                tracing::warn!(
                    target: "reimagine_runtime",
                    run_id = %self.run_id.as_str(),
                    "run event sink panicked; continuing without failing the run"
                );
            }
        }
    }

    fn build_event(
        &self,
        kind: RunEventKind,
        node_id: Option<NodeId>,
        diagnostics: &[Diagnostic],
    ) -> RunEvent {
        let event_id_index = self.next_event_seq.fetch_add(1, Ordering::Relaxed);
        let mut event = RunEvent::new(
            RunEventId::new(format!("{}-evt-{event_id_index}", self.run_id.as_str())),
            self.run_id.clone(),
            self.plan.workflow_id().clone(),
            self.workflow_version(),
            kind,
            self.clock.now(),
        );
        if let Some(nid) = node_id {
            event = event.with_node_id(nid);
        }
        for diag in diagnostics {
            event = event.with_diagnostic(diag.clone());
        }
        if let Some(cid) = self.started_correlation_id() {
            event = event.with_correlation_id(cid);
        }
        event
    }
}
