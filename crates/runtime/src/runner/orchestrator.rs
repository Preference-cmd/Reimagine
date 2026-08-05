use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::event::RunEventKind;
use reimagine_core::model::{NodeId, RunId, WorkflowVersion};
use reimagine_core::readiness::ExecutionPlan;
use reimagine_inference::{
    BackendInstanceRuntimeHooks, BackendRunLifecycleRequest, NodeExecutorRegistry,
    ResourceHintSink, ResourceHints,
};
use tokio::sync::Mutex;

use super::diagnostics::make_run_diagnostic;
use super::service::RuntimeOptions;
use crate::artifacts::ArtifactStore;
use crate::cancellation::CancellationToken;
use crate::clock::Clock;
use crate::consumer_index::PlanConsumerIndex;
use crate::events::RunEventSink;
use crate::handle::RunState;
use crate::run_inputs::RunInputs;
use crate::run_session::{NodeOutcome, RunSession};
use crate::scheduler::{ReadySetScheduler, StageExecutionPolicy};
use crate::store::RunStore;
use crate::value_store::OutputKey;

pub(super) struct Runner {
    pub(super) run_id: RunId,
    pub(super) plan: Arc<ExecutionPlan>,
    pub(super) run_inputs: RunInputs,
    pub(super) options: RuntimeOptions,
    pub(super) cancellation: CancellationToken,
    pub(super) store: RunStore,
    pub(super) registry: Arc<NodeExecutorRegistry>,
    pub(super) backend: Arc<dyn BackendInstanceRuntimeHooks>,
    /// Optional backend resource-hint sink. When absent, hints are
    /// skipped (the default runtime configuration has no sink wired).
    pub(super) hint_sink: Option<Arc<dyn ResourceHintSink>>,
    pub(super) sink: Arc<dyn RunEventSink>,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) next_event_seq: Arc<AtomicU64>,
}

impl Runner {
    pub(super) async fn run(self) {
        let session = RunSession::new(
            self.run_id.clone(),
            self.plan.workflow_id().clone(),
            self.plan.workflow_version(),
            self.options.correlation_id.clone(),
            self.cancellation.clone(),
        );
        let artifact_store = Arc::new(Mutex::new(ArtifactStore::new()));
        let consumer_index = PlanConsumerIndex::from_plan(&self.plan);

        let request = BackendRunLifecycleRequest {
            run_id: self.run_id.clone(),
        };
        let mut lifecycle_diagnostics = Vec::new();
        match self.backend.begin_run(request.clone()).await {
            Ok(report) => lifecycle_diagnostics.extend(report.diagnostics),
            Err(err) => {
                tracing::warn!(%err, run_id = %self.run_id, "begin_run failed");
            }
        }
        let mut session = if self.options.use_ready_set {
            self.run_to_completion_ready_set(session, artifact_store, &consumer_index)
                .await
        } else {
            self.run_to_completion(session, artifact_store, &consumer_index)
                .await
        };
        session.values_mut().clear();
        match self.backend.cleanup_run(request).await {
            Ok(report) => lifecycle_diagnostics.extend(report.diagnostics),
            Err(err) => {
                tracing::warn!(%err, run_id = %self.run_id, "cleanup_run failed");
            }
        }
        if !lifecycle_diagnostics.is_empty() {
            self.store
                .append_diagnostics(&self.run_id, &lifecycle_diagnostics);
        }
    }

    async fn run_to_completion(
        &self,
        mut session: RunSession,
        artifact_store: Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
    ) -> RunSession {
        let started_at = self.clock.now();
        let mut policy = StageExecutionPolicy::new();
        let stages: Vec<_> = self.plan.stages().into_iter().cloned().collect();

        for (i, stage) in stages.iter().enumerate() {
            if self.cancellation.is_cancelled() {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }

            // Build and send resource hints for this stage
            let next_node_ids = stages.get(i + 1).map(|s| s.node_ids()).unwrap_or(&[]);
            let hints = self.build_resource_hints(stage.node_ids(), next_node_ids);
            self.send_resource_hints(&hints).await;

            if self
                .run_stage(
                    stage.node_ids(),
                    &mut session,
                    &started_at,
                    &artifact_store,
                    consumer_index,
                    &mut policy,
                )
                .await
            {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }
        }

        let finished_at = self.clock.now();
        let (state, lifecycle_kind, diagnostics) = if let Some(message) = policy.failed_message() {
            let diag = make_run_diagnostic(&self.run_id, message);
            (RunState::Failed, RunEventKind::RunFailed, vec![diag])
        } else {
            (RunState::Completed, RunEventKind::RunCompleted, Vec::new())
        };
        self.emit_lifecycle_event(lifecycle_kind, None, &diagnostics);
        self.publish_summary(
            &session,
            state,
            started_at.clone(),
            finished_at,
            &artifact_store,
            &diagnostics,
        )
        .await;
        self.publish_snapshot_with_state(&session, state, &started_at, &artifact_store)
            .await;
        self.store.finalize(&self.run_id);
        session
    }

    pub(super) fn workflow_version(&self) -> WorkflowVersion {
        self.plan.workflow_version()
    }

    pub(super) fn started_correlation_id(&self) -> Option<CorrelationId> {
        self.options.correlation_id.clone()
    }

    /// Feed run-stage results back into the ready-set scheduler so its
    /// dependency tracking reflects what actually executed.
    ///
    /// Without this, a multi-stage workflow executed through the ready-set
    /// path would never discover downstream nodes (nothing marks outputs as
    /// satisfied) and would silently "complete" without running them.
    fn mark_ready_set_progress(
        &self,
        ready_set: &mut ReadySetScheduler,
        session: &RunSession,
        batch: &[NodeId],
    ) {
        for node_id in batch {
            match session.node_outcome(node_id) {
                Some(NodeOutcome::Completed) => {
                    if let Some(node) = self.plan.nodes().iter().find(|n| n.node_id() == node_id) {
                        for slot in node.output_slots() {
                            ready_set.mark_output(OutputKey::new(node_id.clone(), slot.clone()));
                        }
                    }
                    ready_set.mark_completed(node_id);
                }
                Some(NodeOutcome::Failed { .. }) | Some(NodeOutcome::Cancelled) => {
                    ready_set.mark_failed(node_id);
                }
                _ => {}
            }
        }
    }

    /// Build [`ResourceHints`] for the given stage.
    ///
    /// V1 forwards the VRAM budget from [`RuntimeOptions`]. Prefetch and
    /// component lifecycle hints are left empty — populating them requires
    /// access to `NodeDef` resource requirements, which will be wired when
    /// the runtime gains a `NodeCatalog` reference.
    fn build_resource_hints(
        &self,
        _stage_node_ids: &[NodeId],
        _next_stage_node_ids: &[NodeId],
    ) -> ResourceHints {
        let mut hints = ResourceHints::new(self.run_id.clone());

        if let Some(budget) = self.options.vram_budget {
            hints = hints.with_vram_budget(budget);
        }

        hints
    }

    /// Send resource hints to the backend before a stage.
    ///
    /// Resource hints are advisory: when no sink is wired, the transport
    /// fails, or the worker does not support the operation, the stage
    /// still runs without them. Failures are logged, never propagated.
    async fn send_resource_hints(&self, hints: &ResourceHints) {
        let Some(sink) = &self.hint_sink else {
            tracing::debug!(
                run_id = %self.run_id,
                vram_budget = ?hints.vram_budget,
                "no resource hint sink wired; skipping hints for stage"
            );
            return;
        };
        match sink.apply_resource_hints(hints.clone()).await {
            Ok(()) => {
                tracing::debug!(
                    run_id = %self.run_id,
                    vram_budget = ?hints.vram_budget,
                    "resource hints applied for stage"
                );
            }
            Err(error) => {
                tracing::warn!(
                    run_id = %self.run_id,
                    %error,
                    vram_budget = ?hints.vram_budget,
                    "failed to apply resource hints; continuing stage without hints"
                );
            }
        }
    }

    /// Run to completion using the dynamic ready-set scheduler.
    ///
    /// Instead of iterating pre-computed stages, this method maintains
    /// a ready-set of nodes whose inputs are all satisfied and dispatches
    /// them immediately. This enables better GPU utilization for
    /// multi-branch workflows.
    async fn run_to_completion_ready_set(
        &self,
        mut session: RunSession,
        artifact_store: Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
    ) -> RunSession {
        let started_at = self.clock.now();
        let mut policy = StageExecutionPolicy::new();
        let mut ready_set = ReadySetScheduler::from_plan(self.plan.nodes(), self.plan.edges());

        // Kick off initial ready nodes
        let initial = ready_set.take_ready(usize::MAX);
        if !initial.is_empty() {
            if self.cancellation.is_cancelled() {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }

            // Send resource hints for initial batch
            let hints = self.build_resource_hints(&initial, &[]);
            self.send_resource_hints(&hints).await;

            if self
                .run_stage(
                    &initial,
                    &mut session,
                    &started_at,
                    &artifact_store,
                    consumer_index,
                    &mut policy,
                )
                .await
            {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }
            self.mark_ready_set_progress(&mut ready_set, &session, &initial);
        }

        // Continue dispatching as nodes complete and new ones become ready
        while !ready_set.is_complete() {
            if self.cancellation.is_cancelled() {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }

            // Find newly ready nodes from outputs produced in the last batch
            let newly_ready = ready_set.ready_nodes();
            if newly_ready.is_empty() {
                // No more nodes can become ready — either all done or deadlock
                break;
            }

            let batch = ready_set.take_ready(usize::MAX);

            // Peek at next ready nodes for lookahead
            let next_ready = ready_set.ready_nodes();
            let hints = self.build_resource_hints(&batch, &next_ready);
            self.send_resource_hints(&hints).await;

            if self
                .run_stage(
                    &batch,
                    &mut session,
                    &started_at,
                    &artifact_store,
                    consumer_index,
                    &mut policy,
                )
                .await
            {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }
            self.mark_ready_set_progress(&mut ready_set, &session, &batch);
        }

        let finished_at = self.clock.now();
        let (state, lifecycle_kind, diagnostics) = if let Some(message) = policy.failed_message() {
            let diag = make_run_diagnostic(&self.run_id, message);
            (RunState::Failed, RunEventKind::RunFailed, vec![diag])
        } else {
            (RunState::Completed, RunEventKind::RunCompleted, Vec::new())
        };
        self.emit_lifecycle_event(lifecycle_kind, None, &diagnostics);
        self.publish_summary(
            &session,
            state,
            started_at.clone(),
            finished_at,
            &artifact_store,
            &diagnostics,
        )
        .await;
        self.publish_snapshot_with_state(&session, state, &started_at, &artifact_store)
            .await;
        self.store.finalize(&self.run_id);
        session
    }
}
