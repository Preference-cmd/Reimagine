use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::event::{RunEventKind, Timestamp};
use reimagine_core::model::{NodeId, RunId, WorkflowVersion};
use reimagine_core::readiness::{ExecutionPlan, ExecutionStage};
use reimagine_inference::{
    BackendInstanceRuntimeHooks, BackendRunLifecycleRequest, NodeExecutorRegistry,
    ResourceHintSink, ResourceHints, StageId,
};
use tokio::sync::Mutex;
use tokio::task::JoinSet;

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

/// A run of consecutive [`ExecutionStage`]s with no transitive
/// cross-stage data dependencies (BE-35).
///
/// Member stages may execute concurrently; groups themselves always
/// execute in stage order, so value-store merges and
/// `StageScoped` retention drops stay deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ParallelStageGroup {
    stages: Vec<ExecutionStage>,
}

impl ParallelStageGroup {
    fn new(stages: Vec<ExecutionStage>) -> Self {
        Self { stages }
    }

    pub(super) fn stages(&self) -> &[ExecutionStage] {
        &self.stages
    }

    fn len(&self) -> usize {
        self.stages.len()
    }
}

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
        let runner = Arc::new(self);
        let session = Arc::new(Mutex::new(RunSession::new(
            runner.run_id.clone(),
            runner.plan.workflow_id().clone(),
            runner.plan.workflow_version(),
            runner.options.correlation_id.clone(),
            runner.cancellation.clone(),
        )));
        let artifact_store = Arc::new(Mutex::new(ArtifactStore::new()));
        let consumer_index = Arc::new(PlanConsumerIndex::from_plan(&runner.plan));

        let request = BackendRunLifecycleRequest {
            run_id: runner.run_id.clone(),
        };
        let mut lifecycle_diagnostics = Vec::new();
        match runner.backend.begin_run(request.clone()).await {
            Ok(report) => lifecycle_diagnostics.extend(report.diagnostics),
            Err(err) => {
                tracing::warn!(%err, run_id = %runner.run_id, "begin_run failed");
            }
        }
        let mut session = if runner.options.use_ready_set {
            runner
                .run_to_completion_ready_set(session, artifact_store, consumer_index)
                .await
        } else {
            runner
                .run_to_completion(session, artifact_store, consumer_index)
                .await
        };
        session.values_mut().clear();
        match runner.backend.cleanup_run(request).await {
            Ok(report) => lifecycle_diagnostics.extend(report.diagnostics),
            Err(err) => {
                tracing::warn!(%err, run_id = %runner.run_id, "cleanup_run failed");
            }
        }
        if !lifecycle_diagnostics.is_empty() {
            runner
                .store
                .append_diagnostics(&runner.run_id, &lifecycle_diagnostics);
        }
    }

    /// Unwrap the run session from its shared wrapper.
    ///
    /// Every parallel stage task has completed (its `Arc` clone dropped)
    /// when this is called, so the unwrap cannot fail in practice.
    fn into_session(session: Arc<Mutex<RunSession>>) -> RunSession {
        Arc::try_unwrap(session)
            .ok()
            .expect("run session still shared after all stage tasks completed")
            .into_inner()
    }

    async fn run_to_completion(
        self: &Arc<Self>,
        session: Arc<Mutex<RunSession>>,
        artifact_store: Arc<Mutex<ArtifactStore>>,
        consumer_index: Arc<PlanConsumerIndex>,
    ) -> RunSession {
        let started_at = self.clock.now();
        let policy = Arc::new(Mutex::new(StageExecutionPolicy::new()));
        let stages = self.plan.stages().to_vec();
        let groups = self.build_stage_groups(&stages);

        for group in groups {
            if self.cancellation.is_cancelled() {
                let mut session_guard = session.lock().await;
                self.handle_cancellation(&mut session_guard, &started_at, &artifact_store)
                    .await;
                drop(session_guard);
                return Self::into_session(session);
            }

            // Send resource hints for the group's stages, in stage order.
            for (offset, stage) in group.stages().iter().enumerate() {
                let next_node_ids = group
                    .stages()
                    .get(offset + 1)
                    .map(|s| s.node_ids())
                    .unwrap_or(&[]);
                let hints = self.build_resource_hints(stage.node_ids(), next_node_ids);
                self.send_resource_hints(&hints).await;
            }

            if self
                .run_stage_group(
                    &group,
                    &session,
                    &started_at,
                    &artifact_store,
                    &consumer_index,
                    &policy,
                )
                .await
            {
                let mut session_guard = session.lock().await;
                self.handle_cancellation(&mut session_guard, &started_at, &artifact_store)
                    .await;
                drop(session_guard);
                return Self::into_session(session);
            }
        }

        let finished_at = self.clock.now();
        let (state, lifecycle_kind, diagnostics) =
            if let Some(message) = policy.lock().await.failed_message() {
                let diag = make_run_diagnostic(&self.run_id, message);
                (RunState::Failed, RunEventKind::RunFailed, vec![diag])
            } else {
                (RunState::Completed, RunEventKind::RunCompleted, Vec::new())
            };
        self.emit_lifecycle_event(lifecycle_kind, None, &diagnostics);
        let session_guard = session.lock().await;
        self.publish_summary(
            &session_guard,
            state,
            started_at.clone(),
            finished_at,
            &artifact_store,
            &diagnostics,
        )
        .await;
        self.publish_snapshot_with_state(&session_guard, state, &started_at, &artifact_store)
            .await;
        drop(session_guard);
        self.store.finalize(&self.run_id);
        Self::into_session(session)
    }

    /// Partition the plan's stages into maximal groups of consecutive
    /// stages with no transitive cross-stage data dependencies (BE-35).
    ///
    /// Dependency information is derived from `ExecutionPlan::edges()`:
    /// stage `i` depends on stage `j < i` when any node of `i`
    /// transitively consumes any node of `j`. A stage joins the current
    /// group only when it depends on no earlier member, so every member
    /// of a group can safely start once the previous group completed.
    ///
    /// Wave-leveled plans (the planner's Kahn leveling) always produce
    /// singleton groups, preserving V1's strictly sequential stage
    /// execution; plans whose levels carry nodes at different depths
    /// group their independent tails.
    fn build_stage_groups(&self, stages: &[ExecutionStage]) -> Vec<ParallelStageGroup> {
        let mut stage_of: HashMap<&NodeId, usize> = HashMap::new();
        for (index, stage) in stages.iter().enumerate() {
            for node_id in stage.node_ids() {
                stage_of.insert(node_id, index);
            }
        }

        // Stage-level dependency edges: stage_deps[i] = stages i
        // directly consumes from.
        let mut stage_deps: Vec<HashSet<usize>> = vec![HashSet::new(); stages.len()];
        for edge in self.plan.edges() {
            let from_stage = stage_of.get(edge.from_node_id()).copied();
            let to_stage = stage_of.get(edge.to_node_id()).copied();
            if let (Some(from), Some(to)) = (from_stage, to_stage)
                && from != to
            {
                stage_deps[to].insert(from);
            }
        }

        // Transitive closure: stage_deps[i] = every stage i depends on,
        // directly or indirectly.
        for index in 0..stages.len() {
            let mut stack: Vec<usize> = stage_deps[index].iter().copied().collect();
            while let Some(dep) = stack.pop() {
                let ancestors: Vec<usize> = stage_deps[dep].iter().copied().collect();
                for ancestor in ancestors {
                    if stage_deps[index].insert(ancestor) {
                        stack.push(ancestor);
                    }
                }
            }
        }

        let mut groups = Vec::new();
        let mut start = 0usize;
        for index in 0..stages.len() {
            let independent = stage_deps[index].iter().all(|&dep| dep < start);
            if !independent {
                groups.push(ParallelStageGroup::new(stages[start..index].to_vec()));
                start = index;
            }
        }
        groups.push(ParallelStageGroup::new(stages[start..].to_vec()));
        groups
    }

    /// Execute the stages of a group, honoring
    /// `max_stage_group_concurrency` (BE-35).
    ///
    /// Singleton groups — and groups under a concurrency cap of 1 —
    /// execute each stage sequentially through [`Runner::run_stage`],
    /// identical to V1. Otherwise up to `max_stage_group_concurrency`
    /// member stages run concurrently on a [`JoinSet`], sharing the run
    /// session, fail-fast policy, consumer index, and memory budget.
    /// Returns `true` when any member stage observed run cancellation;
    /// callers run the usual cancellation handling once, after every
    /// member stage has joined.
    async fn run_stage_group(
        self: &Arc<Self>,
        group: &ParallelStageGroup,
        session: &Arc<Mutex<RunSession>>,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        consumer_index: &Arc<PlanConsumerIndex>,
        policy: &Arc<Mutex<StageExecutionPolicy>>,
    ) -> bool {
        let max_concurrency = self.options.max_stage_group_concurrency.unwrap_or(1).max(1);
        if group.len() <= 1 || max_concurrency <= 1 {
            let mut cancelled = false;
            for stage in group.stages() {
                if self
                    .run_stage(
                        stage.node_ids(),
                        session,
                        started_at,
                        artifact_store,
                        consumer_index,
                        policy,
                        Some(stage.index()),
                    )
                    .await
                {
                    cancelled = true;
                }
            }
            return cancelled;
        }

        let mut joins = JoinSet::new();
        let mut next = 0usize;
        let mut cancelled = false;
        while next < group.len() || !joins.is_empty() {
            while next < group.len() && joins.len() < max_concurrency {
                let stage = &group.stages()[next];
                next += 1;
                let runner = self.clone();
                let session = session.clone();
                let artifact_store = artifact_store.clone();
                let consumer_index = consumer_index.clone();
                let policy = policy.clone();
                let node_ids = stage.node_ids().to_vec();
                let stage_index = stage.index();
                let started_at = started_at.clone();
                joins.spawn(async move {
                    runner
                        .run_stage(
                            &node_ids,
                            &session,
                            &started_at,
                            &artifact_store,
                            &consumer_index,
                            &policy,
                            Some(stage_index),
                        )
                        .await
                });
            }

            let Some(result) = joins.join_next().await else {
                break;
            };
            match result {
                Ok(run_cancelled) => cancelled |= run_cancelled,
                Err(err) => {
                    tracing::warn!(
                        target: "reimagine_runtime",
                        run_id = %self.run_id.as_str(),
                        error = %err,
                        "stage group task failed to join"
                    );
                }
            }
        }
        cancelled
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
    ///
    /// The ready-set path deliberately keeps batches sequential: batches
    /// are waves of mutually-ready nodes, so stage-group parallelism
    /// (BE-35) cannot apply safely — there is no notion of an
    /// independent "stage" to group. `StageScoped` retention (BE-18) is
    /// honored at batch boundaries: a plan stage is considered complete
    /// once every node it contains has a terminal outcome.
    async fn run_to_completion_ready_set(
        self: &Arc<Self>,
        session: Arc<Mutex<RunSession>>,
        artifact_store: Arc<Mutex<ArtifactStore>>,
        consumer_index: Arc<PlanConsumerIndex>,
    ) -> RunSession {
        let started_at = self.clock.now();
        let policy = Arc::new(Mutex::new(StageExecutionPolicy::new()));
        let mut ready_set = ReadySetScheduler::from_plan(self.plan.nodes(), self.plan.edges());

        // Kick off initial ready nodes
        let initial = ready_set.take_ready(usize::MAX);
        if !initial.is_empty() {
            if self.cancellation.is_cancelled() {
                let mut session_guard = session.lock().await;
                self.handle_cancellation(&mut session_guard, &started_at, &artifact_store)
                    .await;
                drop(session_guard);
                return Self::into_session(session);
            }

            // Send resource hints for initial batch
            let hints = self.build_resource_hints(&initial, &[]);
            self.send_resource_hints(&hints).await;

            if self
                .run_stage(
                    &initial,
                    &session,
                    &started_at,
                    &artifact_store,
                    &consumer_index,
                    &policy,
                    None,
                )
                .await
            {
                let mut session_guard = session.lock().await;
                self.handle_cancellation(&mut session_guard, &started_at, &artifact_store)
                    .await;
                drop(session_guard);
                return Self::into_session(session);
            }
            {
                let session_guard = session.lock().await;
                self.mark_ready_set_progress(&mut ready_set, &session_guard, &initial);
            }
            self.drop_completed_stage_scoped(&session).await;
        }

        // Continue dispatching as nodes complete and new ones become ready
        while !ready_set.is_complete() {
            if self.cancellation.is_cancelled() {
                let mut session_guard = session.lock().await;
                self.handle_cancellation(&mut session_guard, &started_at, &artifact_store)
                    .await;
                drop(session_guard);
                return Self::into_session(session);
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
                    &session,
                    &started_at,
                    &artifact_store,
                    &consumer_index,
                    &policy,
                    None,
                )
                .await
            {
                let mut session_guard = session.lock().await;
                self.handle_cancellation(&mut session_guard, &started_at, &artifact_store)
                    .await;
                drop(session_guard);
                return Self::into_session(session);
            }
            {
                let session_guard = session.lock().await;
                self.mark_ready_set_progress(&mut ready_set, &session_guard, &batch);
            }
            self.drop_completed_stage_scoped(&session).await;
        }

        let finished_at = self.clock.now();
        let (state, lifecycle_kind, diagnostics) =
            if let Some(message) = policy.lock().await.failed_message() {
                let diag = make_run_diagnostic(&self.run_id, message);
                (RunState::Failed, RunEventKind::RunFailed, vec![diag])
            } else {
                (RunState::Completed, RunEventKind::RunCompleted, Vec::new())
            };
        self.emit_lifecycle_event(lifecycle_kind, None, &diagnostics);
        let session_guard = session.lock().await;
        self.publish_summary(
            &session_guard,
            state,
            started_at.clone(),
            finished_at,
            &artifact_store,
            &diagnostics,
        )
        .await;
        self.publish_snapshot_with_state(&session_guard, state, &started_at, &artifact_store)
            .await;
        drop(session_guard);
        self.store.finalize(&self.run_id);
        Self::into_session(session)
    }

    /// Drop `StageScoped` values whose plan stage has fully completed
    /// (BE-18).
    ///
    /// The ready-set path executes arbitrary ready batches rather than
    /// plan stages, so a stage is considered complete once every node
    /// in the plan stage has a terminal outcome. Idempotent: stages
    /// already dropped have nothing left to remove.
    async fn drop_completed_stage_scoped(&self, session: &Mutex<RunSession>) {
        let mut session_guard = session.lock().await;
        for stage in self.plan.stages() {
            let all_terminal = stage.node_ids().iter().all(|node_id| {
                session_guard
                    .node_outcome(node_id)
                    .is_some_and(NodeOutcome::is_terminal)
            });
            if all_terminal {
                session_guard
                    .values_mut()
                    .drop_stage_scoped(StageId::new(stage.index()));
            }
        }
    }
}
