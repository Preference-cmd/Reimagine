//! Public [`RuntimeService`] entry point and the background runner that
//! actually drives a run.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reimagine_core::diagnostic::{
    CorrelationId, Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName,
    DiagnosticTarget, DiagnosticTargetDomain,
};
use reimagine_core::event::{RunEvent, RunEventId, RunEventKind, Timestamp};
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_core::readiness::{ExecutionNode, ExecutionPlan};
use tokio::sync::Mutex;
use tokio::task::JoinSet;

use crate::artifacts::ArtifactStore;
use crate::cancellation::CancellationToken;
use crate::clock::{Clock, SystemClock};
use crate::consumer_index::PlanConsumerIndex;
use crate::error::RuntimeError;
use crate::events::RunEventSink;
use crate::handle::{RunHandle, RunState};
use crate::resources::NoopBackendInstanceRuntimeHooks;
use crate::run_inputs::RunInputs;
use crate::run_session::{NodeOutcome, RunSession};
use crate::scheduler::{NodeState, StageExecutionPolicy, StageNodeDecision};
use crate::snapshot::{RunArtifactRef, RunSnapshot, RunSummary};
use crate::stage_runner::{
    PreparedNodeBindings, StageExecutionContext, StageNodePrepareError, StageNodeResult,
    StageNodeWork, execute_stage_node, missing_upstream_value_message,
    missing_workflow_input_message,
};
use crate::store::RunStore;
use crate::value_store::OutputKey;

use reimagine_inference::{
    BackendInstanceObservation, BackendInstanceRuntimeHooks, BackendInstanceSnapshot,
    BackendRunLifecycleRequest, ExecutionValueRetention, NodeExecutorRegistry, NodeInputs,
    NodeParams,
};

/// Options passed to [`RuntimeService::run`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RuntimeOptions {
    /// Optional correlation id propagated to events and node contexts.
    pub correlation_id: Option<CorrelationId>,
    /// Maximum number of same-stage node invocations admitted concurrently.
    ///
    /// `None` preserves V1's sequential compatibility behavior. `Some(1)` is
    /// also sequential. Values greater than one enable bounded same-stage
    /// concurrency. `Some(0)` is rejected by [`RuntimeService::run`].
    pub max_stage_concurrency: Option<usize>,
}

/// Public errors returned from `RuntimeService` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeServiceError {
    UnknownRun { run_id: String },
    EmptyPlan { run_id: String },
    MissingExecutor { run_id: String, type_id: String },
    EventSink { message: String },
    InvalidStageConcurrency { value: usize },
}

impl std::fmt::Display for RuntimeServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRun { run_id } => write!(f, "unknown run id: {run_id}"),
            Self::EmptyPlan { run_id } => write!(f, "empty execution plan for run {run_id}"),
            Self::MissingExecutor { run_id, type_id } => write!(
                f,
                "no executor registered for node type {type_id} in run {run_id}"
            ),
            Self::EventSink { message } => write!(f, "run event sink failed: {message}"),
            Self::InvalidStageConcurrency { value } => {
                write!(
                    f,
                    "invalid max_stage_concurrency: {value}; expected at least 1"
                )
            }
        }
    }
}

impl std::error::Error for RuntimeServiceError {}

impl From<RuntimeError> for RuntimeServiceError {
    fn from(value: RuntimeError) -> Self {
        match value {
            RuntimeError::UnknownRun { run_id } => Self::UnknownRun { run_id },
            RuntimeError::EmptyPlan { run_id } => Self::EmptyPlan { run_id },
            RuntimeError::EmptyExecutionGraph { run_id } => Self::EmptyPlan { run_id },
            RuntimeError::MissingExecutor { run_id, type_id } => {
                Self::MissingExecutor { run_id, type_id }
            }
            RuntimeError::EventSink { message } => Self::EventSink { message },
        }
    }
}

pub struct RuntimeService {
    store: RunStore,
    registry: NodeExecutorRegistry,
    backend: Arc<dyn BackendInstanceRuntimeHooks>,
    sink: Arc<dyn RunEventSink>,
    clock: Arc<dyn Clock>,
    next_run_seq: Arc<AtomicU64>,
    next_event_seq: Arc<AtomicU64>,
}

impl std::fmt::Debug for RuntimeService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeService")
            .field("store", &self.store)
            .field("registry", &self.registry)
            .finish()
    }
}

impl RuntimeService {
    pub fn new(
        registry: NodeExecutorRegistry,
        backend: Arc<dyn BackendInstanceRuntimeHooks>,
        sink: Arc<dyn RunEventSink>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            store: RunStore::new(),
            registry,
            backend,
            sink,
            clock,
            next_run_seq: Arc::new(AtomicU64::new(0)),
            next_event_seq: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn with_defaults(registry: NodeExecutorRegistry, sink: Arc<dyn RunEventSink>) -> Self {
        Self::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink,
            Arc::new(SystemClock),
        )
    }

    pub fn store(&self) -> &RunStore {
        &self.store
    }

    pub fn registry(&self) -> &NodeExecutorRegistry {
        &self.registry
    }

    pub fn run(
        &self,
        plan: Arc<ExecutionPlan>,
        run_inputs: RunInputs,
        options: RuntimeOptions,
    ) -> Result<RunHandle, RuntimeServiceError> {
        if plan.nodes().is_empty() {
            return Err(RuntimeServiceError::EmptyPlan {
                run_id: String::new(),
            });
        }
        if options.max_stage_concurrency == Some(0) {
            return Err(RuntimeServiceError::InvalidStageConcurrency { value: 0 });
        }
        for node in plan.nodes() {
            if self.registry.get(node.type_id()).is_none() {
                return Err(RuntimeServiceError::MissingExecutor {
                    run_id: String::new(),
                    type_id: node.type_id().to_string(),
                });
            }
        }

        let run_seq = self.next_run_seq.fetch_add(1, Ordering::Relaxed);
        let run_id = RunId::new(format!("run-{run_seq}"));
        let cancellation = CancellationToken::new();
        let handle = RunHandle::new(
            run_id.clone(),
            plan.workflow_id().clone(),
            plan.workflow_version(),
            cancellation.clone(),
        );

        let started_at = self.clock.now();
        let initial_snapshot = RunSnapshot::new(
            run_id.clone(),
            plan.workflow_id().clone(),
            plan.workflow_version(),
            RunState::Queued,
            Default::default(),
            Vec::new(),
            Vec::new(),
            started_at.clone(),
            started_at,
        );
        self.store.put_snapshot(initial_snapshot);
        self.store.register_active(handle.clone());

        self.emit_lifecycle(
            &run_id,
            plan.workflow_id(),
            plan.workflow_version(),
            RunEventKind::RunQueued,
            None,
            &[],
            None,
        );
        self.emit_lifecycle(
            &run_id,
            plan.workflow_id(),
            plan.workflow_version(),
            RunEventKind::RunStarted,
            None,
            &[],
            options.correlation_id.clone(),
        );

        let runner = Runner {
            run_id: run_id.clone(),
            plan,
            run_inputs,
            options,
            cancellation,
            store: self.store.clone(),
            registry: self.registry.clone_for_runner(),
            backend: self.backend.clone(),
            sink: self.sink.clone(),
            clock: self.clock.clone(),
            next_event_seq: self.next_event_seq.clone(),
        };
        tokio::spawn(runner.run());

        Ok(handle)
    }

    pub fn cancel(&self, run_id: &RunId) -> Result<(), RuntimeServiceError> {
        match self.store.active_cancellation(run_id) {
            Some(token) => {
                token.cancel();
                Ok(())
            }
            None => Err(RuntimeServiceError::UnknownRun {
                run_id: run_id.to_string(),
            }),
        }
    }

    pub fn snapshot(&self, run_id: &RunId) -> Option<RunSnapshot> {
        self.store.snapshot(run_id)
    }

    pub fn summary(&self, run_id: &RunId) -> Option<RunSummary> {
        self.store.summary(run_id)
    }

    pub async fn backend_instance_snapshots(&self) -> Vec<BackendInstanceSnapshot> {
        BackendInstanceObservation::snapshots(self.backend.as_ref()).await
    }

    fn emit_lifecycle(
        &self,
        run_id: &RunId,
        workflow_id: &WorkflowId,
        workflow_version: WorkflowVersion,
        kind: RunEventKind,
        node_id: Option<NodeId>,
        diagnostics: &[Diagnostic],
        correlation_id: Option<CorrelationId>,
    ) {
        let event_id_index = self.next_event_seq.fetch_add(1, Ordering::Relaxed);
        let mut event = RunEvent::new(
            RunEventId::new(format!("{run_id}-evt-{event_id_index}")),
            run_id.clone(),
            workflow_id.clone(),
            workflow_version,
            kind,
            self.clock.now(),
        );
        if let Some(nid) = node_id {
            event = event.with_node_id(nid);
        }
        for diag in diagnostics {
            event = event.with_diagnostic(diag.clone());
        }
        if let Some(cid) = correlation_id {
            event = event.with_correlation_id(cid);
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.sink.emit(event))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(
                    target: "reimagine_runtime",
                    error = %error,
                    "run event sink failed; continuing without failing the run"
                );
            }
            Err(_) => {
                tracing::warn!(
                    target: "reimagine_runtime",
                    "run event sink panicked; continuing without failing the run"
                );
            }
        }
    }
}

struct Runner {
    run_id: RunId,
    plan: Arc<ExecutionPlan>,
    run_inputs: RunInputs,
    options: RuntimeOptions,
    cancellation: CancellationToken,
    store: RunStore,
    registry: Arc<NodeExecutorRegistry>,
    backend: Arc<dyn BackendInstanceRuntimeHooks>,
    sink: Arc<dyn RunEventSink>,
    clock: Arc<dyn Clock>,
    next_event_seq: Arc<AtomicU64>,
}

impl Runner {
    async fn run(self) {
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
        let mut session = self
            .run_to_completion(session, artifact_store, &consumer_index)
            .await;
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

        for stage in self.plan.stages() {
            if self.cancellation.is_cancelled() {
                self.handle_cancellation(&mut session, &started_at, &artifact_store)
                    .await;
                return session;
            }
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

    async fn run_stage(
        &self,
        node_ids: &[NodeId],
        session: &mut RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
        policy: &mut StageExecutionPolicy,
    ) -> bool {
        let max_concurrency = self.options.max_stage_concurrency.unwrap_or(1).max(1);
        let mut joins = JoinSet::new();
        let mut next_index = 0usize;
        let failure_cancellation = CancellationToken::new();

        while next_index < node_ids.len() || !joins.is_empty() {
            while next_index < node_ids.len()
                && joins.len() < max_concurrency
                && policy.failed_message().is_none()
            {
                if self.cancellation.is_cancelled() {
                    failure_cancellation.cancel();
                    return true;
                }

                let node_id = &node_ids[next_index];
                next_index += 1;
                let node = match self.plan.nodes().iter().find(|n| n.node_id() == node_id) {
                    Some(node) => node.clone(),
                    None => continue,
                };

                match policy.decision_for(node_id) {
                    StageNodeDecision::Skip { reason } => {
                        self.reduce_node_skipped(
                            &node,
                            reason,
                            session,
                            started_at,
                            artifact_store,
                        )
                        .await;
                        continue;
                    }
                    StageNodeDecision::Execute => {}
                }

                let work = match self.prepare_stage_node_work(&node, session) {
                    Ok(work) => work,
                    Err(StageNodePrepareError::Failed(message)) => {
                        self.reduce_node_failed(
                            &node,
                            message,
                            session,
                            started_at,
                            artifact_store,
                            consumer_index,
                            policy,
                        )
                        .await;
                        failure_cancellation.cancel();
                        continue;
                    }
                };

                self.admit_stage_node(work.node(), session, started_at, artifact_store)
                    .await;
                let execution = StageExecutionContext {
                    run_id: self.run_id.clone(),
                    workflow_id: self.plan.workflow_id().clone(),
                    workflow_version: self.plan.workflow_version(),
                    correlation_id: self.options.correlation_id.clone(),
                    sink: self.sink.clone(),
                    clock: self.clock.clone(),
                    registry: self.registry.clone(),
                    cancellation: self.cancellation.clone(),
                };
                joins.spawn(execute_stage_node(
                    execution,
                    work,
                    artifact_store.clone(),
                    failure_cancellation.clone(),
                ));
            }

            if joins.is_empty() {
                break;
            }

            let result = match joins.join_next().await {
                Some(Ok(result)) => result,
                Some(Err(err)) => {
                    tracing::warn!(
                        target: "reimagine_runtime",
                        run_id = %self.run_id.as_str(),
                        error = %err,
                        "stage node task failed to join"
                    );
                    continue;
                }
                None => break,
            };

            let was_failing = policy.failed_message().is_some();
            let cancelled = self
                .reduce_stage_node_result(
                    result,
                    was_failing,
                    session,
                    started_at,
                    artifact_store,
                    consumer_index,
                    policy,
                )
                .await;

            if cancelled {
                failure_cancellation.cancel();
                return true;
            }

            if !was_failing && policy.failed_message().is_some() {
                failure_cancellation.cancel();
            }
        }

        if policy.failed_message().is_some() {
            while next_index < node_ids.len() {
                let node_id = &node_ids[next_index];
                next_index += 1;
                let node = match self.plan.nodes().iter().find(|n| n.node_id() == node_id) {
                    Some(node) => node.clone(),
                    None => continue,
                };
                if matches!(session.node_outcome(node_id), Some(outcome) if outcome.is_terminal()) {
                    continue;
                }
                let reason = match policy.decision_for(node_id) {
                    StageNodeDecision::Skip { reason } => reason,
                    StageNodeDecision::Execute => "run is already failing".to_owned(),
                };
                self.reduce_node_skipped(&node, reason, session, started_at, artifact_store)
                    .await;
            }
        }

        self.cancellation.is_cancelled() && policy.failed_message().is_none()
    }

    fn prepare_stage_node_work(
        &self,
        node: &ExecutionNode,
        session: &RunSession,
    ) -> Result<StageNodeWork, StageNodePrepareError> {
        let bindings = self.prepare_node_bindings(node, session)?;
        if self.registry.get(node.type_id()).is_none() {
            return Err(StageNodePrepareError::Failed(format!(
                "no executor for {}",
                node.type_id().as_str()
            )));
        }
        Ok(StageNodeWork::new(node.clone(), bindings))
    }

    fn prepare_node_bindings(
        &self,
        node: &ExecutionNode,
        session: &RunSession,
    ) -> Result<PreparedNodeBindings, StageNodePrepareError> {
        let mut inputs = NodeInputs::new();
        let mut params = NodeParams::new();
        for binding in node.input_bindings() {
            use reimagine_core::readiness::ExecutionInputSource;
            match binding.source() {
                ExecutionInputSource::Edge {
                    from_node_id,
                    from_slot_id,
                    ..
                } => {
                    let key = OutputKey::new(from_node_id.clone(), from_slot_id.clone());
                    match session.values().get(&key) {
                        Some(value) => {
                            inputs.insert(binding.slot_id().clone(), value);
                        }
                        None => {
                            return Err(StageNodePrepareError::Failed(
                                missing_upstream_value_message(
                                    from_node_id.as_str(),
                                    from_slot_id.as_str(),
                                ),
                            ));
                        }
                    }
                }
                ExecutionInputSource::WorkflowInput {
                    workflow_input_id, ..
                } => {
                    if let Some(value) = self.run_inputs.workflow_input(workflow_input_id) {
                        inputs.insert(binding.slot_id().clone(), value.clone());
                    } else {
                        return Err(StageNodePrepareError::Failed(
                            missing_workflow_input_message(
                                workflow_input_id.as_str(),
                                binding.slot_id().as_str(),
                            ),
                        ));
                    }
                }
                ExecutionInputSource::Param { .. } | ExecutionInputSource::Default { .. } => {
                    if let Some(value) = self
                        .run_inputs
                        .node_param(node.node_id(), binding.slot_id())
                    {
                        params.insert(binding.slot_id().clone(), value.clone());
                    }
                }
            }
        }
        Ok(PreparedNodeBindings::new(inputs, params))
    }

    async fn admit_stage_node(
        &self,
        node: &ExecutionNode,
        session: &mut RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        session.record_outcome(node.node_id().clone(), NodeOutcome::Queued);
        self.emit_node_event(node, RunEventKind::NodeQueued, &[]);
        self.publish_snapshot(session, started_at, artifact_store)
            .await;

        session.record_outcome(node.node_id().clone(), NodeOutcome::Running);
        self.emit_node_event(node, RunEventKind::NodeStarted, &[]);
        self.publish_snapshot(session, started_at, artifact_store)
            .await;
    }

    async fn reduce_stage_node_result(
        &self,
        result: StageNodeResult,
        discard_success: bool,
        session: &mut RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
        policy: &mut StageExecutionPolicy,
    ) -> bool {
        match result {
            StageNodeResult::Completed { node, outputs } => {
                let node_id = node.node_id().clone();
                if discard_success {
                    session.record_outcome(node_id, NodeOutcome::Cancelled);
                    self.drop_consumed_single_use_values(&node, consumer_index, session);
                    self.emit_node_event(&node, RunEventKind::NodeCancelled, &[]);
                    self.publish_snapshot(session, started_at, artifact_store)
                        .await;
                    return false;
                }

                session.record_outcome(node_id.clone(), NodeOutcome::Completed);
                for output in outputs {
                    let key = OutputKey::new(node_id.clone(), output.slot_id().clone());
                    let retention = output.retention();
                    if let Some(diag) =
                        self.check_single_use_fan_out(consumer_index, &key, retention)
                    {
                        let message = diag.message().to_string();
                        self.emit_node_event(
                            &node,
                            RunEventKind::NodeFailed,
                            std::slice::from_ref(&diag),
                        );
                        session.record_outcome(
                            node_id.clone(),
                            NodeOutcome::Failed {
                                message: message.clone(),
                            },
                        );
                        policy.record_failure(node_id, message);
                        self.publish_snapshot(session, started_at, artifact_store)
                            .await;
                        return false;
                    }
                    session
                        .values_mut()
                        .insert_with_retention(key, output.into_value(), retention);
                }
                self.emit_node_event(&node, RunEventKind::NodeCompleted, &[]);
                self.drop_consumed_single_use_values(&node, consumer_index, session);
                self.publish_snapshot(session, started_at, artifact_store)
                    .await;
                false
            }
            StageNodeResult::Failed { node, message } => {
                self.reduce_node_failed(
                    &node,
                    message,
                    session,
                    started_at,
                    artifact_store,
                    consumer_index,
                    policy,
                )
                .await;
                false
            }
            StageNodeResult::Cancelled { node } => {
                let already_failing = policy.failed_message().is_some();
                session.record_outcome(node.node_id().clone(), NodeOutcome::Cancelled);
                self.drop_consumed_single_use_values(&node, consumer_index, session);
                self.emit_node_event(&node, RunEventKind::NodeCancelled, &[]);
                self.publish_snapshot(session, started_at, artifact_store)
                    .await;
                !already_failing
            }
        }
    }

    async fn reduce_node_failed(
        &self,
        node: &ExecutionNode,
        message: String,
        session: &mut RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
        policy: &mut StageExecutionPolicy,
    ) {
        let diagnostic = make_diagnostic(&self.run_id, node.node_id(), &message);
        session.record_outcome(
            node.node_id().clone(),
            NodeOutcome::Failed {
                message: message.clone(),
            },
        );
        self.emit_node_event(
            node,
            RunEventKind::NodeFailed,
            std::slice::from_ref(&diagnostic),
        );
        policy.record_failure(node.node_id().clone(), message);
        self.drop_consumed_single_use_values(node, consumer_index, session);
        self.publish_snapshot(session, started_at, artifact_store)
            .await;
    }

    async fn reduce_node_skipped(
        &self,
        node: &ExecutionNode,
        reason: String,
        session: &mut RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        self.emit_node_skipped(node.node_id(), &node.type_id().clone(), &reason);
        session.record_outcome(
            node.node_id().clone(),
            NodeOutcome::Skipped {
                reason: reason.clone(),
            },
        );
        self.publish_snapshot(session, started_at, artifact_store)
            .await;
    }

    fn check_single_use_fan_out(
        &self,
        consumer_index: &PlanConsumerIndex,
        key: &OutputKey,
        retention: ExecutionValueRetention,
    ) -> Option<Diagnostic> {
        if retention != ExecutionValueRetention::SingleUse {
            return None;
        }
        let fan_out = consumer_index.fan_out(key);
        if fan_out > 1 {
            let node_id = key.node_id().clone();
            let slot_id = key.slot_id().clone();
            let message = format!(
                "SingleUse output {node_id}:{slot_id} has {fan_out} edge-sourced consumers in the active execution plan; SingleUse fan-out must be exactly one"
            );
            Some(make_diagnostic(&self.run_id, &node_id, &message))
        } else {
            None
        }
    }

    fn drop_consumed_single_use_values(
        &self,
        node: &ExecutionNode,
        consumer_index: &PlanConsumerIndex,
        session: &mut RunSession,
    ) {
        let upstream_keys: Vec<OutputKey> = node
            .input_bindings()
            .iter()
            .filter_map(|binding| match binding.source() {
                reimagine_core::readiness::ExecutionInputSource::Edge {
                    from_node_id,
                    from_slot_id,
                    ..
                } => Some(OutputKey::new(from_node_id.clone(), from_slot_id.clone())),
                _ => None,
            })
            .collect();
        let mut to_drop = Vec::new();
        for upstream in upstream_keys {
            let retention = match session.values().retention(&upstream) {
                Some(retention) => retention,
                None => continue,
            };
            if retention != ExecutionValueRetention::SingleUse {
                continue;
            }
            match consumer_index.unique_consumer(&upstream) {
                Some(unique) if unique.to_node_id == *node.node_id() => to_drop.push(upstream),
                _ => {}
            }
        }
        for key in to_drop {
            session.values_mut().remove(&key);
        }
    }

    async fn handle_cancellation(
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
        let visited: std::collections::HashSet<NodeId> =
            session.node_outcomes().keys().cloned().collect();
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

    async fn publish_snapshot(
        &self,
        session: &RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        self.publish_snapshot_with_state(session, RunState::Running, started_at, artifact_store)
            .await;
    }

    async fn publish_snapshot_with_state(
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
        node_outcomes: &std::collections::HashMap<NodeId, NodeOutcome>,
        state: RunState,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        let node_states: std::collections::HashMap<NodeId, NodeState> = node_outcomes
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
            self.plan.workflow_version(),
            state,
            node_states,
            diagnostics,
            artifacts,
            started_at.clone(),
            self.clock.now(),
        );
        self.store.put_snapshot(snapshot);
    }

    async fn publish_summary(
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
            self.plan.workflow_version(),
            state,
            diagnostics.to_vec(),
            artifacts,
            started_at,
            finished_at,
        );
        self.store.put_summary(summary);
    }

    fn emit_node_event(
        &self,
        node: &ExecutionNode,
        kind: RunEventKind,
        diagnostics: &[Diagnostic],
    ) {
        let event = self.build_event(kind, Some(node.node_id().clone()), diagnostics);
        self.safe_emit(event);
    }

    fn emit_lifecycle_event(
        &self,
        kind: RunEventKind,
        node_id: Option<NodeId>,
        diagnostics: &[Diagnostic],
    ) {
        let event = self.build_event(kind, node_id, diagnostics);
        self.safe_emit(event);
    }

    fn emit_node_skipped(
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
            self.plan.workflow_version(),
            kind,
            self.clock.now(),
        );
        if let Some(nid) = node_id {
            event = event.with_node_id(nid);
        }
        for diag in diagnostics {
            event = event.with_diagnostic(diag.clone());
        }
        if let Some(cid) = &self.options.correlation_id {
            event = event.with_correlation_id(cid.clone());
        }
        event
    }
}

fn make_diagnostic(run_id: &RunId, node_id: &NodeId, message: &str) -> Diagnostic {
    let target = DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow.node"))
        .with_id(node_id.as_str())
        .with_path(run_id.as_str());
    Diagnostic::new(
        reimagine_core::model::DiagnosticId::new(format!(
            "runtime-{}-{}",
            run_id.as_str(),
            node_id.as_str()
        )),
        DiagnosticCode::new("RUNTIME/NODE_EXECUTION_FAILED"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("runtime"),
        message,
        target,
    )
}

fn make_run_diagnostic(run_id: &RunId, message: &str) -> Diagnostic {
    let target =
        DiagnosticTarget::new(DiagnosticTargetDomain::new("workflow.run")).with_id(run_id.as_str());
    Diagnostic::new(
        reimagine_core::model::DiagnosticId::new(format!("runtime-{}", run_id.as_str())),
        DiagnosticCode::new("RUNTIME/RUN_EXECUTION_FAILED"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("runtime"),
        message,
        target,
    )
}
