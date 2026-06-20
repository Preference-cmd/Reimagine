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
use reimagine_core::readiness::ExecutionPlan;
use tokio::sync::Mutex;

use crate::artifacts::{ArtifactStore, RuntimeNodeArtifactCapability};
use crate::cancellation::CancellationToken;
use crate::clock::{Clock, SystemClock};
use crate::consumer_index::PlanConsumerIndex;
use crate::error::RuntimeError;
use crate::events::RunEventSink;
use crate::handle::{RunHandle, RunState};
use crate::resources::NoopRunResourceBackend;
use crate::run_inputs::RunInputs;
use crate::run_session::{NodeOutcome, RunSession};
use crate::scheduler::{NodeState, StageExecutionPolicy, StageNodeDecision};
use crate::snapshot::{RunArtifactRef, RunSnapshot, RunSummary};
use crate::store::RunStore;
use crate::value_store::OutputKey;

use reimagine_inference::{
    ArtifactPublisher, ExecutionValueRetention, NodeCancellation, NodeExecutionContext,
    NodeExecutorError, NodeExecutorRegistry, NodeInputs, NodeParams,
};
use reimagine_inference_core::RunResourceBackend;

/// Options passed to [`RuntimeService::run`].
///
/// Marked `#[non_exhaustive]` so future hosts (Tauri, Axum) can extend
/// the option set without breaking existing call sites. New fields can be
/// added with a default via `..RuntimeOptions::default()`.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RuntimeOptions {
    /// Optional correlation id propagated to events and node contexts.
    pub correlation_id: Option<CorrelationId>,
}

/// Public errors returned from `RuntimeService` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeServiceError {
    /// The run id is not known to the underlying store.
    UnknownRun { run_id: String },
    /// The plan is empty (no nodes).
    EmptyPlan { run_id: String },
    /// A plan node references a `NodeTypeId` with no registered executor.
    MissingExecutor { run_id: String, type_id: String },
    /// A host-provided event sink failed to emit an event.
    EventSink { message: String },
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

/// Long-lived, host-independent runtime service.
///
/// Held by `app-host` (or any other host) and shared across run requests.
/// V1 uses a simple `Arc<RwLock<RunStoreInner>>` lock model inside the
/// [`RunStore`].
pub struct RuntimeService {
    store: RunStore,
    registry: NodeExecutorRegistry,
    backend: Arc<dyn RunResourceBackend>,
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
    /// Construct a runtime service with a custom clock, resource backend,
    /// and event sink.
    pub fn new(
        registry: NodeExecutorRegistry,
        backend: Arc<dyn RunResourceBackend>,
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

    /// Convenience constructor with the system clock and a no-op resource
    /// backend.
    pub fn with_defaults(registry: NodeExecutorRegistry, sink: Arc<dyn RunEventSink>) -> Self {
        Self::new(
            registry,
            Arc::new(NoopRunResourceBackend),
            sink,
            Arc::new(SystemClock),
        )
    }

    /// Borrow the underlying store (for tests / diagnostics).
    pub fn store(&self) -> &RunStore {
        &self.store
    }

    /// Borrow the underlying executor registry.
    pub fn registry(&self) -> &NodeExecutorRegistry {
        &self.registry
    }

    /// Start a run for a prepared `ExecutionPlan`.
    ///
    /// Returns a [`RunHandle`] immediately. The actual run executes on a
    /// background `tokio` task and pushes events through the configured
    /// [`RunEventSink`]. Hosts observe progress through [`Self::snapshot`]
    /// and [`Self::summary`].
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

    /// Request cancellation of an active run.
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

    /// Read the latest snapshot for the given run id.
    pub fn snapshot(&self, run_id: &RunId) -> Option<RunSnapshot> {
        self.store.snapshot(run_id)
    }

    /// Read the terminal summary for the given run id (only present once the
    /// run has reached a terminal state).
    pub fn summary(&self, run_id: &RunId) -> Option<RunSummary> {
        self.store.summary(run_id)
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
        // Catch panics in the sink so a misbehaving implementation does
        // not abort the host that called `RuntimeService::run`.
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

/// Background runner task. Owns the [`RunSession`] and drives the
/// per-stage scheduler.
struct Runner {
    run_id: RunId,
    plan: Arc<ExecutionPlan>,
    run_inputs: RunInputs,
    options: RuntimeOptions,
    cancellation: CancellationToken,
    store: RunStore,
    registry: Arc<NodeExecutorRegistry>,
    backend: Arc<dyn RunResourceBackend>,
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

        self.backend.begin_run(&self.run_id).await;
        let mut session = self
            .run_to_completion(session, artifact_store, &consumer_index)
            .await;
        // Drop the runtime's run-scoped `Arc<ExecutionValue>` references.
        // Per-value release callbacks were removed from the
        // `RunResourceBackend` trait; backend-owned payloads remain
        // alive as long as the backend itself keeps a handle or
        // workspace cache entry.
        session.values_mut().clear();
        self.backend.cleanup_run(&self.run_id).await;
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

            for node_id in stage.node_ids() {
                if self.cancellation.is_cancelled() {
                    self.handle_cancellation(&mut session, &started_at, &artifact_store)
                        .await;
                    return session;
                }

                let node = match self.plan.nodes().iter().find(|n| n.node_id() == node_id) {
                    Some(node) => node.clone(),
                    None => continue,
                };

                match policy.decision_for(node_id) {
                    StageNodeDecision::Skip { reason } => {
                        self.emit_node_skipped(node_id, &node.type_id().clone(), &reason);
                        session.record_outcome(
                            node_id.clone(),
                            NodeOutcome::Skipped {
                                reason: reason.clone(),
                            },
                        );
                        self.publish_snapshot(&session, &started_at, &artifact_store)
                            .await;
                        continue;
                    }
                    StageNodeDecision::Execute => {}
                }

                session.record_outcome(node_id.clone(), NodeOutcome::Queued);
                self.emit_node_event(&node, RunEventKind::NodeQueued, &[]);
                self.publish_snapshot(&session, &started_at, &artifact_store)
                    .await;

                let result = self
                    .execute_node(&node, &session, artifact_store.clone())
                    .await;

                match result {
                    Ok(outputs) => {
                        session.record_outcome(node_id.clone(), NodeOutcome::Completed);
                        for output in outputs {
                            let key = OutputKey::new(node_id.clone(), output.slot_id().clone());
                            let retention = output.retention();
                            // Enforce producer-declared retention fan-out
                            // for the active plan. A `SingleUse` output
                            // with more than one edge-sourced consumer in
                            // the active plan must fail the run before any
                            // downstream consumer sees the value.
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
                                policy.record_failure(node_id.clone(), message);
                                self.publish_snapshot(&session, &started_at, &artifact_store)
                                    .await;
                                break;
                            }
                            session.values_mut().insert_with_retention(
                                key,
                                output.into_value(),
                                retention,
                            );
                        }
                        if matches!(
                            session.node_outcome(node_id),
                            Some(NodeOutcome::Failed { .. })
                        ) {
                            // Skip the NodeCompleted event; the failure
                            // path has already emitted it.
                            continue;
                        }
                        self.emit_node_event(&node, RunEventKind::NodeCompleted, &[]);
                    }
                    Err(NodeFailure::Failed(message)) => {
                        let diagnostic = make_diagnostic(&self.run_id, &node_id, &message);
                        session.record_outcome(
                            node_id.clone(),
                            NodeOutcome::Failed {
                                message: message.clone(),
                            },
                        );
                        self.emit_node_event(
                            &node,
                            RunEventKind::NodeFailed,
                            std::slice::from_ref(&diagnostic),
                        );
                        policy.record_failure(node_id.clone(), message);
                    }
                    Err(NodeFailure::Cancelled) => {
                        session.record_outcome(node_id.clone(), NodeOutcome::Cancelled);
                        // Issue 05 contract: drop upstream `SingleUse`
                        // values whose unique consumer was this node,
                        // even when the attempt was cancelled. The
                        // terminal `clear()` in `Runner::run` would
                        // also drop them, but we apply the drop here
                        // so the lifetime matches the issue's "after
                        // the unique consumer completes its execution
                        // attempt (success/failure/cancel)" rule.
                        self.drop_consumed_single_use_values(&node, consumer_index, &mut session);
                        self.emit_node_event(&node, RunEventKind::NodeCancelled, &[]);
                        self.handle_cancellation(&mut session, &started_at, &artifact_store)
                            .await;
                        return session;
                    }
                }

                // Retention-driven drop: after the node's execution
                // attempt completes (success/failure), walk its
                // edge-sourced input bindings and drop any upstream
                // `SingleUse` value whose unique consumer is this node.
                // Do not run drop logic for nodes that were skipped —
                // skipped nodes did not consume their inputs. The
                // `Cancelled` case is handled inside the arm above so
                // it runs before `handle_cancellation` returns the
                // session early.
                if matches!(
                    session.node_outcome(node_id),
                    Some(NodeOutcome::Completed) | Some(NodeOutcome::Failed { .. })
                ) {
                    self.drop_consumed_single_use_values(&node, consumer_index, &mut session);
                }

                self.publish_snapshot(&session, &started_at, &artifact_store)
                    .await;
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

    /// Check the active-plan fan-out for a `SingleUse` output. Returns a
    /// `Diagnostic` when the value's fan-out is greater than one, so the
    /// caller can fail the run before any downstream consumer sees the
    /// value. `RunScoped` and `WorkspaceScoped` values are not
    /// constrained by the consumer index.
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

    /// Retention-driven drop: for every edge-sourced input binding of
    /// `node`, look up the upstream `OutputKey` and, if the upstream
    /// value is `SingleUse` and `node` is its unique edge-sourced
    /// consumer, drop the value from `RunValueStore`.
    fn drop_consumed_single_use_values(
        &self,
        node: &reimagine_core::readiness::ExecutionNode,
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
        let mut to_drop: Vec<OutputKey> = Vec::new();
        for upstream in upstream_keys {
            let retention = match session.values().retention(&upstream) {
                Some(retention) => retention,
                None => continue,
            };
            if retention != ExecutionValueRetention::SingleUse {
                continue;
            }
            match consumer_index.unique_consumer(&upstream) {
                Some(unique) if unique.to_node_id == *node.node_id() => {
                    to_drop.push(upstream);
                }
                _ => {}
            }
        }
        for key in to_drop {
            session.values_mut().remove(&key);
        }
    }

    async fn execute_node(
        &self,
        node: &reimagine_core::readiness::ExecutionNode,
        session: &RunSession,
        artifact_store: Arc<Mutex<ArtifactStore>>,
    ) -> Result<Vec<crate::value::ExecutionOutput>, NodeFailure> {
        self.emit_node_event(node, RunEventKind::NodeStarted, &[]);
        self.publish_node_running_snapshot(node, session, &artifact_store)
            .await;

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
                            return Err(NodeFailure::Failed(format!(
                                "missing upstream value for {}:{}",
                                from_node_id.as_str(),
                                from_slot_id.as_str()
                            )));
                        }
                    }
                }
                ExecutionInputSource::WorkflowInput {
                    workflow_input_id, ..
                } => {
                    if let Some(value) = self.run_inputs.workflow_input(workflow_input_id) {
                        inputs.insert(binding.slot_id().clone(), value.clone());
                    } else {
                        return Err(NodeFailure::Failed(format!(
                            "missing workflow input {} for slot {}",
                            workflow_input_id.as_str(),
                            binding.slot_id().as_str()
                        )));
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

        let publisher: Arc<dyn ArtifactPublisher> = Arc::new(RuntimeNodeArtifactCapability::new(
            self.run_id.clone(),
            self.plan.workflow_id().clone(),
            self.plan.workflow_version(),
            node.node_id().clone(),
            artifact_store,
            self.sink.clone(),
            self.clock.clone(),
            self.cancellation.clone(),
        ));
        let cancellation: Arc<dyn NodeCancellation> = Arc::new(self.cancellation.clone());

        let ctx = NodeExecutionContext::new(
            self.run_id.clone(),
            self.plan.workflow_id().clone(),
            self.plan.workflow_version(),
            self.options.correlation_id.clone(),
            node.node_id().clone(),
            node.type_id().clone(),
            inputs,
            params,
            publisher,
            cancellation,
            self.clock.now(),
        );

        let executor = self.registry.get(node.type_id()).ok_or_else(|| {
            NodeFailure::Failed(format!("no executor for {}", node.type_id().as_str()))
        })?;
        let result = executor.execute(ctx).await;
        match result {
            Ok(outputs) => Ok(outputs),
            Err(NodeExecutorError::Cancelled) => Err(NodeFailure::Cancelled),
            Err(NodeExecutorError::MissingInput { slot_id }) => {
                Err(NodeFailure::Failed(format!("missing input {slot_id}")))
            }
            Err(NodeExecutorError::Failed { message }) => Err(NodeFailure::Failed(message)),
            Err(NodeExecutorError::Infra { message }) => Err(NodeFailure::Failed(message)),
        }
    }

    async fn handle_cancellation(
        &self,
        session: &mut RunSession,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        // Mark every visited-but-not-terminal node as cancelled and emit
        // NodeCancelled events for them. This only covers nodes that were
        // already touched by the runner.
        for (node_id, outcome) in session.node_outcomes_mut() {
            if matches!(outcome, NodeOutcome::Queued) {
                *outcome = NodeOutcome::Cancelled;
                let event =
                    self.build_event(RunEventKind::NodeCancelled, Some(node_id.clone()), &[]);
                self.safe_emit(event);
            }
        }
        // Now walk the entire plan and emit NodeCancelled for any node that
        // the runner never got to (no entry in the outcome map yet). This
        // includes future-stage nodes that were never visited.
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

    async fn publish_node_running_snapshot(
        &self,
        node: &reimagine_core::readiness::ExecutionNode,
        session: &RunSession,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        let mut node_outcomes = session.node_outcomes().clone();
        node_outcomes.insert(node.node_id().clone(), NodeOutcome::Running);
        self.publish_snapshot_for_outcomes(
            &node_outcomes,
            RunState::Running,
            &self.clock.now(),
            artifact_store,
        )
        .await;
    }

    fn emit_node_skipped(
        &self,
        node_id: &NodeId,
        type_id: &reimagine_core::model::NodeTypeId,
        reason: &str,
    ) {
        let diagnostic = make_diagnostic(&self.run_id, node_id, reason);
        let placeholder_node = reimagine_core::readiness::ExecutionNode::new(
            node_id.clone(),
            type_id.clone(),
            Vec::new(),
            Vec::new(),
        );
        self.emit_node_event(
            &placeholder_node,
            RunEventKind::NodeSkipped,
            std::slice::from_ref(&diagnostic),
        );
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
        node: &reimagine_core::readiness::ExecutionNode,
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

    /// Emit an event to the configured sink, catching any panic raised by
    /// the sink implementation. Sink failure is recorded via `tracing` and
    /// does not fail the run.
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

/// Build a per-node `NodeFailed` style diagnostic.
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

/// Build a run-level diagnostic (used for `RunFailed` and cancellation
/// summaries). Uses a `workflow.run` domain target so it is distinguishable
/// from per-node diagnostics.
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

/// Distinguish failure from cancellation at the runner boundary.
enum NodeFailure {
    Failed(String),
    Cancelled,
}
