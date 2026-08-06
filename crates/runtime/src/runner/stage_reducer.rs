use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::{RunEventKind, Timestamp};
use reimagine_core::model::NodeId;
use reimagine_core::readiness::{ExecutionInputSource, ExecutionNode};
use reimagine_inference::{ExecutionValueRetention, NodeInputs, NodeParams, StageId};
use tokio::sync::Mutex;
use tokio::task::JoinSet;

use super::diagnostics::make_diagnostic;
use super::orchestrator::Runner;
use crate::artifacts::ArtifactStore;
use crate::cancellation::CancellationToken;
use crate::consumer_index::PlanConsumerIndex;
use crate::run_session::{NodeOutcome, RunSession};
use crate::scheduler::{
    AdmissionDecision, DEFAULT_NODE_ESTIMATED_VRAM_BYTES, MemoryBudget, StageExecutionPolicy,
    StageNodeDecision,
};
use crate::stage_runner::{
    PreparedNodeBindings, StageExecutionContext, StageNodePrepareError, StageNodeResult,
    StageNodeWork, execute_stage_node, missing_upstream_value_message,
    missing_workflow_input_message,
};
use crate::value_store::OutputKey;

/// How long the reducer waits for a timed-out task to observe cancellation
/// and return before aborting the stage. Cooperative executors return within
/// milliseconds; a task still running after the grace window is assumed
/// stuck and is aborted together with the stage.
const STAGE_ABORT_GRACE: Duration = Duration::from_secs(1);

/// Diagnostic message for a node that exceeded its execution deadline.
fn node_timeout_message(node_id: &NodeId, timeout: Option<Duration>) -> String {
    match timeout {
        Some(timeout) => format!(
            "node {node_id} exceeded its execution deadline (timeout {timeout:?}); node aborted"
        ),
        None => format!("node {node_id} exceeded its execution deadline; node aborted"),
    }
}

/// Per-node VRAM estimate fed to the memory budget (BE-36).
///
/// The runner holds no `NodeCatalog`, so
/// `NodeResourceRequirements::estimated_vram_bytes` is not reachable
/// from `ExecutionNode`; every node currently falls back to
/// [`DEFAULT_NODE_ESTIMATED_VRAM_BYTES`]. Wire catalog-backed
/// requirements through here when the runtime gains a catalog.
fn estimate_node_vram_bytes(_node: &ExecutionNode) -> usize {
    DEFAULT_NODE_ESTIMATED_VRAM_BYTES
}

/// Out-of-memory-style failure message for a node rejected by the
/// memory budget, mirroring `InferenceError::OutOfMemory`'s wording.
fn node_out_of_memory_message(node_id: &NodeId, estimated: usize, budget: &MemoryBudget) -> String {
    format!(
        "node {node_id} was rejected by the run memory budget: out of memory (requested {estimated} bytes, available {} bytes)",
        budget.available_bytes()
    )
}

/// Shared run-session guard helper used by stage reductions.
///
/// All `session`-mutating helpers below take a `&mut RunSession`; callers
/// that run stages concurrently hold the shared
/// `tokio::sync::Mutex<RunSession>` guard while calling them.
struct StageReductionContext<'a> {
    session: &'a mut RunSession,
    started_at: &'a Timestamp,
    artifact_store: &'a Arc<Mutex<ArtifactStore>>,
    consumer_index: &'a PlanConsumerIndex,
    policy: &'a mut StageExecutionPolicy,
}

impl Runner {
    /// Run every node in `node_ids` to a terminal state.
    ///
    /// `session` and `policy` are shared: parallel stage groups
    /// (BE-35) run multiple stages concurrently against the same
    /// session and fail-fast policy, so all session/policy access is
    /// serialized through the passed mutexes. Each call to this method
    /// owns its own in-flight node deadline tracking (BE-16) and
    /// memory-budget claims (BE-36).
    ///
    /// When `stage_index` is `Some(index)`, values declared
    /// `StageScoped(index)` are released as soon as this stage
    /// completes — before any later stage, including one running
    /// concurrently in the same parallel group, starts (BE-18). The
    /// ready-set path passes `None` (its batches do not map to plan
    /// stages) and releases stage-scoped values at batch boundaries
    /// instead.
    ///
    /// Returns `true` when the run is cancelled (the caller runs the
    /// usual cancellation handling).
    #[allow(
        clippy::too_many_arguments,
        reason = "stage admission needs the runner, node list, shared session/policy, and stage identity"
    )]
    pub(super) async fn run_stage(
        &self,
        node_ids: &[NodeId],
        session: &Mutex<RunSession>,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
        policy: &Mutex<StageExecutionPolicy>,
        stage_index: Option<usize>,
    ) -> bool {
        let cancelled = self
            .run_stage_nodes(
                node_ids,
                session,
                started_at,
                artifact_store,
                consumer_index,
                policy,
            )
            .await;
        if let Some(index) = stage_index {
            let mut session_guard = session.lock().await;
            session_guard
                .values_mut()
                .drop_stage_scoped(StageId::new(index));
        }
        cancelled
    }

    async fn run_stage_nodes(
        &self,
        node_ids: &[NodeId],
        session: &Mutex<RunSession>,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
        policy: &Mutex<StageExecutionPolicy>,
    ) -> bool {
        let max_concurrency = self.options.max_stage_concurrency.unwrap_or(1).max(1);
        let mut joins = JoinSet::new();
        let mut next_index = 0usize;
        let failure_cancellation = CancellationToken::new();
        // node_id -> deadline, for in-flight nodes with an armed deadline.
        let mut in_flight_deadlines: HashMap<NodeId, Instant> = HashMap::new();
        // In-flight nodes whose deadline already expired (reduced as timed
        // out) but whose task has not returned yet.
        let mut expired_in_flight: HashSet<NodeId> = HashSet::new();
        // Estimated VRAM claimed from the shared memory budget per node
        // admitted into this stage; released once the node reaches a
        // terminal state (BE-36).
        let mut claimed_usage: HashMap<NodeId, usize> = HashMap::new();

        while next_index < node_ids.len() || !joins.is_empty() {
            if joins.is_empty() {
                while next_index < node_ids.len()
                    && joins.len() < max_concurrency
                    && policy.lock().await.failed_message().is_none()
                {
                    if self.cancellation.is_cancelled() {
                        failure_cancellation.cancel();
                        return true;
                    }

                    let node_id = &node_ids[next_index];
                    let node = match self.plan.nodes().iter().find(|n| n.node_id() == node_id) {
                        Some(node) => node.clone(),
                        None => {
                            next_index += 1;
                            continue;
                        }
                    };

                    let decision = policy.lock().await.decision_for(node_id);
                    match decision {
                        StageNodeDecision::Skip { reason } => {
                            next_index += 1;
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

                    // Node-level cancellation at admission: this node (or the
                    // run) was cancelled before it started. Only this node is
                    // affected; the stage continues.
                    if self.cancellation.is_node_cancelled(node_id) {
                        next_index += 1;
                        self.reduce_node_cancelled(&node, session, started_at, artifact_store)
                            .await;
                        continue;
                    }

                    // Deadline at admission: the node's deadline was armed
                    // before admission (e.g. by the host) and has already
                    // passed. Fail it without failing the run.
                    if self.cancellation.is_node_expired(node_id) {
                        next_index += 1;
                        let message =
                            node_timeout_message(node_id, self.options.default_node_timeout);
                        self.reduce_node_timed_out(
                            &node,
                            message,
                            session,
                            started_at,
                            artifact_store,
                            consumer_index,
                        )
                        .await;
                        continue;
                    }

                    let prepared = {
                        let session_guard = session.lock().await;
                        self.prepare_stage_node_work(&node, &session_guard)
                    };
                    let work = match prepared {
                        Ok(work) => work,
                        Err(StageNodePrepareError::Failed(message)) => {
                            next_index += 1;
                            let mut session_guard = session.lock().await;
                            let mut policy_guard = policy.lock().await;
                            let mut reduction = StageReductionContext {
                                session: &mut session_guard,
                                started_at,
                                artifact_store,
                                consumer_index,
                                policy: &mut policy_guard,
                            };
                            self.reduce_node_failed(&node, message, &mut reduction)
                                .await;
                            drop(session_guard);
                            drop(policy_guard);
                            failure_cancellation.cancel();
                            continue;
                        }
                    };

                    // Memory budget gate (BE-36): admit the node's
                    // estimated footprint, backpressure while the soft
                    // watermark is crossed, or fail the node when the
                    // hard ceiling would be exceeded. The node index is
                    // only consumed once the node is admitted or
                    // reduced, so a backpressured node is retried on
                    // the next admission pass.
                    if let Some(budget) = &self.options.memory_budget {
                        let estimated = estimate_node_vram_bytes(&node);
                        match budget.admit(estimated) {
                            AdmissionDecision::Allow => {
                                budget.claim(estimated);
                                claimed_usage.insert(node.node_id().clone(), estimated);
                            }
                            AdmissionDecision::Backpressure if !joins.is_empty() => break,
                            // No in-flight node exists to free space (a
                            // single node larger than the soft watermark):
                            // admit anyway — the hard ceiling still
                            // protects the run.
                            AdmissionDecision::Backpressure => {
                                budget.claim(estimated);
                                claimed_usage.insert(node.node_id().clone(), estimated);
                            }
                            AdmissionDecision::Reject => {
                                next_index += 1;
                                let message =
                                    node_out_of_memory_message(node.node_id(), estimated, budget);
                                let mut session_guard = session.lock().await;
                                let mut policy_guard = policy.lock().await;
                                let mut reduction = StageReductionContext {
                                    session: &mut session_guard,
                                    started_at,
                                    artifact_store,
                                    consumer_index,
                                    policy: &mut policy_guard,
                                };
                                self.reduce_node_failed(&node, message, &mut reduction)
                                    .await;
                                drop(session_guard);
                                drop(policy_guard);
                                failure_cancellation.cancel();
                                continue;
                            }
                        }
                    }

                    next_index += 1;
                    {
                        let mut session_guard = session.lock().await;
                        self.admit_stage_node(
                            work.node(),
                            &mut session_guard,
                            started_at,
                            artifact_store,
                        )
                        .await;
                    }

                    // Arm the per-node deadline once the node is admitted.
                    let deadline = self
                        .options
                        .default_node_timeout
                        .map(|timeout| Instant::now() + timeout);
                    if let Some(deadline) = deadline {
                        self.cancellation.set_deadline_at(node_id, deadline);
                        in_flight_deadlines.insert(node_id.clone(), deadline);
                    }

                    let execution = StageExecutionContext {
                        run_id: self.run_id.clone(),
                        workflow_id: self.plan.workflow_id().clone(),
                        workflow_version: self.workflow_version(),
                        correlation_id: self.started_correlation_id(),
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
            }

            if joins.is_empty() {
                break;
            }

            // Wait for the next node result, racing the earliest in-flight
            // deadline. Expired nodes are failed with a timeout diagnostic
            // (node-level, not a run failure).
            let result = match in_flight_deadlines.values().copied().min() {
                Some(deadline) => {
                    tokio::select! {
                        result = joins.join_next() => result,
                        _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
                            self.expire_in_flight_nodes(
                                &mut in_flight_deadlines,
                                &mut expired_in_flight,
                                session,
                                started_at,
                                artifact_store,
                                consumer_index,
                                &mut claimed_usage,
                            )
                            .await;
                            continue;
                        }
                    }
                }
                None if !expired_in_flight.is_empty() => {
                    // Every in-flight node is past its deadline. Give the
                    // expired tasks a grace window to observe cancellation
                    // and return before aborting the stage (dropping the
                    // JoinSet then aborts any task that is truly stuck).
                    tokio::select! {
                        result = joins.join_next() => result,
                        _ = tokio::time::sleep(STAGE_ABORT_GRACE) => {
                            failure_cancellation.cancel();
                            self.skip_stage_remaining(
                                next_index,
                                node_ids,
                                "stage aborted: in-flight node(s) exceeded their execution deadline",
                                session,
                                started_at,
                                artifact_store,
                            )
                            .await;
                            return false;
                        }
                    }
                }
                None => joins.join_next().await,
            };

            let Some(result) = result else {
                break;
            };

            let result = match result {
                Ok(result) => result,
                Err(err) => {
                    tracing::warn!(
                        target: "reimagine_runtime",
                        run_id = %self.run_id.as_str(),
                        error = %err,
                        "stage node task failed to join"
                    );
                    continue;
                }
            };

            let node_id = match &result {
                StageNodeResult::Completed { node, .. } => node.node_id(),
                StageNodeResult::Failed { node, .. } => node.node_id(),
                StageNodeResult::Cancelled { node } => node.node_id(),
            };
            in_flight_deadlines.remove(node_id);
            expired_in_flight.remove(node_id);
            self.release_claimed_budget(node_id, &mut claimed_usage);

            // A node reduced earlier (e.g. timed out while its task was
            // still running) must not be reduced twice.
            if session
                .lock()
                .await
                .node_outcome(node_id)
                .is_some_and(NodeOutcome::is_terminal)
            {
                continue;
            }

            let was_failing = policy.lock().await.failed_message().is_some();
            let mut session_guard = session.lock().await;
            let mut policy_guard = policy.lock().await;
            let mut reduction = StageReductionContext {
                session: &mut session_guard,
                started_at,
                artifact_store,
                consumer_index,
                policy: &mut policy_guard,
            };
            let cancelled = self
                .reduce_stage_node_result(result, was_failing, &mut reduction)
                .await;
            drop(session_guard);
            drop(policy_guard);

            if cancelled {
                failure_cancellation.cancel();
                return true;
            }

            if !was_failing && policy.lock().await.failed_message().is_some() {
                failure_cancellation.cancel();
            }
        }

        if policy.lock().await.failed_message().is_some() {
            while next_index < node_ids.len() {
                let node_id = &node_ids[next_index];
                next_index += 1;
                let node = match self.plan.nodes().iter().find(|n| n.node_id() == node_id) {
                    Some(node) => node.clone(),
                    None => continue,
                };
                if session
                    .lock()
                    .await
                    .node_outcome(node_id)
                    .is_some_and(NodeOutcome::is_terminal)
                {
                    continue;
                }
                let decision = policy.lock().await.decision_for(node_id);
                let reason = match decision {
                    StageNodeDecision::Skip { reason } => reason,
                    StageNodeDecision::Execute => "run is already failing".to_owned(),
                };
                self.reduce_node_skipped(&node, reason, session, started_at, artifact_store)
                    .await;
            }
        }

        self.cancellation.is_cancelled() && policy.lock().await.failed_message().is_none()
    }

    /// Release the memory-budget claim held for `node_id`, if any.
    ///
    /// Idempotent: nodes that were never admitted (skipped, rejected at
    /// admission) or whose claim was already released (timed out while
    /// still in flight) have no entry in `claimed_usage`.
    fn release_claimed_budget(&self, node_id: &NodeId, claimed_usage: &mut HashMap<NodeId, usize>) {
        let Some(estimated) = claimed_usage.remove(node_id) else {
            return;
        };
        if let Some(budget) = &self.options.memory_budget {
            budget.release(estimated);
        }
    }

    /// Fail every in-flight node whose deadline has passed, arming its node
    /// token so the still-running executor can abort cooperatively.
    ///
    /// Timeouts are node-level: the failure is recorded in the session (with
    /// a timeout diagnostic) but not in the stage policy, so the run
    /// continues with the remaining nodes.
    #[allow(clippy::too_many_arguments)]
    async fn expire_in_flight_nodes(
        &self,
        in_flight_deadlines: &mut HashMap<NodeId, Instant>,
        expired_in_flight: &mut HashSet<NodeId>,
        session: &Mutex<RunSession>,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
        claimed_usage: &mut HashMap<NodeId, usize>,
    ) {
        let expired: Vec<NodeId> = in_flight_deadlines
            .iter()
            .filter(|(_, deadline)| Instant::now() >= **deadline)
            .map(|(node_id, _)| node_id.clone())
            .collect();
        for node_id in expired {
            in_flight_deadlines.remove(&node_id);
            expired_in_flight.insert(node_id.clone());
            let Some(node) = self
                .plan
                .nodes()
                .iter()
                .find(|n| n.node_id() == &node_id)
                .cloned()
            else {
                continue;
            };
            // Cooperative: the node token lets the still-running executor
            // abort at its next check point.
            self.cancellation.cancel_node(&node_id);
            let message = node_timeout_message(&node_id, self.options.default_node_timeout);
            self.reduce_node_timed_out(
                &node,
                message,
                session,
                started_at,
                artifact_store,
                consumer_index,
            )
            .await;
            // The timed-out node is terminal, so its budget claim is
            // released here rather than when its task finally returns.
            self.release_claimed_budget(&node_id, claimed_usage);
        }
    }

    /// Mark every not-yet-terminal node in `node_ids[next_index..]` as
    /// skipped with the given reason.
    async fn skip_stage_remaining(
        &self,
        next_index: usize,
        node_ids: &[NodeId],
        reason: &str,
        session: &Mutex<RunSession>,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        for node_id in &node_ids[next_index..] {
            if session
                .lock()
                .await
                .node_outcome(node_id)
                .is_some_and(NodeOutcome::is_terminal)
            {
                continue;
            }
            let Some(node) = self
                .plan
                .nodes()
                .iter()
                .find(|n| n.node_id() == node_id)
                .cloned()
            else {
                continue;
            };
            self.reduce_node_skipped(
                &node,
                reason.to_owned(),
                session,
                started_at,
                artifact_store,
            )
            .await;
        }
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
        reduction: &mut StageReductionContext<'_>,
    ) -> bool {
        match result {
            StageNodeResult::Completed { node, outputs } => {
                let node_id = node.node_id().clone();
                if discard_success {
                    reduction
                        .session
                        .record_outcome(node_id, NodeOutcome::Cancelled);
                    self.drop_consumed_single_use_values(
                        &node,
                        reduction.consumer_index,
                        reduction.session,
                    );
                    self.emit_node_event(&node, RunEventKind::NodeCancelled, &[]);
                    self.publish_snapshot(
                        reduction.session,
                        reduction.started_at,
                        reduction.artifact_store,
                    )
                    .await;
                    return false;
                }

                reduction
                    .session
                    .record_outcome(node_id.clone(), NodeOutcome::Completed);
                for output in outputs {
                    let key = OutputKey::new(node_id.clone(), output.slot_id().clone());
                    let retention = output.retention();
                    if let Some(diag) =
                        self.check_single_use_fan_out(reduction.consumer_index, &key, retention)
                    {
                        let message = diag.message().to_string();
                        self.emit_node_event(
                            &node,
                            RunEventKind::NodeFailed,
                            std::slice::from_ref(&diag),
                        );
                        reduction.session.record_outcome(
                            node_id.clone(),
                            NodeOutcome::Failed {
                                message: message.clone(),
                            },
                        );
                        reduction.policy.record_failure(node_id, message);
                        self.publish_snapshot(
                            reduction.session,
                            reduction.started_at,
                            reduction.artifact_store,
                        )
                        .await;
                        return false;
                    }
                    reduction.session.values_mut().insert_with_retention(
                        key,
                        output.into_value(),
                        retention,
                    );
                }
                self.emit_node_event(&node, RunEventKind::NodeCompleted, &[]);
                self.drop_consumed_single_use_values(
                    &node,
                    reduction.consumer_index,
                    reduction.session,
                );
                self.publish_snapshot(
                    reduction.session,
                    reduction.started_at,
                    reduction.artifact_store,
                )
                .await;
                false
            }
            StageNodeResult::Failed { node, message } => {
                self.reduce_node_failed(&node, message, reduction).await;
                false
            }
            StageNodeResult::Cancelled { node } => {
                let already_failing = reduction.policy.failed_message().is_some();
                reduction
                    .session
                    .record_outcome(node.node_id().clone(), NodeOutcome::Cancelled);
                self.drop_consumed_single_use_values(
                    &node,
                    reduction.consumer_index,
                    reduction.session,
                );
                self.emit_node_event(&node, RunEventKind::NodeCancelled, &[]);
                self.publish_snapshot(
                    reduction.session,
                    reduction.started_at,
                    reduction.artifact_store,
                )
                .await;
                // Classify the cancellation: a run token cancellation, or an
                // unprompted executor Cancelled (no node token armed), stops
                // the whole run. A node-scoped cancellation — the host asked
                // only this node to stop — lets the run continue.
                let run_cancelled = self.cancellation.is_cancelled();
                let node_cancelled = self.cancellation.is_node_cancelled(node.node_id());
                !already_failing && (run_cancelled || !node_cancelled)
            }
        }
    }

    /// Record a node as failed. The failure is recorded in the stage
    /// policy, failing the run fail-fast.
    async fn reduce_node_failed(
        &self,
        node: &ExecutionNode,
        message: String,
        reduction: &mut StageReductionContext<'_>,
    ) {
        let diagnostic = make_diagnostic(&self.run_id, node.node_id(), &message);
        reduction.session.record_outcome(
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
        reduction
            .policy
            .record_failure(node.node_id().clone(), message);
        self.drop_consumed_single_use_values(node, reduction.consumer_index, reduction.session);
        self.publish_snapshot(
            reduction.session,
            reduction.started_at,
            reduction.artifact_store,
        )
        .await;
    }

    /// Record a node as cancelled without affecting the run (node-level
    /// cancellation semantics). Used when a node is cancelled before it is
    /// admitted into the stage.
    async fn reduce_node_cancelled(
        &self,
        node: &ExecutionNode,
        session: &Mutex<RunSession>,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        let mut session_guard = session.lock().await;
        session_guard.record_outcome(node.node_id().clone(), NodeOutcome::Cancelled);
        self.emit_node_event(node, RunEventKind::NodeCancelled, &[]);
        self.publish_snapshot(&session_guard, started_at, artifact_store)
            .await;
    }

    /// Record a node as failed with a timeout diagnostic.
    ///
    /// Unlike [`Runner::reduce_node_failed`], the failure is **not**
    /// recorded in the stage policy: a node that exceeded its deadline is a
    /// node-level failure, and the run continues with the remaining nodes
    /// (see
    /// [`RuntimeOptions::default_node_timeout`](crate::runner::RuntimeOptions)).
    /// Downstream nodes that depend on the timed-out node's outputs fail on
    /// missing values at prepare time, which does fail the run.
    async fn reduce_node_timed_out(
        &self,
        node: &ExecutionNode,
        message: String,
        session: &Mutex<RunSession>,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
        consumer_index: &PlanConsumerIndex,
    ) {
        let diagnostic = make_diagnostic(&self.run_id, node.node_id(), &message);
        let mut session_guard = session.lock().await;
        session_guard.record_outcome(
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
        self.drop_consumed_single_use_values(node, consumer_index, &mut session_guard);
        self.publish_snapshot(&session_guard, started_at, artifact_store)
            .await;
    }

    async fn reduce_node_skipped(
        &self,
        node: &ExecutionNode,
        reason: String,
        session: &Mutex<RunSession>,
        started_at: &Timestamp,
        artifact_store: &Arc<Mutex<ArtifactStore>>,
    ) {
        self.emit_node_skipped(node.node_id(), &node.type_id().clone(), &reason);
        let mut session_guard = session.lock().await;
        session_guard.record_outcome(
            node.node_id().clone(),
            NodeOutcome::Skipped {
                reason: reason.clone(),
            },
        );
        self.publish_snapshot(&session_guard, started_at, artifact_store)
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
                ExecutionInputSource::Edge {
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
}
