use std::sync::Arc;

use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::{RunEventKind, Timestamp};
use reimagine_core::readiness::ExecutionNode;
use reimagine_inference::ExecutionValueRetention;
use tokio::sync::Mutex;

use super::diagnostics::make_diagnostic;
use super::orchestrator::Runner;
use super::stage_reducer::StageReductionContext;
use crate::artifacts::ArtifactStore;
use crate::consumer_index::PlanConsumerIndex;
use crate::run_session::{NodeOutcome, RunSession};
use crate::stage_runner::StageNodeResult;
use crate::value_store::OutputKey;

impl Runner {
    /// Reduce a completed stage node result into the run session.
    ///
    /// Handles three cases:
    /// - `Completed`: records outputs, checks single-use fan-out, drops consumed values
    /// - `Failed`: records failure in session and policy (fail-fast)
    /// - `Cancelled`: records cancellation, determines if run should stop
    ///
    /// Returns `true` if the run should be cancelled.
    pub(super) async fn reduce_stage_node_result(
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
    pub(super) async fn reduce_node_failed(
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
    pub(super) async fn reduce_node_cancelled(
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
    /// node-level failure, and the run continues with the remaining nodes.
    /// Downstream nodes that depend on the timed-out node's outputs fail on
    /// missing values at prepare time, which does fail the run.
    pub(super) async fn reduce_node_timed_out(
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

    pub(super) async fn reduce_node_skipped(
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

    /// Check if a SingleUse output has more than one consumer.
    ///
    /// Returns a diagnostic if the fan-out is invalid.
    pub(super) fn check_single_use_fan_out(
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

    /// Drop SingleUse values that have been consumed by this node.
    pub(super) fn drop_consumed_single_use_values(
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
}
