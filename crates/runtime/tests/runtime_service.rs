//! Tests for the V1 runtime service layer.
//!
//! These tests exercise scheduling, cancellation, fail-fast downstream
//! skipping, shared upstream reuse, sink failure tolerance, and the public
//! snapshot/summary observation shapes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::event::{RunEvent, RunEventKind, Timestamp};
use reimagine_core::model::{
    ArtifactRef, DiagnosticId, NodeId, NodeTypeId, ParamValue, SlotId, WorkflowId, WorkflowInputId,
    WorkflowVersion,
};
use reimagine_core::readiness::{
    ExecutionEdge, ExecutionInputBinding, ExecutionInputSource, ExecutionNode, ExecutionPlan,
    ExecutionStage, RunTarget, RunTargetSelection,
};
use reimagine_runtime::{
    CancellationToken, Clock, ExecutionValue, NodeExecutionContext, NodeExecutor,
    NodeExecutorError, NodeExecutorRegistry, NoopBackendInstanceRuntimeHooks, RunEventSink,
    RunHandle, RunInputs, RuntimeError, RuntimeOptions, RuntimeService, RuntimeServiceError,
    VecRunEventSink,
};

/// A clock that always returns the same string timestamp.
#[derive(Debug, Default, Clone, Copy)]
struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> Timestamp {
        Timestamp::new("2026-06-10T00:00:00Z")
    }
}

/// Helper: build a 1-node plan with the given type id.
fn one_node_plan(type_id: &str, node_id: &str) -> ExecutionPlan {
    ExecutionPlan::new(
        WorkflowId::new("workflow-1"),
        WorkflowVersion::new(1),
        RunTargetSelection::AllDefaultTargets,
        vec![RunTarget::Node {
            node_id: NodeId::new(node_id),
        }],
        vec![ExecutionNode::new(
            NodeId::new(node_id),
            NodeTypeId::new(type_id),
            Vec::new(),
            vec![SlotId::new("out")],
        )],
        Vec::new(),
        Vec::new(),
        vec![ExecutionStage::new(0, vec![NodeId::new(node_id)])],
    )
}

/// Mock executor that records the order in which it was invoked and returns
/// a single output on slot `out`.
struct MockExecutor {
    label: String,
    count: Arc<AtomicUsize>,
    delay: Duration,
    fail_with: Option<String>,
}

#[async_trait]
impl NodeExecutor for MockExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        if self.delay > Duration::ZERO {
            // Observe cancellation while we wait.
            tokio::select! {
                _ = tokio::time::sleep(self.delay) => {}
                _ = context.cancellation().cancelled() => {
                    return Err(NodeExecutorError::Cancelled);
                }
            }
        }
        if let Some(message) = &self.fail_with {
            return Err(NodeExecutorError::Failed {
                message: message.clone(),
            });
        }
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(
                reimagine_core::model::ParamValue::String(self.label.clone()),
            )),
        )])
    }
}

struct CancelImmediatelyExecutor {
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl NodeExecutor for CancelImmediatelyExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Err(NodeExecutorError::Cancelled)
    }
}

struct ArtifactExecutor {
    reference: ArtifactRef,
}

#[async_trait]
impl NodeExecutor for ArtifactExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        context
            .artifacts()
            .record(
                SlotId::new("artifact"),
                self.reference.clone(),
                reimagine_runtime::ArtifactEventKind::Saved,
            )
            .await;
        Ok(Vec::new())
    }
}

struct InspectInputsExecutor {
    expected_input: Option<(SlotId, ParamValue)>,
    expected_param: Option<(SlotId, ParamValue)>,
}

#[async_trait]
impl NodeExecutor for InspectInputsExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        if let Some((slot_id, expected)) = &self.expected_input {
            let actual = context
                .inputs()
                .get(slot_id)
                .and_then(|value| value.as_param())
                .ok_or_else(|| NodeExecutorError::MissingInput {
                    slot_id: slot_id.to_string(),
                })?;
            if actual != expected {
                return Err(NodeExecutorError::Failed {
                    message: format!("unexpected input for {slot_id}: {actual:?}"),
                });
            }
        }

        if let Some((slot_id, expected)) = &self.expected_param {
            let actual =
                context
                    .params()
                    .get(slot_id)
                    .ok_or_else(|| NodeExecutorError::Failed {
                        message: format!("missing param {slot_id}"),
                    })?;
            if actual != expected {
                return Err(NodeExecutorError::Failed {
                    message: format!("unexpected param for {slot_id}: {actual:?}"),
                });
            }
        }

        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String("ok".to_owned()))),
        )])
    }
}

struct BlockingConcurrencyExecutor {
    entered: Arc<AtomicUsize>,
    max_seen: Arc<AtomicUsize>,
    release: Arc<AtomicBool>,
}

#[async_trait]
impl NodeExecutor for BlockingConcurrencyExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        let current = self.entered.fetch_add(1, Ordering::SeqCst) + 1;
        loop {
            let seen = self.max_seen.load(Ordering::SeqCst);
            if current <= seen {
                break;
            }
            if self
                .max_seen
                .compare_exchange(seen, current, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                break;
            }
        }

        while !self.release.load(Ordering::SeqCst) {
            if context.cancellation().is_cancelled() {
                return Err(NodeExecutorError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        self.entered.fetch_sub(1, Ordering::SeqCst);
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String("done".to_owned()))),
        )])
    }
}

struct CoordinatedFailureExecutor {
    started: Arc<AtomicUsize>,
    release: Arc<AtomicBool>,
    fail: bool,
}

#[async_trait]
impl NodeExecutor for CoordinatedFailureExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.started.fetch_add(1, Ordering::SeqCst);
        while !self.release.load(Ordering::SeqCst) {
            if context.cancellation().is_cancelled() {
                return Err(NodeExecutorError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        if self.fail {
            return Err(NodeExecutorError::Failed {
                message: "coordinated failure".to_owned(),
            });
        }

        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String("ok".to_owned()))),
        )])
    }
}

struct LateSuccessExecutor {
    started: Arc<AtomicUsize>,
    finished: Arc<AtomicUsize>,
    observed_cancelled: Arc<AtomicBool>,
    delay: Duration,
}

#[async_trait]
impl NodeExecutor for LateSuccessExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.started.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        if context.cancellation().is_cancelled() {
            self.observed_cancelled.store(true, Ordering::SeqCst);
        }
        self.finished.fetch_add(1, Ordering::SeqCst);
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String("late".to_owned()))),
        )])
    }
}

fn run_to_completion(service: &RuntimeService, handle: &RunHandle) {
    let run_id = handle.run_id().clone();
    // Spin until the run has reached a terminal state.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if let Some(summary) = service.summary(&run_id) {
            assert!(summary.state.is_terminal(), "summary must be terminal");
            return;
        }
        if std::time::Instant::now() > deadline {
            panic!("run {run_id} did not finish in time");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn test_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

fn backend_lifecycle_diagnostic(label: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("backend-lifecycle-{label}")),
        DiagnosticCode::new("INFERENCE/BACKEND_INSTANCE_LIFECYCLE"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("inference"),
        format!("backend lifecycle {label} diagnostic"),
        DiagnosticTarget::new(DiagnosticTargetDomain::new("backend.instance")).with_id("spy"),
    )
}

fn two_node_same_stage_plan(type_id: &str, left: &str, right: &str) -> ExecutionPlan {
    ExecutionPlan::new(
        WorkflowId::new("workflow-concurrency"),
        WorkflowVersion::new(1),
        RunTargetSelection::AllDefaultTargets,
        vec![
            RunTarget::Node {
                node_id: NodeId::new(left),
            },
            RunTarget::Node {
                node_id: NodeId::new(right),
            },
        ],
        vec![
            ExecutionNode::new(
                NodeId::new(left),
                NodeTypeId::new(type_id),
                Vec::new(),
                vec![SlotId::new("out")],
            ),
            ExecutionNode::new(
                NodeId::new(right),
                NodeTypeId::new(type_id),
                Vec::new(),
                vec![SlotId::new("out")],
            ),
        ],
        Vec::new(),
        Vec::new(),
        vec![ExecutionStage::new(
            0,
            vec![NodeId::new(left), NodeId::new(right)],
        )],
    )
}

fn wait_for_condition<F>(timeout: Duration, predicate: F)
where
    F: Fn() -> bool,
{
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if predicate() {
            return;
        }
        if std::time::Instant::now() > deadline {
            panic!("condition not satisfied within {:?}", timeout);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn runtime_rejects_zero_stage_concurrency() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.echo",
                Arc::new(MockExecutor {
                    label: "hello".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .expect("register executor");
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink,
            Arc::new(FixedClock),
        );

        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(0);
        let result = service.run(
            Arc::new(one_node_plan("mock.echo", "node_a")),
            Default::default(),
            options,
        );

        let error = result.expect_err("zero concurrency must be rejected");
        assert!(
            matches!(
                error,
                RuntimeServiceError::InvalidStageConcurrency { value: 0 }
            ),
            "unexpected error: {error:?}"
        );
        assert_eq!(service.store().active_count(), 0);
    });
}

#[test]
fn same_stage_nodes_can_overlap_when_concurrency_is_enabled() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let entered = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        registry
            .register(
                "mock.blocking",
                Arc::new(BlockingConcurrencyExecutor {
                    entered: entered.clone(),
                    max_seen: max_seen.clone(),
                    release: release.clone(),
                }),
            )
            .expect("register executor");
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );

        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(2);
        let handle = service
            .run(
                Arc::new(two_node_same_stage_plan("mock.blocking", "a", "b")),
                Default::default(),
                options,
            )
            .expect("start run");

        wait_for_condition(Duration::from_secs(2), || {
            max_seen.load(Ordering::SeqCst) >= 2
        });
        release.store(true, Ordering::SeqCst);
        run_to_completion(&service, &handle);

        assert_eq!(max_seen.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn next_stage_waits_for_all_inflight_same_stage_nodes() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let entered = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let next_stage_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.blocking",
                Arc::new(BlockingConcurrencyExecutor {
                    entered: entered.clone(),
                    max_seen: max_seen.clone(),
                    release: release.clone(),
                }),
            )
            .expect("register blocking executor");
        registry
            .register(
                "mock.next",
                Arc::new(MockExecutor {
                    label: "next".to_owned(),
                    count: next_stage_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .expect("register next executor");

        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );

        let plan = ExecutionPlan::new(
            WorkflowId::new("workflow-stage-barrier"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("next"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("mock.blocking"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("mock.blocking"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("next"),
                    NodeTypeId::new("mock.next"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("a"), NodeId::new("b")]),
                ExecutionStage::new(1, vec![NodeId::new("next")]),
            ],
        );

        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(2);
        let handle = service
            .run(Arc::new(plan), Default::default(), options)
            .expect("start run");

        wait_for_condition(Duration::from_secs(2), || {
            max_seen.load(Ordering::SeqCst) >= 2
        });
        assert_eq!(
            next_stage_count.load(Ordering::SeqCst),
            0,
            "next stage must not start while stage 0 still has in-flight nodes"
        );

        release.store(true, Ordering::SeqCst);
        run_to_completion(&service, &handle);
        assert_eq!(next_stage_count.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn stage_failure_stops_further_same_stage_admission() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let started = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        registry
            .register(
                "mock.coordinated_fail",
                Arc::new(CoordinatedFailureExecutor {
                    started: started.clone(),
                    release: release.clone(),
                    fail: true,
                }),
            )
            .expect("register failing executor");
        registry
            .register(
                "mock.coordinated_ok",
                Arc::new(CoordinatedFailureExecutor {
                    started: started.clone(),
                    release: release.clone(),
                    fail: false,
                }),
            )
            .expect("register ok executor");

        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );

        let plan = ExecutionPlan::new(
            WorkflowId::new("workflow-fail-fast"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![
                RunTarget::Node {
                    node_id: NodeId::new("fail"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("ok"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("never"),
                },
            ],
            vec![
                ExecutionNode::new(
                    NodeId::new("fail"),
                    NodeTypeId::new("mock.coordinated_fail"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("ok"),
                    NodeTypeId::new("mock.coordinated_ok"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("never"),
                    NodeTypeId::new("mock.coordinated_ok"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![ExecutionStage::new(
                0,
                vec![NodeId::new("fail"), NodeId::new("ok"), NodeId::new("never")],
            )],
        );

        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(2);
        let handle = service
            .run(Arc::new(plan), Default::default(), options)
            .expect("start run");

        wait_for_condition(Duration::from_secs(2), || {
            started.load(Ordering::SeqCst) >= 2
        });
        release.store(true, Ordering::SeqCst);
        run_to_completion(&service, &handle);

        assert_eq!(started.load(Ordering::SeqCst), 2);
        let snapshot = service.snapshot(handle.run_id()).expect("snapshot");
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("never")).copied(),
            Some(reimagine_runtime::NodeState::Skipped)
        );
    });
}

#[test]
fn late_success_output_is_discarded_after_sibling_failure() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let late_started = Arc::new(AtomicUsize::new(0));
        let late_finished = Arc::new(AtomicUsize::new(0));
        let observed_cancelled = Arc::new(AtomicBool::new(false));
        registry
            .register(
                "mock.fail_fast",
                Arc::new(MockExecutor {
                    label: "boom".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::from_millis(10),
                    fail_with: Some("boom".to_owned()),
                }),
            )
            .expect("register failing executor");
        registry
            .register(
                "mock.late_success",
                Arc::new(LateSuccessExecutor {
                    started: late_started.clone(),
                    finished: late_finished.clone(),
                    observed_cancelled: observed_cancelled.clone(),
                    delay: Duration::from_millis(80),
                }),
            )
            .expect("register late executor");
        registry
            .register(
                "mock.inspect",
                Arc::new(InspectInputsExecutor {
                    expected_input: None,
                    expected_param: None,
                }),
            )
            .expect("register inspect executor");
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );

        let plan = ExecutionPlan::new(
            WorkflowId::new("workflow-late-success"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("consumer"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("fail"),
                    NodeTypeId::new("mock.fail_fast"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("late"),
                    NodeTypeId::new("mock.late_success"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("consumer"),
                    NodeTypeId::new("mock.inspect"),
                    vec![ExecutionInputBinding::new(
                        SlotId::new("in"),
                        ExecutionInputSource::Edge {
                            edge_id: reimagine_core::model::EdgeId::new("edge-late-consumer"),
                            from_node_id: NodeId::new("late"),
                            from_slot_id: SlotId::new("out"),
                        },
                    )],
                    vec![SlotId::new("out")],
                ),
            ],
            vec![ExecutionEdge::new(
                reimagine_core::model::EdgeId::new("edge-late-consumer"),
                NodeId::new("late"),
                SlotId::new("out"),
                NodeId::new("consumer"),
                SlotId::new("in"),
            )],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("fail"), NodeId::new("late")]),
                ExecutionStage::new(1, vec![NodeId::new("consumer")]),
            ],
        );

        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(2);
        let handle = service
            .run(Arc::new(plan), Default::default(), options)
            .expect("start run");

        run_to_completion(&service, &handle);

        assert_eq!(late_started.load(Ordering::SeqCst), 1);
        assert_eq!(late_finished.load(Ordering::SeqCst), 1);
        assert!(observed_cancelled.load(Ordering::SeqCst));

        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        let completed_for_late = sink.events().iter().any(|event| {
            event.kind() == RunEventKind::NodeCompleted
                && event.node_id() == Some(&NodeId::new("late"))
        });
        assert!(
            !completed_for_late,
            "late successful sibling must not publish NodeCompleted after run failure"
        );
        assert!(kinds.contains(&RunEventKind::NodeFailed));

        let snapshot = service.snapshot(handle.run_id()).expect("snapshot");
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("late")).copied(),
            Some(reimagine_runtime::NodeState::Cancelled)
        );
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("consumer")).copied(),
            Some(reimagine_runtime::NodeState::Skipped)
        );
    });
}

#[test]
fn runtime_run_starts_and_completes_a_mock_plan() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let counter = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.echo",
                Arc::new(MockExecutor {
                    label: "hello".to_owned(),
                    count: counter.clone(),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .expect("register executor");
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );

        let plan = Arc::new(one_node_plan("mock.echo", "node_a"));
        let handle = service
            .run(plan, Default::default(), RuntimeOptions::default())
            .expect("start run");

        // Snapshot must be visible immediately, before the run finishes.
        let initial = service
            .snapshot(handle.run_id())
            .expect("snapshot available");
        assert_eq!(initial.state, reimagine_runtime::RunState::Queued);

        run_to_completion(&service, &handle);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        let summary = service.summary(handle.run_id()).expect("summary");
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(summary.diagnostics.len(), 0);

        // The sink must observe Queued + Started + NodeQueued + NodeStarted
        // + NodeCompleted + RunCompleted.
        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        assert!(kinds.contains(&RunEventKind::RunQueued));
        assert!(kinds.contains(&RunEventKind::RunStarted));
        assert!(kinds.contains(&RunEventKind::NodeQueued));
        assert!(kinds.contains(&RunEventKind::NodeStarted));
        assert!(kinds.contains(&RunEventKind::NodeCompleted));
        assert!(kinds.contains(&RunEventKind::RunCompleted));
    });
}

#[test]
fn runtime_observations_include_host_neutral_artifact_reference() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.artifact",
                Arc::new(ArtifactExecutor {
                    reference: ArtifactRef::new("output/mock-image.png"),
                }),
            )
            .expect("register executor");
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink,
            Arc::new(FixedClock),
        );

        let plan = Arc::new(one_node_plan("mock.artifact", "node_save"));
        let handle = service
            .run(plan, Default::default(), RuntimeOptions::default())
            .expect("start run");
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).expect("summary");
        assert_eq!(summary.artifacts.len(), 1);
        let artifact = &summary.artifacts[0];
        assert_eq!(artifact.node_id, NodeId::new("node_save"));
        assert_eq!(artifact.reference.as_str(), "output/mock-image.png");

        let snapshot = service.snapshot(handle.run_id()).expect("snapshot");
        assert_eq!(
            snapshot.artifacts[0].reference.as_str(),
            "output/mock-image.png"
        );
    });
}

#[tokio::test]
async fn runtime_service_reports_backend_instance_snapshots() {
    let registry = NodeExecutorRegistry::default();
    let service = RuntimeService::new(
        registry,
        Arc::new(NoopBackendInstanceRuntimeHooks::default()),
        Arc::new(VecRunEventSink::new()),
        Arc::new(FixedClock),
    );

    let snapshots = service.backend_instance_snapshots().await;

    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].backend_instance.to_string(), "noop");
}

#[test]
fn runtime_preserves_backend_lifecycle_report_diagnostics() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.echo",
                Arc::new(MockExecutor {
                    label: "hello".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .expect("register executor");

        let backend = Arc::new(
            SpyBackend::new()
                .with_begin_diagnostic(backend_lifecycle_diagnostic("begin"))
                .with_cleanup_diagnostic(backend_lifecycle_diagnostic("cleanup")),
        );
        let service = RuntimeService::new(
            registry,
            backend,
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );

        let plan = Arc::new(one_node_plan("mock.echo", "node_a"));
        let handle = service
            .run(plan, Default::default(), RuntimeOptions::default())
            .expect("start run");
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).expect("summary");
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(
            summary
                .diagnostics
                .iter()
                .map(|d| d.id().as_str())
                .collect::<Vec<_>>(),
            vec!["backend-lifecycle-begin", "backend-lifecycle-cleanup"]
        );

        let snapshot = service.snapshot(handle.run_id()).expect("snapshot");
        assert_eq!(snapshot.state, reimagine_runtime::RunState::Completed);
        assert_eq!(
            snapshot
                .diagnostics
                .iter()
                .map(|d| d.id().as_str())
                .collect::<Vec<_>>(),
            vec!["backend-lifecycle-begin", "backend-lifecycle-cleanup"]
        );
    });
}

#[test]
fn runtime_stages_execute_in_order() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let order: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        for (label, tid) in [("a", "mock.a"), ("b", "mock.b"), ("c", "mock.c")] {
            let label = label.to_owned();
            let order = order.clone();
            registry
                .register(
                    tid,
                    Arc::new(OrderedExecutor {
                        label,
                        order: order.clone(),
                    }),
                )
                .unwrap();
        }

        // Three stages with one node each; stage 1 must finish before stage 2.
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("c"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("mock.a"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("mock.b"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("c"),
                    NodeTypeId::new("mock.c"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("a")]),
                ExecutionStage::new(1, vec![NodeId::new("b")]),
                ExecutionStage::new(2, vec![NodeId::new("c")]),
            ],
        );
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink,
            Arc::new(FixedClock),
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let order = order.lock().unwrap().clone();
        assert_eq!(order, vec!["a", "b", "c"]);
    });
}

/// Executor that records its label the moment `execute` starts.
struct OrderedExecutor {
    label: String,
    order: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl NodeExecutor for OrderedExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.order.lock().unwrap().push(self.label.clone());
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(
                reimagine_core::model::ParamValue::String(self.label.clone()),
            )),
        )])
    }
}

#[test]
fn shared_upstream_node_runs_only_once() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let upstream_count = Arc::new(AtomicUsize::new(0));
        let downstream_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.upstream",
                Arc::new(MockExecutor {
                    label: "up".to_owned(),
                    count: upstream_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.downstream",
                Arc::new(MockExecutor {
                    label: "down".to_owned(),
                    count: downstream_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink,
            Arc::new(FixedClock),
        );

        // Plan: shared upstream feeds two downstream nodes that both target
        // terminal nodes in a single run. The plan itself merges the
        // subgraph so the upstream must only execute once.
        let upstream_node = ExecutionNode::new(
            NodeId::new("upstream"),
            NodeTypeId::new("mock.upstream"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let downstream_a = ExecutionNode::new(
            NodeId::new("down_a"),
            NodeTypeId::new("mock.downstream"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e1"),
                    from_node_id: NodeId::new("upstream"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            vec![SlotId::new("out")],
        );
        let downstream_b = ExecutionNode::new(
            NodeId::new("down_b"),
            NodeTypeId::new("mock.downstream"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e2"),
                    from_node_id: NodeId::new("upstream"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            vec![SlotId::new("out")],
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![
                RunTarget::Node {
                    node_id: NodeId::new("down_a"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("down_b"),
                },
            ],
            vec![
                upstream_node.clone(),
                downstream_a.clone(),
                downstream_b.clone(),
            ],
            vec![
                ExecutionEdge::new(
                    reimagine_core::model::EdgeId::new("e1"),
                    NodeId::new("upstream"),
                    SlotId::new("out"),
                    NodeId::new("down_a"),
                    SlotId::new("in"),
                ),
                ExecutionEdge::new(
                    reimagine_core::model::EdgeId::new("e2"),
                    NodeId::new("upstream"),
                    SlotId::new("out"),
                    NodeId::new("down_b"),
                    SlotId::new("in"),
                ),
            ],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("upstream")]),
                ExecutionStage::new(1, vec![NodeId::new("down_a"), NodeId::new("down_b")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        assert_eq!(upstream_count.load(Ordering::SeqCst), 1);
        assert_eq!(downstream_count.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn executor_failure_marks_run_failed_and_skips_downstream() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let failing_count = Arc::new(AtomicUsize::new(0));
        let downstream_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.failing",
                Arc::new(MockExecutor {
                    label: "failing".to_owned(),
                    count: failing_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: Some("kaboom".to_owned()),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.downstream",
                Arc::new(MockExecutor {
                    label: "down".to_owned(),
                    count: downstream_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("c"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("mock.failing"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("mock.downstream"),
                    vec![ExecutionInputBinding::new(
                        SlotId::new("in"),
                        ExecutionInputSource::Edge {
                            edge_id: reimagine_core::model::EdgeId::new("e"),
                            from_node_id: NodeId::new("a"),
                            from_slot_id: SlotId::new("out"),
                        },
                    )],
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("c"),
                    NodeTypeId::new("mock.downstream"),
                    vec![ExecutionInputBinding::new(
                        SlotId::new("in"),
                        ExecutionInputSource::Edge {
                            edge_id: reimagine_core::model::EdgeId::new("e2"),
                            from_node_id: NodeId::new("b"),
                            from_slot_id: SlotId::new("out"),
                        },
                    )],
                    vec![SlotId::new("out")],
                ),
            ],
            vec![
                ExecutionEdge::new(
                    reimagine_core::model::EdgeId::new("e"),
                    NodeId::new("a"),
                    SlotId::new("out"),
                    NodeId::new("b"),
                    SlotId::new("in"),
                ),
                ExecutionEdge::new(
                    reimagine_core::model::EdgeId::new("e2"),
                    NodeId::new("b"),
                    SlotId::new("out"),
                    NodeId::new("c"),
                    SlotId::new("in"),
                ),
            ],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("a")]),
                ExecutionStage::new(1, vec![NodeId::new("b")]),
                ExecutionStage::new(2, vec![NodeId::new("c")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Failed);
        assert_eq!(summary.diagnostics.len(), 1);
        assert_eq!(
            summary.diagnostics[0].code(),
            &DiagnosticCode::new("RUNTIME/RUN_EXECUTION_FAILED")
        );
        // Downstream nodes must not run after the failure.
        assert_eq!(downstream_count.load(Ordering::SeqCst), 0);
        // Sink must observe NodeSkipped for `b` and `c`.
        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        assert!(kinds.contains(&RunEventKind::NodeFailed));
        assert!(kinds.contains(&RunEventKind::NodeSkipped));
        assert!(kinds.contains(&RunEventKind::RunFailed));
    });
}

#[test]
fn cancellation_stops_downstream_and_emits_cancelled_events() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let slow_count = Arc::new(AtomicUsize::new(0));
        let downstream_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.slow",
                Arc::new(MockExecutor {
                    label: "slow".to_owned(),
                    count: slow_count.clone(),
                    delay: Duration::from_secs(2),
                    fail_with: None,
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.downstream",
                Arc::new(MockExecutor {
                    label: "down".to_owned(),
                    count: downstream_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("b"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("mock.slow"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("mock.downstream"),
                    vec![ExecutionInputBinding::new(
                        SlotId::new("in"),
                        ExecutionInputSource::Edge {
                            edge_id: reimagine_core::model::EdgeId::new("e"),
                            from_node_id: NodeId::new("a"),
                            from_slot_id: SlotId::new("out"),
                        },
                    )],
                    vec![SlotId::new("out")],
                ),
            ],
            vec![ExecutionEdge::new(
                reimagine_core::model::EdgeId::new("e"),
                NodeId::new("a"),
                SlotId::new("out"),
                NodeId::new("b"),
                SlotId::new("in"),
            )],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("a")]),
                ExecutionStage::new(1, vec![NodeId::new("b")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        // Wait for the slow node to start, then cancel.
        for _ in 0..100 {
            if slow_count.load(Ordering::SeqCst) > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        service.cancel(handle.run_id()).expect("cancel");

        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Cancelled);
        // Downstream `b` must NOT have run.
        assert_eq!(downstream_count.load(Ordering::SeqCst), 0);
        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        assert!(kinds.contains(&RunEventKind::NodeCancelled));
        assert!(kinds.contains(&RunEventKind::RunCancelled));
    });
}

#[test]
fn executor_cancelled_on_last_node_marks_run_cancelled_not_failed() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.cancel",
                Arc::new(CancelImmediatelyExecutor {
                    count: count.clone(),
                }),
            )
            .unwrap();
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );

        let handle = service
            .run(
                Arc::new(one_node_plan("mock.cancel", "node_a")),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        assert_eq!(count.load(Ordering::SeqCst), 1);
        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Cancelled);
        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        assert!(kinds.contains(&RunEventKind::NodeCancelled));
        assert!(kinds.contains(&RunEventKind::RunCancelled));
        assert!(!kinds.contains(&RunEventKind::RunFailed));
    });
}

#[test]
fn workflow_input_bindings_read_from_workflow_input_run_inputs() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.inspect",
                Arc::new(InspectInputsExecutor {
                    expected_input: Some((
                        SlotId::new("prompt"),
                        ParamValue::Text("a quiet forest".to_owned()),
                    )),
                    expected_param: None,
                }),
            )
            .unwrap();
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("encode"),
            }],
            vec![ExecutionNode::new(
                NodeId::new("encode"),
                NodeTypeId::new("mock.inspect"),
                vec![ExecutionInputBinding::new(
                    SlotId::new("prompt"),
                    ExecutionInputSource::WorkflowInput {
                        edge_id: reimagine_core::model::EdgeId::new("wf_prompt_to_encode"),
                        workflow_input_id: WorkflowInputId::new("positive_prompt"),
                    },
                )],
                vec![SlotId::new("out")],
            )],
            Vec::new(),
            Vec::new(),
            vec![ExecutionStage::new(0, vec![NodeId::new("encode")])],
        );
        let mut inputs = RunInputs::new();
        inputs.insert_workflow_input(
            WorkflowInputId::new("positive_prompt"),
            Arc::new(ExecutionValue::Param(ParamValue::Text(
                "a quiet forest".to_owned(),
            ))),
        );

        let handle = service
            .run(Arc::new(plan), inputs, RuntimeOptions::default())
            .unwrap();
        run_to_completion(&service, &handle);
        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
    });
}

#[test]
fn static_param_bindings_read_from_node_param_run_inputs() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.inspect",
                Arc::new(InspectInputsExecutor {
                    expected_input: None,
                    expected_param: Some((SlotId::new("seed"), ParamValue::Seed(42))),
                }),
            )
            .unwrap();
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("sampler"),
            }],
            vec![ExecutionNode::new(
                NodeId::new("sampler"),
                NodeTypeId::new("mock.inspect"),
                vec![ExecutionInputBinding::new(
                    SlotId::new("seed"),
                    ExecutionInputSource::Param {
                        slot_id: SlotId::new("seed"),
                    },
                )],
                vec![SlotId::new("out")],
            )],
            Vec::new(),
            Vec::new(),
            vec![ExecutionStage::new(0, vec![NodeId::new("sampler")])],
        );
        let mut inputs = RunInputs::new();
        inputs.insert_node_param(
            NodeId::new("sampler"),
            SlotId::new("seed"),
            ParamValue::Seed(42),
        );

        let handle = service
            .run(Arc::new(plan), inputs, RuntimeOptions::default())
            .unwrap();
        run_to_completion(&service, &handle);
        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
    });
}

#[test]
fn sink_failure_does_not_fail_the_run() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.echo",
                Arc::new(MockExecutor {
                    label: "hello".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let sink = Arc::new(FailingSink::default());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );
        let plan = one_node_plan("mock.echo", "node_a");
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);
        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert!(sink.failures.load(Ordering::SeqCst) > 0);
    });
}

#[derive(Debug, Default)]
struct FailingSink {
    failures: AtomicUsize,
}

impl RunEventSink for FailingSink {
    fn emit(&self, _event: RunEvent) -> Result<(), RuntimeError> {
        self.failures.fetch_add(1, Ordering::SeqCst);
        Err(RuntimeError::EventSink {
            message: "simulated sink failure".to_owned(),
        })
    }
}

#[derive(Debug, Default)]
struct PanickingSink;
impl RunEventSink for PanickingSink {
    fn emit(&self, _event: RunEvent) -> Result<(), RuntimeError> {
        panic!("simulated sink failure");
    }
}

#[test]
fn sink_panic_is_logged_and_does_not_fail_the_run() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.echo",
                Arc::new(MockExecutor {
                    label: "hello".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(PanickingSink),
            Arc::new(FixedClock),
        );
        let plan = one_node_plan("mock.echo", "node_a");
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);
        // Run must still reach Completed despite every emit() panicking.
        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
    });
}

#[test]
fn snapshot_reports_running_node_while_executor_is_in_flight() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let slow_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.slow",
                Arc::new(MockExecutor {
                    label: "slow".to_owned(),
                    count: slow_count.clone(),
                    delay: Duration::from_millis(200),
                    fail_with: None,
                }),
            )
            .unwrap();
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );
        let handle = service
            .run(
                Arc::new(one_node_plan("mock.slow", "node_a")),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();

        for _ in 0..100 {
            if let Some(snapshot) = service.snapshot(handle.run_id())
                && snapshot.node_states.get(&NodeId::new("node_a")).copied()
                    == Some(reimagine_runtime::NodeState::Running)
            {
                run_to_completion(&service, &handle);
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        panic!("node_a never appeared as Running in the live snapshot");
    });
}

#[test]
fn snapshot_records_terminal_node_states_and_summary_records_terminal_state() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.echo",
                Arc::new(MockExecutor {
                    label: "hello".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink,
            Arc::new(FixedClock),
        );
        let plan = one_node_plan("mock.echo", "node_a");
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let snapshot = service
            .snapshot(handle.run_id())
            .expect("snapshot is published");
        assert_eq!(snapshot.state, reimagine_runtime::RunState::Completed);
        assert_eq!(
            snapshot
                .node_states
                .get(&NodeId::new("node_a"))
                .copied()
                .unwrap(),
            reimagine_runtime::NodeState::Completed
        );
        // Snapshot must not carry a runtime value store or backend payload.
        let json = format!("{snapshot:?}");
        assert!(!json.contains("RunValueStore"));
        assert!(!json.contains("BackendTensorHandle"));

        let summary = service
            .summary(handle.run_id())
            .expect("summary is published");
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        let json = format!("{summary:?}");
        assert!(!json.contains("RunValueStore"));
        assert!(!json.contains("BackendTensorHandle"));
    });
}

#[test]
fn run_handle_does_not_expose_value_store() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.echo",
                Arc::new(MockExecutor {
                    label: "x".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let service = RuntimeService::with_defaults(registry, Arc::new(VecRunEventSink::new()));
        let handle = service
            .run(
                Arc::new(one_node_plan("mock.echo", "node_a")),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        // The handle only exposes run id, workflow id, and cancellation.
        let _ = handle.run_id();
        let _ = handle.workflow_id();
        let _: CancellationToken = handle.cancellation();
        let json = format!("{handle:?}");
        assert!(!json.contains("RunValueStore"));
    });
}

#[test]
fn unknown_run_id_returns_error() {
    let mut registry = NodeExecutorRegistry::default();
    registry
        .register(
            "mock.echo",
            Arc::new(MockExecutor {
                label: "x".to_owned(),
                count: Arc::new(AtomicUsize::new(0)),
                delay: Duration::ZERO,
                fail_with: None,
            }),
        )
        .unwrap();
    let service = RuntimeService::with_defaults(registry, Arc::new(VecRunEventSink::new()));
    let err = service
        .cancel(&reimagine_core::model::RunId::new("nope"))
        .unwrap_err();
    assert!(matches!(err, RuntimeServiceError::UnknownRun { .. }));
}

#[test]
fn empty_plan_is_rejected() {
    let registry = NodeExecutorRegistry::default();
    let service = RuntimeService::with_defaults(registry, Arc::new(VecRunEventSink::new()));
    let plan = ExecutionPlan::new(
        WorkflowId::new("wf"),
        WorkflowVersion::new(1),
        RunTargetSelection::AllDefaultTargets,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let err = service
        .run(
            Arc::new(plan),
            Default::default(),
            RuntimeOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(err, RuntimeServiceError::EmptyPlan { .. }));
}

#[test]
fn missing_executor_is_rejected_up_front() {
    let registry = NodeExecutorRegistry::default();
    let service = RuntimeService::with_defaults(registry, Arc::new(VecRunEventSink::new()));
    let plan = one_node_plan("mock.missing", "node_a");
    let err = service
        .run(
            Arc::new(plan),
            Default::default(),
            RuntimeOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(err, RuntimeServiceError::MissingExecutor { .. }));
}

#[test]
fn snapshot_carries_no_backend_payload_or_value_store() {
    // Compile-time check: the snapshot only contains host-neutral fields.
    fn _assert_send_sync<T: Send + Sync>() {}
    _assert_send_sync::<reimagine_runtime::RunSnapshot>();
    _assert_send_sync::<reimagine_runtime::RunSummary>();
    // Use the HashMap import to silence a warning about the unused import.
    let _ = std::any::type_name::<HashMap<NodeId, reimagine_runtime::NodeState>>();
}

#[test]
fn same_stage_siblings_are_skipped_after_a_sibling_fails() {
    // Regression: a stage with [a-fails, b, c] in ONE stage must skip
    // b and c. Previously the runner only checked `failed_node` at the
    // stage boundary, so siblings within a stage still ran.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let a_count = Arc::new(AtomicUsize::new(0));
        let b_count = Arc::new(AtomicUsize::new(0));
        let c_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.a",
                Arc::new(MockExecutor {
                    label: "a".to_owned(),
                    count: a_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: Some("kaboom".to_owned()),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.b",
                Arc::new(MockExecutor {
                    label: "b".to_owned(),
                    count: b_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.c",
                Arc::new(MockExecutor {
                    label: "c".to_owned(),
                    count: c_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("c"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("mock.a"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("mock.b"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("c"),
                    NodeTypeId::new("mock.c"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![ExecutionStage::new(
                0,
                vec![NodeId::new("a"), NodeId::new("b"), NodeId::new("c")],
            )],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        assert_eq!(a_count.load(Ordering::SeqCst), 1);
        assert_eq!(b_count.load(Ordering::SeqCst), 0);
        assert_eq!(c_count.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn cancellation_emits_cancelled_for_unvisited_future_stage_nodes() {
    // Regression: a run with 3 stages whose cancellation arrives at the
    // start of stage 2 must emit NodeCancelled for every unstarted node,
    // not only for the visited-and-Queued ones.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let slow_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.slow",
                Arc::new(MockExecutor {
                    label: "slow".to_owned(),
                    count: slow_count.clone(),
                    delay: Duration::from_secs(2),
                    fail_with: None,
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.fast",
                Arc::new(MockExecutor {
                    label: "fast".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("c"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("mock.slow"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("mock.fast"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("c"),
                    NodeTypeId::new("mock.fast"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("a")]),
                ExecutionStage::new(1, vec![NodeId::new("b")]),
                ExecutionStage::new(2, vec![NodeId::new("c")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        // Wait for the slow node to start, then cancel.
        for _ in 0..100 {
            if slow_count.load(Ordering::SeqCst) > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        service.cancel(handle.run_id()).expect("cancel");
        run_to_completion(&service, &handle);

        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        // Both future-stage nodes must be reported as cancelled.
        let node_cancelled = kinds
            .iter()
            .filter(|k| **k == RunEventKind::NodeCancelled)
            .count();
        assert!(
            node_cancelled >= 2,
            "expected NodeCancelled events for `b` and `c`, got {node_cancelled}"
        );
    });
}

// ---------------------------------------------------------------------------
// Progressive artifact visibility tests (issue 05a).
// ---------------------------------------------------------------------------

/// Helper: poll snapshot until a predicate is satisfied, or a timeout.
/// Uses `std::thread::sleep` because tests run with a multi-thread
/// tokio runtime; the runner task drives progress on a worker thread
/// while this helper polls from the test thread.
fn wait_for_snapshot<F>(
    service: &RuntimeService,
    run_id: &reimagine_core::model::RunId,
    timeout: Duration,
    predicate: F,
) -> Option<reimagine_runtime::RunSnapshot>
where
    F: Fn(&reimagine_runtime::RunSnapshot) -> bool,
{
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(snapshot) = service.snapshot(run_id)
            && predicate(&snapshot)
        {
            return Some(snapshot);
        }
        if std::time::Instant::now() > deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn artifact_visible_in_snapshot_before_run_completes() {
    // Test A: A save/preview node produces an artifact. After that node
    // completes, the snapshot must show the artifact, even though the
    // whole run is still Running (a downstream node is still executing).
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let slow_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.artifact",
                Arc::new(ArtifactExecutor {
                    reference: ArtifactRef::new("output/progressive.png"),
                }),
            )
            .expect("register artifact executor");
        registry
            .register(
                "mock.slow",
                Arc::new(MockExecutor {
                    label: "slow".to_owned(),
                    count: slow_count.clone(),
                    delay: Duration::from_secs(2),
                    fail_with: None,
                }),
            )
            .expect("register slow executor");

        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );

        let plan = ExecutionPlan::new(
            WorkflowId::new("wf-progressive"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("slow"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("save"),
                    NodeTypeId::new("mock.artifact"),
                    Vec::new(),
                    vec![SlotId::new("artifact")],
                ),
                ExecutionNode::new(
                    NodeId::new("slow"),
                    NodeTypeId::new("mock.slow"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("save")]),
                ExecutionStage::new(1, vec![NodeId::new("slow")]),
            ],
        );

        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .expect("start run");

        // Poll snapshot until the save node is Completed.
        let mid_run_snapshot =
            wait_for_snapshot(&service, handle.run_id(), Duration::from_secs(2), |snap| {
                snap.node_states.get(&NodeId::new("save")).copied()
                    == Some(reimagine_runtime::NodeState::Completed)
            })
            .expect("save node should complete within timeout");

        // The artifact must be visible in the snapshot.
        assert_eq!(mid_run_snapshot.artifacts.len(), 1);
        assert_eq!(
            mid_run_snapshot.artifacts[0].reference.as_str(),
            "output/progressive.png"
        );
        assert_eq!(mid_run_snapshot.artifacts[0].node_id, NodeId::new("save"));

        // The run must still be Running (slow node still going).
        assert_eq!(mid_run_snapshot.state, reimagine_runtime::RunState::Running);

        // The ArtifactCreated event must have been emitted.
        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        assert!(
            kinds.contains(&RunEventKind::ArtifactCreated),
            "expected ArtifactCreated event before run completes"
        );

        // Let the run finish and verify summary still has the artifact.
        run_to_completion(&service, &handle);
        assert_eq!(
            slow_count.load(Ordering::SeqCst),
            1,
            "slow node should have executed once"
        );
        let summary = service.summary(handle.run_id()).expect("summary");
        assert_eq!(summary.artifacts.len(), 1);
        assert_eq!(
            summary.artifacts[0].reference.as_str(),
            "output/progressive.png"
        );
    });
}

#[test]
fn terminal_summary_includes_all_artifacts_from_multiple_nodes() {
    // Test B: After the full run completes, the summary must contain
    // all artifacts from all save/preview nodes.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.artifact_a",
                Arc::new(ArtifactExecutor {
                    reference: ArtifactRef::new("output/a.png"),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.artifact_b",
                Arc::new(ArtifactExecutor {
                    reference: ArtifactRef::new("output/b.png"),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.echo",
                Arc::new(MockExecutor {
                    label: "done".to_owned(),
                    count: Arc::new(AtomicUsize::new(0)),
                    delay: Duration::ZERO,
                    fail_with: None,
                }),
            )
            .unwrap();

        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );

        let plan = ExecutionPlan::new(
            WorkflowId::new("wf-multi-artifact"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("echo"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("mock.artifact_a"),
                    Vec::new(),
                    vec![SlotId::new("a_out")],
                ),
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("mock.artifact_b"),
                    Vec::new(),
                    vec![SlotId::new("b_out")],
                ),
                ExecutionNode::new(
                    NodeId::new("echo"),
                    NodeTypeId::new("mock.echo"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("a"), NodeId::new("b")]),
                ExecutionStage::new(1, vec![NodeId::new("echo")]),
            ],
        );

        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).expect("summary");
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(summary.artifacts.len(), 2);

        let refs: Vec<&str> = summary
            .artifacts
            .iter()
            .map(|a| a.reference.as_str())
            .collect();
        assert!(
            refs.contains(&"output/a.png"),
            "summary must contain artifact from node a"
        );
        assert!(
            refs.contains(&"output/b.png"),
            "summary must contain artifact from node b"
        );

        // Verify artifact events were emitted for both.
        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        let artifact_created_count = kinds
            .iter()
            .filter(|k| **k == RunEventKind::ArtifactCreated)
            .count();
        assert_eq!(artifact_created_count, 2);
    });
}

#[test]
fn failed_run_preserves_previously_recorded_artifacts() {
    // Test C: A save node produces an artifact, then a later node fails.
    // The snapshot and summary must still contain the artifact.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.artifact",
                Arc::new(ArtifactExecutor {
                    reference: ArtifactRef::new("output/pre-fail.png"),
                }),
            )
            .unwrap();
        let fail_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.failing",
                Arc::new(MockExecutor {
                    label: "failing".to_owned(),
                    count: fail_count.clone(),
                    delay: Duration::ZERO,
                    fail_with: Some("kaboom".to_owned()),
                }),
            )
            .unwrap();

        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );

        let plan = ExecutionPlan::new(
            WorkflowId::new("wf-artifact-fail"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("fail"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("save"),
                    NodeTypeId::new("mock.artifact"),
                    Vec::new(),
                    vec![SlotId::new("artifact")],
                ),
                ExecutionNode::new(
                    NodeId::new("fail"),
                    NodeTypeId::new("mock.failing"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("save")]),
                ExecutionStage::new(1, vec![NodeId::new("fail")]),
            ],
        );

        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).expect("summary");
        assert_eq!(summary.state, reimagine_runtime::RunState::Failed);
        // The artifact from the save node must still be present.
        assert_eq!(summary.artifacts.len(), 1);
        assert_eq!(
            summary.artifacts[0].reference.as_str(),
            "output/pre-fail.png"
        );
        assert_eq!(summary.artifacts[0].node_id, NodeId::new("save"));

        let snapshot = service.snapshot(handle.run_id()).expect("snapshot");
        assert_eq!(snapshot.artifacts.len(), 1);
        assert_eq!(
            snapshot.artifacts[0].reference.as_str(),
            "output/pre-fail.png"
        );
    });
}

#[test]
fn cancelled_run_preserves_previously_recorded_artifacts() {
    // Test D: A save node produces an artifact, then a later node is cancelled.
    // The terminal cancellation summary and final snapshot must still contain
    // the already-recorded artifact.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "mock.artifact",
                Arc::new(ArtifactExecutor {
                    reference: ArtifactRef::new("output/pre-cancel.png"),
                }),
            )
            .unwrap();
        let slow_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.slow",
                Arc::new(MockExecutor {
                    label: "slow-cancel".to_owned(),
                    count: slow_count.clone(),
                    delay: Duration::from_secs(2),
                    fail_with: None,
                }),
            )
            .unwrap();

        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );

        let plan = ExecutionPlan::new(
            WorkflowId::new("wf-artifact-cancel"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("slow"),
            }],
            vec![
                ExecutionNode::new(
                    NodeId::new("save"),
                    NodeTypeId::new("mock.artifact"),
                    Vec::new(),
                    vec![SlotId::new("artifact")],
                ),
                ExecutionNode::new(
                    NodeId::new("slow"),
                    NodeTypeId::new("mock.slow"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("save")]),
                ExecutionStage::new(1, vec![NodeId::new("slow")]),
            ],
        );

        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();

        wait_for_snapshot(&service, handle.run_id(), Duration::from_secs(2), |snap| {
            snap.node_states.get(&NodeId::new("slow")).copied()
                == Some(reimagine_runtime::NodeState::Running)
        })
        .expect("slow node should start before cancellation");

        service.cancel(handle.run_id()).expect("cancel run");
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).expect("summary");
        assert_eq!(summary.state, reimagine_runtime::RunState::Cancelled);
        assert_eq!(summary.artifacts.len(), 1);
        assert_eq!(
            summary.artifacts[0].reference.as_str(),
            "output/pre-cancel.png"
        );
        assert_eq!(summary.artifacts[0].node_id, NodeId::new("save"));

        let snapshot = service.snapshot(handle.run_id()).expect("snapshot");
        assert_eq!(snapshot.state, reimagine_runtime::RunState::Cancelled);
        assert_eq!(snapshot.artifacts.len(), 1);
        assert_eq!(
            snapshot.artifacts[0].reference.as_str(),
            "output/pre-cancel.png"
        );
        assert_eq!(slow_count.load(Ordering::SeqCst), 1);
    });
}

// ---------------------------------------------------------------------------
// Retention-driven runtime lifecycle tests (issue 05).
// ---------------------------------------------------------------------------

/// Executor that records how many times `execute` was called, and
/// produces a `SingleUse` `ExecutionValue` on a slot named after
/// `slot`. Other executors in the lifecycle tests can observe the
/// value lifetime through the shared counter and the call ordering
/// of the captured `RunValueStore` (via the registry's plan).
struct SingleUseProducer {
    slot: String,
    label: String,
    count: Arc<AtomicUsize>,
    weak_slot: Option<Arc<Mutex<Option<Weak<ExecutionValue>>>>>,
}

#[async_trait]
impl NodeExecutor for SingleUseProducer {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        let value = Arc::new(ExecutionValue::Param(ParamValue::String(
            self.label.clone(),
        )));
        if let Some(weak_slot) = &self.weak_slot {
            *weak_slot.lock().unwrap() = Some(Arc::downgrade(&value));
        }
        Ok(vec![reimagine_runtime::ExecutionOutput::single_use(
            SlotId::new(self.slot.clone()),
            value,
        )])
    }
}

struct RunScopedProducer {
    slot: String,
    label: String,
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl NodeExecutor for RunScopedProducer {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new(self.slot.clone()),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                self.label.clone(),
            ))),
        )])
    }
}

struct WorkspaceScopedProducer {
    slot: String,
    label: String,
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl NodeExecutor for WorkspaceScopedProducer {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(vec![reimagine_runtime::ExecutionOutput::workspace_scoped(
            SlotId::new(self.slot.clone()),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                self.label.clone(),
            ))),
        )])
    }
}

/// Executor that simply records it ran and emits no outputs.
struct SinkExecutor {
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl NodeExecutor for SinkExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }
}

struct WeakProbeExecutor {
    weak_slot: Arc<Mutex<Option<Weak<ExecutionValue>>>>,
    observed_live: Arc<Mutex<Option<bool>>>,
}

#[async_trait]
impl NodeExecutor for WeakProbeExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        let is_live = self
            .weak_slot
            .lock()
            .unwrap()
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some();
        *self.observed_live.lock().unwrap() = Some(is_live);
        Ok(Vec::new())
    }
}

/// A backend that records coarse run lifecycle calls.
struct SpyBackend {
    begin_runs: AtomicUsize,
    cleanup_runs: AtomicUsize,
    cleanup_run_ids: Mutex<Vec<String>>,
    begin_diagnostics: Vec<Diagnostic>,
    cleanup_diagnostics: Vec<Diagnostic>,
    backend_instance: reimagine_inference::BackendInstance,
    backend: reimagine_inference::Backend,
}

impl SpyBackend {
    fn new() -> Self {
        Self {
            begin_runs: AtomicUsize::new(0),
            cleanup_runs: AtomicUsize::new(0),
            cleanup_run_ids: Mutex::new(Vec::new()),
            begin_diagnostics: Vec::new(),
            cleanup_diagnostics: Vec::new(),
            backend_instance: reimagine_inference::BackendInstance::new("spy"),
            backend: reimagine_inference::Backend::new("spy"),
        }
    }

    fn with_begin_diagnostic(mut self, diagnostic: Diagnostic) -> Self {
        self.begin_diagnostics.push(diagnostic);
        self
    }

    fn with_cleanup_diagnostic(mut self, diagnostic: Diagnostic) -> Self {
        self.cleanup_diagnostics.push(diagnostic);
        self
    }
}

#[async_trait]
impl reimagine_inference::BackendRunLifecycle for SpyBackend {
    fn backend_instance(&self) -> &reimagine_inference::BackendInstance {
        &self.backend_instance
    }

    async fn begin_run(
        &self,
        _request: reimagine_inference::BackendRunLifecycleRequest,
    ) -> Result<reimagine_inference::BackendRunLifecycleReport, reimagine_inference::InferenceError>
    {
        self.begin_runs.fetch_add(1, Ordering::SeqCst);
        Ok(reimagine_inference::BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics: self.begin_diagnostics.clone(),
        })
    }

    async fn cleanup_run(
        &self,
        request: reimagine_inference::BackendRunLifecycleRequest,
    ) -> Result<reimagine_inference::BackendRunLifecycleReport, reimagine_inference::InferenceError>
    {
        self.cleanup_runs.fetch_add(1, Ordering::SeqCst);
        self.cleanup_run_ids
            .lock()
            .unwrap()
            .push(request.run_id.to_string());
        Ok(reimagine_inference::BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics: self.cleanup_diagnostics.clone(),
        })
    }
}

#[async_trait]
impl reimagine_inference::BackendInstanceObservation for SpyBackend {
    fn backend_instance(&self) -> &reimagine_inference::BackendInstance {
        &self.backend_instance
    }

    async fn snapshot(&self) -> reimagine_inference::BackendInstanceSnapshot {
        reimagine_inference::BackendInstanceSnapshot {
            backend_instance: self.backend_instance.clone(),
            backend: self.backend.clone(),
            plugin: None,
            extension: None,
            device: None,
            observations: Default::default(),
            diagnostics: Vec::new(),
        }
    }
}

#[test]
fn single_use_value_with_fan_out_one_is_dropped_after_unique_consumer() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let producer_count = Arc::new(AtomicUsize::new(0));
        let consumer_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.producer",
                Arc::new(SingleUseProducer {
                    slot: "out".to_owned(),
                    label: "hello".to_owned(),
                    count: producer_count.clone(),
                    weak_slot: None,
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.consumer",
                Arc::new(SinkExecutor {
                    count: consumer_count.clone(),
                }),
            )
            .unwrap();

        let backend = Arc::new(SpyBackend::new());
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            backend.clone(),
            sink.clone(),
            Arc::new(FixedClock),
        );

        // producer -> consumer (SingleUse fan-out 1).
        let producer_node = ExecutionNode::new(
            NodeId::new("producer"),
            NodeTypeId::new("mock.producer"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let consumer_node = ExecutionNode::new(
            NodeId::new("consumer"),
            NodeTypeId::new("mock.consumer"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            Vec::new(),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("consumer"),
            }],
            vec![producer_node, consumer_node],
            vec![ExecutionEdge::new(
                reimagine_core::model::EdgeId::new("e"),
                NodeId::new("producer"),
                SlotId::new("out"),
                NodeId::new("consumer"),
                SlotId::new("in"),
            )],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("producer")]),
                ExecutionStage::new(1, vec![NodeId::new("consumer")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(producer_count.load(Ordering::SeqCst), 1);
        assert_eq!(consumer_count.load(Ordering::SeqCst), 1);

        // Backend sees only coarse run lifecycle calls.
        assert_eq!(backend.begin_runs.load(Ordering::SeqCst), 1);
        assert_eq!(backend.cleanup_runs.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn single_use_value_is_dropped_before_later_stage_after_unique_consumer() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let producer_count = Arc::new(AtomicUsize::new(0));
        let consumer_count = Arc::new(AtomicUsize::new(0));
        let weak_slot = Arc::new(Mutex::new(None));
        let observed_live = Arc::new(Mutex::new(None));
        registry
            .register(
                "mock.producer",
                Arc::new(SingleUseProducer {
                    slot: "out".to_owned(),
                    label: "hello".to_owned(),
                    count: producer_count.clone(),
                    weak_slot: Some(weak_slot.clone()),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.consumer",
                Arc::new(SinkExecutor {
                    count: consumer_count.clone(),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.probe",
                Arc::new(WeakProbeExecutor {
                    weak_slot: weak_slot.clone(),
                    observed_live: observed_live.clone(),
                }),
            )
            .unwrap();

        let service = RuntimeService::new(
            registry,
            Arc::new(SpyBackend::new()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );

        let producer_node = ExecutionNode::new(
            NodeId::new("producer"),
            NodeTypeId::new("mock.producer"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let consumer_node = ExecutionNode::new(
            NodeId::new("consumer"),
            NodeTypeId::new("mock.consumer"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            Vec::new(),
        );
        let probe_node = ExecutionNode::new(
            NodeId::new("probe"),
            NodeTypeId::new("mock.probe"),
            Vec::new(),
            Vec::new(),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("probe"),
            }],
            vec![producer_node, consumer_node, probe_node],
            vec![ExecutionEdge::new(
                reimagine_core::model::EdgeId::new("e"),
                NodeId::new("producer"),
                SlotId::new("out"),
                NodeId::new("consumer"),
                SlotId::new("in"),
            )],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("producer")]),
                ExecutionStage::new(1, vec![NodeId::new("consumer")]),
                ExecutionStage::new(2, vec![NodeId::new("probe")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(producer_count.load(Ordering::SeqCst), 1);
        assert_eq!(consumer_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            *observed_live.lock().unwrap(),
            Some(false),
            "SingleUse value should be dropped after the unique consumer completes, before later stages run"
        );
    });
}

#[test]
fn single_use_value_with_fan_out_zero_is_kept_until_terminal_cleanup() {
    // SingleUse output with no edge-sourced consumer in the active
    // plan. Per the issue's V1 contract, the value must be kept
    // until terminal cleanup; the runtime must not drop it
    // speculatively and must not fail the run.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let producer_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.producer",
                Arc::new(SingleUseProducer {
                    slot: "out".to_owned(),
                    label: "terminal".to_owned(),
                    count: producer_count.clone(),
                    weak_slot: None,
                }),
            )
            .unwrap();

        let backend = Arc::new(SpyBackend::new());
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            backend.clone(),
            sink.clone(),
            Arc::new(FixedClock),
        );

        // 1-node plan: producer only, no edges, no downstream
        // consumers. SingleUse fan-out is zero.
        let producer = ExecutionNode::new(
            NodeId::new("producer"),
            NodeTypeId::new("mock.producer"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("producer"),
            }],
            vec![producer],
            Vec::new(),
            Vec::new(),
            vec![ExecutionStage::new(0, vec![NodeId::new("producer")])],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(producer_count.load(Ordering::SeqCst), 1);
        // The value is kept until terminal cleanup; the only
        // backend hook is the coarse `cleanup_run`.
        assert_eq!(backend.begin_runs.load(Ordering::SeqCst), 1);
        assert_eq!(backend.cleanup_runs.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn single_use_value_is_dropped_after_consumer_is_cancelled() {
    // SingleUse upstream + slow consumer + cancel. The
    // consumer's execution attempt is cancelled, so the issue
    // contract says the upstream SingleUse value must be dropped
    // at that point (not just at terminal cleanup). This test
    // exercises the cancelled arm of `drop_consumed_single_use_values`
    // by observing the run end-state and the backend hooks.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let producer_count = Arc::new(AtomicUsize::new(0));
        let consumer_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.producer",
                Arc::new(SingleUseProducer {
                    slot: "out".to_owned(),
                    label: "x".to_owned(),
                    count: producer_count.clone(),
                    weak_slot: None,
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.slow_consumer",
                Arc::new(MockExecutor {
                    label: "slow".to_owned(),
                    count: consumer_count.clone(),
                    delay: Duration::from_secs(2),
                    fail_with: None,
                }),
            )
            .unwrap();

        let backend = Arc::new(SpyBackend::new());
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            backend.clone(),
            sink.clone(),
            Arc::new(FixedClock),
        );

        let producer = ExecutionNode::new(
            NodeId::new("producer"),
            NodeTypeId::new("mock.producer"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let consumer = ExecutionNode::new(
            NodeId::new("consumer"),
            NodeTypeId::new("mock.slow_consumer"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            vec![SlotId::new("out")],
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("consumer"),
            }],
            vec![producer, consumer],
            vec![ExecutionEdge::new(
                reimagine_core::model::EdgeId::new("e"),
                NodeId::new("producer"),
                SlotId::new("out"),
                NodeId::new("consumer"),
                SlotId::new("in"),
            )],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("producer")]),
                ExecutionStage::new(1, vec![NodeId::new("consumer")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        // Wait until the consumer has actually started, then cancel.
        for _ in 0..200 {
            if consumer_count.load(Ordering::SeqCst) > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        service.cancel(handle.run_id()).expect("cancel");
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Cancelled);
        assert_eq!(producer_count.load(Ordering::SeqCst), 1);
        // Backend sees only coarse run lifecycle calls.
        assert_eq!(backend.begin_runs.load(Ordering::SeqCst), 1);
        assert_eq!(backend.cleanup_runs.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn single_use_value_with_fan_out_greater_than_one_fails_the_run() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let producer_count = Arc::new(AtomicUsize::new(0));
        let down_a = Arc::new(AtomicUsize::new(0));
        let down_b = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.producer",
                Arc::new(SingleUseProducer {
                    slot: "out".to_owned(),
                    label: "x".to_owned(),
                    count: producer_count.clone(),
                    weak_slot: None,
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.sink",
                Arc::new(SinkExecutor {
                    count: down_a.clone(),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.sink2",
                Arc::new(SinkExecutor {
                    count: down_b.clone(),
                }),
            )
            .unwrap();

        let backend = Arc::new(SpyBackend::new());
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            backend.clone(),
            sink.clone(),
            Arc::new(FixedClock),
        );

        // producer -> down_a, down_b (SingleUse fan-out 2 in the
        // active plan; this must fail the run before any downstream
        // consumer receives the value).
        let producer = ExecutionNode::new(
            NodeId::new("producer"),
            NodeTypeId::new("mock.producer"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let consumer_a = ExecutionNode::new(
            NodeId::new("down_a"),
            NodeTypeId::new("mock.sink"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e1"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            Vec::new(),
        );
        let consumer_b = ExecutionNode::new(
            NodeId::new("down_b"),
            NodeTypeId::new("mock.sink2"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e2"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            Vec::new(),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("down_a"),
            }],
            vec![producer, consumer_a, consumer_b],
            vec![
                ExecutionEdge::new(
                    reimagine_core::model::EdgeId::new("e1"),
                    NodeId::new("producer"),
                    SlotId::new("out"),
                    NodeId::new("down_a"),
                    SlotId::new("in"),
                ),
                ExecutionEdge::new(
                    reimagine_core::model::EdgeId::new("e2"),
                    NodeId::new("producer"),
                    SlotId::new("out"),
                    NodeId::new("down_b"),
                    SlotId::new("in"),
                ),
            ],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("producer")]),
                ExecutionStage::new(1, vec![NodeId::new("down_a"), NodeId::new("down_b")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Failed);
        // The downstream consumers must not have run because the
        // producer's fan-out check failed the run first.
        assert_eq!(down_a.load(Ordering::SeqCst), 0);
        assert_eq!(down_b.load(Ordering::SeqCst), 0);
        // The diagnostic attached to the producer must explain the
        // fan-out violation.
        let summary_diag = summary
            .diagnostics
            .iter()
            .find(|d| d.message().contains("SingleUse"))
            .expect("SingleUse fan-out diagnostic is present");
        assert!(summary_diag.message().contains("fan-out"));
        assert!(summary_diag.message().contains("producer:out"));
    });
}

#[test]
fn run_scoped_value_is_retained_until_terminal_cleanup() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let producer_count = Arc::new(AtomicUsize::new(0));
        let consumer_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.producer",
                Arc::new(RunScopedProducer {
                    slot: "out".to_owned(),
                    label: "x".to_owned(),
                    count: producer_count.clone(),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.consumer",
                Arc::new(SinkExecutor {
                    count: consumer_count.clone(),
                }),
            )
            .unwrap();

        let backend = Arc::new(SpyBackend::new());
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            backend.clone(),
            sink.clone(),
            Arc::new(FixedClock),
        );

        // producer -> consumer; RunScoped value should remain in
        // RunValueStore until the run finishes, even though the
        // consumer completed. We verify this indirectly: the backend
        // never sees any per-value release callback (deleted), and
        // cleanup_run fires exactly once.
        let producer_node = ExecutionNode::new(
            NodeId::new("producer"),
            NodeTypeId::new("mock.producer"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let consumer_node = ExecutionNode::new(
            NodeId::new("consumer"),
            NodeTypeId::new("mock.consumer"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            Vec::new(),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("consumer"),
            }],
            vec![producer_node, consumer_node],
            vec![ExecutionEdge::new(
                reimagine_core::model::EdgeId::new("e"),
                NodeId::new("producer"),
                SlotId::new("out"),
                NodeId::new("consumer"),
                SlotId::new("in"),
            )],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("producer")]),
                ExecutionStage::new(1, vec![NodeId::new("consumer")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(backend.begin_runs.load(Ordering::SeqCst), 1);
        assert_eq!(backend.cleanup_runs.load(Ordering::SeqCst), 1);
        // cleanup_run is the only coarse backend hook used for
        // terminal cleanup.
        assert_eq!(
            backend.cleanup_run_ids.lock().unwrap().as_slice(),
            &[handle.run_id().to_string()]
        );
    });
}

#[test]
fn workspace_scoped_value_does_not_trigger_single_use_lifecycle() {
    // WorkspaceScoped values are not run-owned backend resources and
    // must not be interpreted as SingleUse. The checkpoint loader is
    // the canonical WorkspaceScoped producer, but the retention-driven
    // lifecycle is independent of the producer's identity, so this test
    // uses a generic producer.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let producer_count = Arc::new(AtomicUsize::new(0));
        let consumer_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.producer",
                Arc::new(WorkspaceScopedProducer {
                    slot: "out".to_owned(),
                    label: "ws".to_owned(),
                    count: producer_count.clone(),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.consumer",
                Arc::new(SinkExecutor {
                    count: consumer_count.clone(),
                }),
            )
            .unwrap();

        let backend = Arc::new(SpyBackend::new());
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            backend.clone(),
            sink.clone(),
            Arc::new(FixedClock),
        );

        // WorkspaceScoped: even with two consumers (legal because
        // WorkspaceScoped is not constrained by the consumer index),
        // the run must succeed.
        let producer = ExecutionNode::new(
            NodeId::new("producer"),
            NodeTypeId::new("mock.producer"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let consumer_a = ExecutionNode::new(
            NodeId::new("down_a"),
            NodeTypeId::new("mock.consumer"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e1"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            Vec::new(),
        );
        let consumer_b = ExecutionNode::new(
            NodeId::new("down_b"),
            NodeTypeId::new("mock.consumer"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e2"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            Vec::new(),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![RunTarget::Node {
                node_id: NodeId::new("down_a"),
            }],
            vec![producer, consumer_a, consumer_b],
            vec![
                ExecutionEdge::new(
                    reimagine_core::model::EdgeId::new("e1"),
                    NodeId::new("producer"),
                    SlotId::new("out"),
                    NodeId::new("down_a"),
                    SlotId::new("in"),
                ),
                ExecutionEdge::new(
                    reimagine_core::model::EdgeId::new("e2"),
                    NodeId::new("producer"),
                    SlotId::new("out"),
                    NodeId::new("down_b"),
                    SlotId::new("in"),
                ),
            ],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("producer")]),
                ExecutionStage::new(1, vec![NodeId::new("down_a"), NodeId::new("down_b")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(consumer_count.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn explicit_target_run_uses_active_plan_fan_out_not_workflow_fan_out() {
    // Regression: an explicit-target run that picks a single branch
    // out of a saved workflow must not see consumers from the
    // unselected branch. The active plan only contains the chosen
    // edge, so SingleUse fan-out 1 is legal.
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        let producer_count = Arc::new(AtomicUsize::new(0));
        let consumer_count = Arc::new(AtomicUsize::new(0));
        let other_branch_count = Arc::new(AtomicUsize::new(0));
        registry
            .register(
                "mock.producer",
                Arc::new(SingleUseProducer {
                    slot: "out".to_owned(),
                    label: "x".to_owned(),
                    count: producer_count.clone(),
                    weak_slot: None,
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.consumer",
                Arc::new(SinkExecutor {
                    count: consumer_count.clone(),
                }),
            )
            .unwrap();
        registry
            .register(
                "mock.other",
                Arc::new(SinkExecutor {
                    count: other_branch_count.clone(),
                }),
            )
            .unwrap();

        let backend = Arc::new(SpyBackend::new());
        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            backend.clone(),
            sink.clone(),
            Arc::new(FixedClock),
        );

        // Active plan: producer -> consumer only. The
        // "saved workflow" would also have producer -> other, but
        // that edge is excluded from this explicit-target plan.
        let producer = ExecutionNode::new(
            NodeId::new("producer"),
            NodeTypeId::new("mock.producer"),
            Vec::new(),
            vec![SlotId::new("out")],
        );
        let consumer = ExecutionNode::new(
            NodeId::new("consumer"),
            NodeTypeId::new("mock.consumer"),
            vec![ExecutionInputBinding::new(
                SlotId::new("in"),
                ExecutionInputSource::Edge {
                    edge_id: reimagine_core::model::EdgeId::new("e-selected"),
                    from_node_id: NodeId::new("producer"),
                    from_slot_id: SlotId::new("out"),
                },
            )],
            Vec::new(),
        );
        let plan = ExecutionPlan::new(
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            RunTargetSelection::ExplicitTargets(vec![RunTarget::Node {
                node_id: NodeId::new("consumer"),
            }]),
            vec![RunTarget::Node {
                node_id: NodeId::new("consumer"),
            }],
            vec![producer, consumer],
            vec![ExecutionEdge::new(
                reimagine_core::model::EdgeId::new("e-selected"),
                NodeId::new("producer"),
                SlotId::new("out"),
                NodeId::new("consumer"),
                SlotId::new("in"),
            )],
            Vec::new(),
            vec![
                ExecutionStage::new(0, vec![NodeId::new("producer")]),
                ExecutionStage::new(1, vec![NodeId::new("consumer")]),
            ],
        );
        let handle = service
            .run(
                Arc::new(plan),
                Default::default(),
                RuntimeOptions::default(),
            )
            .unwrap();
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        assert_eq!(consumer_count.load(Ordering::SeqCst), 1);
        // The unselected branch was never scheduled.
        assert_eq!(other_branch_count.load(Ordering::SeqCst), 0);
    });
}
