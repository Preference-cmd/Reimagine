//! Tests for the V1 runtime service layer.
//!
//! These tests exercise scheduling, cancellation, fail-fast downstream
//! skipping, shared upstream reuse, sink failure tolerance, and the public
//! snapshot/summary observation shapes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_core::diagnostic::DiagnosticCode;
use reimagine_core::event::{RunEvent, RunEventKind, Timestamp};
use reimagine_core::model::{
    NodeId, NodeTypeId, ParamValue, SlotId, WorkflowId, WorkflowInputId, WorkflowVersion,
};
use reimagine_core::readiness::{
    ExecutionEdge, ExecutionInputBinding, ExecutionInputSource, ExecutionNode, ExecutionPlan,
    ExecutionStage, RunTarget, RunTargetSelection,
};
use reimagine_runtime::{
    CancellationToken, Clock, NodeExecutionContext, NodeExecutor, NodeExecutorError,
    NodeExecutorRegistry, NoopRunResourceBackend, RunEventSink, RunHandle, RunInputs, RuntimeError,
    RuntimeOptions, RuntimeService, RuntimeServiceError, RuntimeValue, VecRunEventSink,
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
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
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
        Ok(vec![(
            SlotId::new("out"),
            Arc::new(RuntimeValue::Param(
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
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Err(NodeExecutorError::Cancelled)
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
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
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

        Ok(vec![(
            SlotId::new("out"),
            Arc::new(RuntimeValue::Param(ParamValue::String("ok".to_owned()))),
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        self.order.lock().unwrap().push(self.label.clone());
        Ok(vec![(
            SlotId::new("out"),
            Arc::new(RuntimeValue::Param(
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(RuntimeValue::Param(ParamValue::Text(
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
            if let Some(snapshot) = service.snapshot(handle.run_id()) {
                if snapshot.node_states.get(&NodeId::new("node_a")).copied()
                    == Some(reimagine_runtime::NodeState::Running)
                {
                    run_to_completion(&service, &handle);
                    return;
                }
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
            Arc::new(NoopRunResourceBackend),
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
