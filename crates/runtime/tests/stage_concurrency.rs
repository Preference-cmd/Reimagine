//! Stage scheduler concurrency tests (B5-2 / BE-23).
//!
//! These tests exercise the stage reducer with `max_stage_concurrency > 1`:
//!
//! - N independent nodes in one stage must run concurrently (barrier
//!   based, no wall-clock timing assertions).
//! - Concurrent `RunValueStore` writes (producers) and shared reads
//!   (multiple consumers of the same upstream keys) must stay consistent.
//! - Cancelling a run mid-stage must cancel in-flight nodes and never
//!   hang.
//! - A node deadline expiry under concurrency must fail only the slow
//!   node while the rest of the stage completes.
//! - A sibling failure must cancel the remaining in-flight nodes.
//!
//! All synchronization is via atomics + poll-based waits (the same
//! pattern as `runtime_service.rs`); assertions only ever check eventual
//! state, so the tests are deterministic under any scheduler and stable
//! under repeated `--test-threads` runs.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_core::event::{RunEventKind, Timestamp};
use reimagine_core::model::{
    EdgeId, NodeId, NodeTypeId, ParamValue, SlotId, WorkflowId, WorkflowVersion,
};
use reimagine_core::readiness::{
    ExecutionEdge, ExecutionInputBinding, ExecutionInputSource, ExecutionNode, ExecutionPlan,
    ExecutionStage, RunTarget, RunTargetSelection,
};
use reimagine_runtime::{
    Clock, ExecutionValue, NodeExecutionContext, NodeExecutor, NodeExecutorError,
    NodeExecutorRegistry, NoopBackendInstanceRuntimeHooks, RunHandle, RunInputs, RuntimeOptions,
    RuntimeService, VecRunEventSink,
};

/// A clock that always returns the same string timestamp.
#[derive(Debug, Default, Clone, Copy)]
struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> Timestamp {
        Timestamp::new("2026-06-10T00:00:00Z")
    }
}

/// Build an execution plan with the given independent node ids all in
/// stage 0 (no edges, no input bindings).
fn independent_nodes_plan(node_ids: &[&str], type_id: &str) -> ExecutionPlan {
    ExecutionPlan::new(
        WorkflowId::new("workflow-concurrency"),
        WorkflowVersion::new(1),
        RunTargetSelection::AllDefaultTargets,
        node_ids
            .iter()
            .map(|node_id| RunTarget::Node {
                node_id: NodeId::new(*node_id),
            })
            .collect(),
        node_ids
            .iter()
            .map(|node_id| {
                ExecutionNode::new(
                    NodeId::new(*node_id),
                    NodeTypeId::new(type_id),
                    Vec::new(),
                    vec![SlotId::new("out")],
                )
            })
            .collect(),
        Vec::new(),
        Vec::new(),
        vec![ExecutionStage::new(
            0,
            node_ids
                .iter()
                .map(|node_id| NodeId::new(*node_id))
                .collect(),
        )],
    )
}

/// Executor that waits until every sibling has entered, then completes.
///
/// `max_seen` tracks the peak number of concurrently-entered nodes —
/// the core concurrency assertion.
struct BarrierExecutor {
    total: usize,
    entered: Arc<AtomicUsize>,
    max_seen: Arc<AtomicUsize>,
    label: String,
}

#[async_trait]
impl NodeExecutor for BarrierExecutor {
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
        while self.entered.load(Ordering::SeqCst) < self.total {
            if context.cancellation().is_cancelled() {
                return Err(NodeExecutorError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                self.label.clone(),
            ))),
        )])
    }
}

/// Executor that enters (counted) and blocks until cancelled or released.
struct HoldUntilCancelledExecutor {
    entered: Arc<AtomicUsize>,
    release: Arc<AtomicBool>,
}

#[async_trait]
impl NodeExecutor for HoldUntilCancelledExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        while !self.release.load(Ordering::SeqCst) {
            if context.cancellation().is_cancelled() {
                return Err(NodeExecutorError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                "released".to_owned(),
            ))),
        )])
    }
}

/// Executor that waits for an external gate before completing instantly.
struct GateExecutor {
    entered: Arc<AtomicUsize>,
    gate: Arc<AtomicBool>,
    label: String,
}

#[async_trait]
impl NodeExecutor for GateExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        while !self.gate.load(Ordering::SeqCst) {
            if context.cancellation().is_cancelled() {
                return Err(NodeExecutorError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                self.label.clone(),
            ))),
        )])
    }
}

/// Producer that writes one distinct `out` value into the value store.
struct ValueProducer {
    label: String,
}

#[async_trait]
impl NodeExecutor for ValueProducer {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                self.label.clone(),
            ))),
        )])
    }
}

/// Consumer that verifies every expected input slot carries the expected
/// value — proving concurrent reads see consistent store contents.
struct SharedReadConsumer {
    expected: Vec<(String, String)>,
    label: String,
    failures: Arc<std::sync::Mutex<Vec<String>>>,
}

#[async_trait]
impl NodeExecutor for SharedReadConsumer {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        for (slot, expected) in &self.expected {
            let actual = context
                .inputs()
                .get(&SlotId::new(slot))
                .and_then(|value| value.as_param())
                .ok_or_else(|| NodeExecutorError::MissingInput {
                    slot_id: slot.clone(),
                })?;
            if actual != &ParamValue::String(expected.clone()) {
                self.failures
                    .lock()
                    .expect("failures poisoned")
                    .push(format!(
                        "node {}: slot {} expected {expected}, got {actual:?}",
                        self.label, slot
                    ));
            }
        }
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                self.label.clone(),
            ))),
        )])
    }
}

/// Executor that blocks cooperatively (checks cancellation) so a deadline
/// expiry can fail it without leaving a stuck task behind.
struct CooperativeBlockingExecutor {
    started: Arc<AtomicUsize>,
}

#[async_trait]
impl NodeExecutor for CooperativeBlockingExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        self.started.fetch_add(1, Ordering::SeqCst);
        loop {
            if context.cancellation().is_cancelled() {
                return Err(NodeExecutorError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }
}

/// Executor that fails the node immediately.
struct InstantFailExecutor;

#[async_trait]
impl NodeExecutor for InstantFailExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        Err(NodeExecutorError::Failed {
            message: "scripted node failure".to_owned(),
        })
    }
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

fn run_to_completion(service: &RuntimeService, handle: &RunHandle) {
    let run_id = handle.run_id().clone();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(summary) = service.summary(&run_id)
            && summary.state.is_terminal()
        {
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

fn service_with(registry: NodeExecutorRegistry) -> RuntimeService {
    RuntimeService::new(
        registry,
        Arc::new(NoopBackendInstanceRuntimeHooks::default()),
        Arc::new(VecRunEventSink::new()),
        Arc::new(FixedClock),
    )
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn stage_runs_independent_nodes_concurrently() {
    let rt = test_runtime();
    rt.block_on(async {
        let entered = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "test.concurrent",
                Arc::new(BarrierExecutor {
                    total: 4,
                    entered: entered.clone(),
                    max_seen: max_seen.clone(),
                    label: "barrier".to_owned(),
                }),
            )
            .expect("register executor");
        let service = service_with(registry);
        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(4);
        let handle = service
            .run(
                Arc::new(independent_nodes_plan(
                    &["a", "b", "c", "d"],
                    "test.concurrent",
                )),
                RunInputs::new(),
                options,
            )
            .expect("start run");

        wait_for_condition(Duration::from_secs(5), || {
            max_seen.load(Ordering::SeqCst) >= 4
        });
        run_to_completion(&service, &handle);

        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            4,
            "all four nodes must have been in flight together"
        );
        let snapshot = service.snapshot(handle.run_id()).unwrap();
        for node_id in ["a", "b", "c", "d"] {
            assert_eq!(
                snapshot.node_states.get(&NodeId::new(node_id)),
                Some(&reimagine_runtime::NodeState::Completed),
                "node {node_id} must complete"
            );
        }
        assert_eq!(
            service.summary(handle.run_id()).unwrap().state,
            reimagine_runtime::RunState::Completed
        );
    });
}

#[test]
fn value_store_concurrent_producers_and_shared_readers_stay_consistent() {
    let rt = test_runtime();
    rt.block_on(async {
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "test.producer",
                Arc::new(ValueProducer { label: "p0".into() }),
            )
            .expect("register p0");
        // Producers differ by label; the registry maps type id -> executor,
        // so give every node its own type id.
        for (index, label) in ["p1", "p2", "p3"].iter().enumerate() {
            registry
                .register(
                    format!("test.producer{index}"),
                    Arc::new(ValueProducer {
                        label: (*label).to_owned(),
                    }),
                )
                .expect("register producer");
        }

        // Stage 0: producers p0..p3 (distinct keys). Stage 1: two
        // consumers, each reading all four producer outputs concurrently.
        let plan = ExecutionPlan::new(
            WorkflowId::new("workflow-store"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![
                RunTarget::Node {
                    node_id: NodeId::new("p0"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("p1"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("p2"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("p3"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("c0"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("c1"),
                },
            ],
            vec![
                ExecutionNode::new(
                    NodeId::new("p0"),
                    NodeTypeId::new("test.producer"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("p1"),
                    NodeTypeId::new("test.producer0"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("p2"),
                    NodeTypeId::new("test.producer1"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("p3"),
                    NodeTypeId::new("test.producer2"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("c0"),
                    NodeTypeId::new("test.consumer"),
                    vec![
                        ExecutionInputBinding::new(
                            SlotId::new("in0"),
                            ExecutionInputSource::Edge {
                                edge_id: EdgeId::new("e0"),
                                from_node_id: NodeId::new("p0"),
                                from_slot_id: SlotId::new("out"),
                            },
                        ),
                        ExecutionInputBinding::new(
                            SlotId::new("in1"),
                            ExecutionInputSource::Edge {
                                edge_id: EdgeId::new("e1"),
                                from_node_id: NodeId::new("p1"),
                                from_slot_id: SlotId::new("out"),
                            },
                        ),
                        ExecutionInputBinding::new(
                            SlotId::new("in2"),
                            ExecutionInputSource::Edge {
                                edge_id: EdgeId::new("e2"),
                                from_node_id: NodeId::new("p2"),
                                from_slot_id: SlotId::new("out"),
                            },
                        ),
                        ExecutionInputBinding::new(
                            SlotId::new("in3"),
                            ExecutionInputSource::Edge {
                                edge_id: EdgeId::new("e3"),
                                from_node_id: NodeId::new("p3"),
                                from_slot_id: SlotId::new("out"),
                            },
                        ),
                    ],
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("c1"),
                    NodeTypeId::new("test.consumer"),
                    vec![
                        ExecutionInputBinding::new(
                            SlotId::new("in0"),
                            ExecutionInputSource::Edge {
                                edge_id: EdgeId::new("e4"),
                                from_node_id: NodeId::new("p0"),
                                from_slot_id: SlotId::new("out"),
                            },
                        ),
                        ExecutionInputBinding::new(
                            SlotId::new("in1"),
                            ExecutionInputSource::Edge {
                                edge_id: EdgeId::new("e5"),
                                from_node_id: NodeId::new("p1"),
                                from_slot_id: SlotId::new("out"),
                            },
                        ),
                        ExecutionInputBinding::new(
                            SlotId::new("in2"),
                            ExecutionInputSource::Edge {
                                edge_id: EdgeId::new("e6"),
                                from_node_id: NodeId::new("p2"),
                                from_slot_id: SlotId::new("out"),
                            },
                        ),
                        ExecutionInputBinding::new(
                            SlotId::new("in3"),
                            ExecutionInputSource::Edge {
                                edge_id: EdgeId::new("e7"),
                                from_node_id: NodeId::new("p3"),
                                from_slot_id: SlotId::new("out"),
                            },
                        ),
                    ],
                    vec![SlotId::new("out")],
                ),
            ],
            vec![
                ExecutionEdge::new(
                    EdgeId::new("e0"),
                    NodeId::new("p0"),
                    SlotId::new("out"),
                    NodeId::new("c0"),
                    SlotId::new("in0"),
                ),
                ExecutionEdge::new(
                    EdgeId::new("e1"),
                    NodeId::new("p1"),
                    SlotId::new("out"),
                    NodeId::new("c0"),
                    SlotId::new("in1"),
                ),
                ExecutionEdge::new(
                    EdgeId::new("e2"),
                    NodeId::new("p2"),
                    SlotId::new("out"),
                    NodeId::new("c0"),
                    SlotId::new("in2"),
                ),
                ExecutionEdge::new(
                    EdgeId::new("e3"),
                    NodeId::new("p3"),
                    SlotId::new("out"),
                    NodeId::new("c0"),
                    SlotId::new("in3"),
                ),
                ExecutionEdge::new(
                    EdgeId::new("e4"),
                    NodeId::new("p0"),
                    SlotId::new("out"),
                    NodeId::new("c1"),
                    SlotId::new("in0"),
                ),
                ExecutionEdge::new(
                    EdgeId::new("e5"),
                    NodeId::new("p1"),
                    SlotId::new("out"),
                    NodeId::new("c1"),
                    SlotId::new("in1"),
                ),
                ExecutionEdge::new(
                    EdgeId::new("e6"),
                    NodeId::new("p2"),
                    SlotId::new("out"),
                    NodeId::new("c1"),
                    SlotId::new("in2"),
                ),
                ExecutionEdge::new(
                    EdgeId::new("e7"),
                    NodeId::new("p3"),
                    SlotId::new("out"),
                    NodeId::new("c1"),
                    SlotId::new("in3"),
                ),
            ],
            Vec::new(),
            vec![
                ExecutionStage::new(
                    0,
                    vec![
                        NodeId::new("p0"),
                        NodeId::new("p1"),
                        NodeId::new("p2"),
                        NodeId::new("p3"),
                    ],
                ),
                ExecutionStage::new(1, vec![NodeId::new("c0"), NodeId::new("c1")]),
            ],
        );

        let failures = Arc::new(std::sync::Mutex::new(Vec::new()));
        registry
            .register(
                "test.consumer",
                Arc::new(SharedReadConsumer {
                    expected: vec![
                        ("in0".to_owned(), "p0".to_owned()),
                        ("in1".to_owned(), "p1".to_owned()),
                        ("in2".to_owned(), "p2".to_owned()),
                        ("in3".to_owned(), "p3".to_owned()),
                    ],
                    label: "consumer".to_owned(),
                    failures: failures.clone(),
                }),
            )
            .expect("register consumer");
        let service = service_with(registry);
        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(4);
        let handle = service
            .run(Arc::new(plan), RunInputs::new(), options)
            .expect("start run");
        run_to_completion(&service, &handle);

        assert_eq!(
            service.summary(handle.run_id()).unwrap().state,
            reimagine_runtime::RunState::Completed
        );
        assert!(
            failures.lock().expect("failures poisoned").is_empty(),
            "shared reads saw inconsistent values: {:?}",
            failures.lock().expect("failures poisoned")
        );
        let snapshot = service.snapshot(handle.run_id()).unwrap();
        for node_id in ["p0", "p1", "p2", "p3", "c0", "c1"] {
            assert_eq!(
                snapshot.node_states.get(&NodeId::new(node_id)),
                Some(&reimagine_runtime::NodeState::Completed),
                "node {node_id} must complete"
            );
        }
    });
}

#[test]
fn cancel_during_concurrent_execution_cancels_in_flight_without_hang() {
    let rt = test_runtime();
    rt.block_on(async {
        let entered = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "test.hold",
                Arc::new(HoldUntilCancelledExecutor {
                    entered: entered.clone(),
                    release: release.clone(),
                }),
            )
            .expect("register hold executor");
        registry
            .register(
                "test.gated",
                Arc::new(GateExecutor {
                    entered: entered.clone(),
                    gate: gate.clone(),
                    label: "fast".to_owned(),
                }),
            )
            .expect("register gated executor");
        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(4);

        // Plan: `hold_a` and `hold_b` block until cancelled; `fast_c` and
        // `fast_d` wait for the gate then complete.
        let plan = ExecutionPlan::new(
            WorkflowId::new("workflow-cancel"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![
                RunTarget::Node {
                    node_id: NodeId::new("hold_a"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("hold_b"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("fast_c"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("fast_d"),
                },
            ],
            vec![
                ExecutionNode::new(
                    NodeId::new("hold_a"),
                    NodeTypeId::new("test.hold"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("hold_b"),
                    NodeTypeId::new("test.hold"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("fast_c"),
                    NodeTypeId::new("test.gated"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("fast_d"),
                    NodeTypeId::new("test.gated"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![ExecutionStage::new(
                0,
                vec![
                    NodeId::new("hold_a"),
                    NodeId::new("hold_b"),
                    NodeId::new("fast_c"),
                    NodeId::new("fast_d"),
                ],
            )],
        );

        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );
        let handle = service
            .run(Arc::new(plan), RunInputs::new(), options)
            .expect("start run");

        // All four nodes enter; the fast ones are gated, the hold ones
        // block on cancellation.
        wait_for_condition(Duration::from_secs(5), || {
            entered.load(Ordering::SeqCst) >= 4
        });
        // Let the fast nodes complete before the cancel lands, so the
        // outcome is deterministic: fast = Completed, hold = Cancelled.
        gate.store(true, Ordering::SeqCst);
        wait_for_condition(Duration::from_secs(5), || {
            service.snapshot(handle.run_id()).is_some_and(|snapshot| {
                snapshot
                    .node_states
                    .get(&NodeId::new("fast_c"))
                    .is_some_and(|state| state.is_terminal())
                    && snapshot
                        .node_states
                        .get(&NodeId::new("fast_d"))
                        .is_some_and(|state| state.is_terminal())
            })
        });
        service.cancel(handle.run_id()).expect("cancel run");

        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Cancelled);
        let snapshot = service.snapshot(handle.run_id()).unwrap();
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("hold_a")),
            Some(&reimagine_runtime::NodeState::Cancelled)
        );
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("hold_b")),
            Some(&reimagine_runtime::NodeState::Cancelled)
        );
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("fast_c")),
            Some(&reimagine_runtime::NodeState::Completed)
        );
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("fast_d")),
            Some(&reimagine_runtime::NodeState::Completed)
        );
        let kinds: Vec<RunEventKind> = sink.events().iter().map(|e| e.kind()).collect();
        assert!(kinds.contains(&RunEventKind::RunCancelled));
        assert!(kinds.contains(&RunEventKind::NodeCancelled));
    });
}

#[test]
fn deadline_expiry_under_concurrency_fails_only_the_slow_node() {
    let rt = test_runtime();
    rt.block_on(async {
        let started = Arc::new(AtomicUsize::new(0));
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "test.blocking",
                Arc::new(CooperativeBlockingExecutor {
                    started: started.clone(),
                }),
            )
            .expect("register blocking executor");
        registry
            .register(
                "test.fast",
                Arc::new(ValueProducer {
                    label: "fast".to_owned(),
                }),
            )
            .expect("register fast executor");
        let service = service_with(registry);
        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(2);
        options.default_node_timeout = Some(Duration::from_millis(150));

        // `a` uses the blocking executor; b/c/d use the fast one.
        let plan = ExecutionPlan::new(
            WorkflowId::new("workflow-deadline"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![
                RunTarget::Node {
                    node_id: NodeId::new("a"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("b"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("c"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("d"),
                },
            ],
            vec![
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("test.blocking"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("test.fast"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("c"),
                    NodeTypeId::new("test.fast"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("d"),
                    NodeTypeId::new("test.fast"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![ExecutionStage::new(
                0,
                vec![
                    NodeId::new("a"),
                    NodeId::new("b"),
                    NodeId::new("c"),
                    NodeId::new("d"),
                ],
            )],
        );
        let handle = service
            .run(Arc::new(plan), RunInputs::new(), options)
            .expect("start run");
        run_to_completion(&service, &handle);

        assert!(started.load(Ordering::SeqCst) >= 1);
        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Completed);
        let snapshot = service.snapshot(handle.run_id()).unwrap();
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("a")),
            Some(&reimagine_runtime::NodeState::Failed),
            "slow node must fail via deadline expiry"
        );
        for node_id in ["b", "c", "d"] {
            assert_eq!(
                snapshot.node_states.get(&NodeId::new(node_id)),
                Some(&reimagine_runtime::NodeState::Completed),
                "fast node {node_id} must complete despite the sibling timeout"
            );
        }
        let timeout_diag = snapshot
            .diagnostics
            .iter()
            .find(|d| d.message().contains("deadline"))
            .expect("deadline diagnostic must be emitted");
        assert!(
            timeout_diag.message().contains("node a"),
            "{}",
            timeout_diag.message()
        );
    });
}

#[test]
fn sibling_failure_under_concurrency_cancels_in_flight_nodes() {
    let rt = test_runtime();
    rt.block_on(async {
        let entered = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "test.hold",
                Arc::new(HoldUntilCancelledExecutor {
                    entered: entered.clone(),
                    release: release.clone(),
                }),
            )
            .expect("register hold executor");
        registry
            .register("test.fail", Arc::new(InstantFailExecutor))
            .expect("register fail executor");
        let service = service_with(registry);

        // Admission order is b, c, then a: the two hold nodes are in
        // flight before the failing node is admitted, so the stage
        // failure cancels them mid-flight (not merely skips them).
        let plan = ExecutionPlan::new(
            WorkflowId::new("workflow-fail"),
            WorkflowVersion::new(1),
            RunTargetSelection::AllDefaultTargets,
            vec![
                RunTarget::Node {
                    node_id: NodeId::new("b"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("c"),
                },
                RunTarget::Node {
                    node_id: NodeId::new("a"),
                },
            ],
            vec![
                ExecutionNode::new(
                    NodeId::new("b"),
                    NodeTypeId::new("test.hold"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("c"),
                    NodeTypeId::new("test.hold"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
                ExecutionNode::new(
                    NodeId::new("a"),
                    NodeTypeId::new("test.fail"),
                    Vec::new(),
                    vec![SlotId::new("out")],
                ),
            ],
            Vec::new(),
            Vec::new(),
            vec![ExecutionStage::new(
                0,
                vec![NodeId::new("b"), NodeId::new("c"), NodeId::new("a")],
            )],
        );
        let mut options = RuntimeOptions::default();
        options.max_stage_concurrency = Some(3);
        let handle = service
            .run(Arc::new(plan), RunInputs::new(), options)
            .expect("start run");
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Failed);
        let snapshot = service.snapshot(handle.run_id()).unwrap();
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("a")),
            Some(&reimagine_runtime::NodeState::Failed)
        );
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("b")),
            Some(&reimagine_runtime::NodeState::Cancelled),
            "in-flight sibling b must be cancelled by the sibling failure"
        );
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("c")),
            Some(&reimagine_runtime::NodeState::Cancelled),
            "in-flight sibling c must be cancelled by the sibling failure"
        );
    });
}
