//! Runtime-level integration tests for [`ScriptedBackend`].
//!
//! The runtime test harness in `runtime_service.rs` is owned by the
//! concurrent tasks working on cancellation/deadline semantics, so this
//! suite lives in its own file and only exercises the seam between the
//! stage scheduler and a scripted inference backend:
//!
//! - a scripted error at the second node of a two-node run fails the run
//!   with the executor-mapped message (error propagation through the
//!   stage reducer);
//! - a held scripted step cancelled mid-flight marks the run cancelled
//!   (cancellation during a slow backend operation, end to end).
//!
//! The nodes are thin executors that forward to
//! `ScriptedBackend::text_encode_with_invocation`; the plan wiring
//! (readiness → stages → scheduler) is the real runtime path.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    ModelId, NodeId, NodeTypeId, ParamValue, SlotId, WorkflowId, WorkflowVersion,
};
use reimagine_core::readiness::{
    ExecutionNode, ExecutionPlan, ExecutionStage, RunTarget, RunTargetSelection,
};
use reimagine_inference::{
    Backend, ExecutionConditioning, ExecutionValue, InferenceBackend, InferenceError,
    ScriptedBackend, TextEncodeRequest, TextEncodeResponse, into_executor_error,
};
use reimagine_runtime::{
    Clock, NodeExecutionContext, NodeExecutor, NodeExecutorError, NodeExecutorRegistry,
    NoopBackendInstanceRuntimeHooks, RunHandle, RuntimeOptions, RuntimeService, VecRunEventSink,
};

/// A clock that always returns the same string timestamp.
#[derive(Debug, Default, Clone, Copy)]
struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> Timestamp {
        Timestamp::new("2026-06-10T00:00:00Z")
    }
}

/// Thin node executor that forwards to a [`ScriptedBackend`]'s
/// `text.encode` capability, mapping backend errors with
/// [`into_executor_error`] exactly like the built-in
/// `builtin.clip_text_encode` executor does.
struct ScriptedTextNode {
    backend: Arc<ScriptedBackend>,
}

#[async_trait]
impl NodeExecutor for ScriptedTextNode {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<reimagine_runtime::ExecutionOutput>, NodeExecutorError> {
        let request = TextEncodeRequest::new(
            reimagine_inference::RuntimeClipHandle::new(
                ModelId::new("sdxl-base-1.0"),
                Backend::new("scripted"),
                "clip-1",
            ),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                "a prompt".to_owned(),
            ))),
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        let invocation = context.inference_invocation();
        let response = self
            .backend
            .text_encode_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;
        Ok(vec![reimagine_runtime::ExecutionOutput::run_scoped(
            SlotId::new("out"),
            Arc::new(ExecutionValue::Conditioning(response.into_conditioning())),
        )])
    }
}

fn n_node_plan(node_ids: &[&str]) -> ExecutionPlan {
    ExecutionPlan::new(
        WorkflowId::new("workflow-scripted"),
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
                    NodeTypeId::new("test.scripted_text"),
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

fn test_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn scripted_error_at_second_node_fails_the_run_with_mapped_message() {
    let rt = test_runtime();
    rt.block_on(async {
        let backend = Arc::new(ScriptedBackend::new("scripted").text_encode(vec![
            Ok(TextEncodeResponse::new(ExecutionConditioning::new(
                reimagine_inference::BackendTensorHandle::new(
                    Backend::new("scripted"),
                    reimagine_inference::BackendPayloadKey::new("emb-1"),
                    reimagine_core::model::TensorDType::F32,
                    reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                    "cpu",
                ),
                reimagine_inference::ConditioningMetadata::new(64, 64),
            ))),
            Err(InferenceError::TokenizationFailed {
                message: "step 2 scripted failure".to_owned(),
            }),
        ]));
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "test.scripted_text",
                Arc::new(ScriptedTextNode {
                    backend: backend.clone(),
                }),
            )
            .expect("register scripted text node");

        let sink = Arc::new(VecRunEventSink::new());
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            sink.clone(),
            Arc::new(FixedClock),
        );
        let handle = service
            .run(
                Arc::new(n_node_plan(&["a", "b"])),
                Default::default(),
                RuntimeOptions::default(),
            )
            .expect("start run");
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(summary.state, reimagine_runtime::RunState::Failed);
        let snapshot = service.snapshot(handle.run_id()).unwrap();
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("a")),
            Some(&reimagine_runtime::NodeState::Completed)
        );
        match snapshot.node_states.get(&NodeId::new("b")) {
            Some(reimagine_runtime::NodeState::Failed) => {}
            other => panic!("expected b Failed, got {other:?}"),
        }
        let failure_diag = snapshot
            .diagnostics
            .iter()
            .find(|d| d.message().contains("step 2 scripted failure"))
            .expect("scripted failure message must propagate into the run diagnostics");
        assert_eq!(
            failure_diag.primary().id(),
            Some("b"),
            "diagnostic must be attributed to node b"
        );
        assert_eq!(backend.total_calls(), 2);
    });
}

#[test]
fn held_scripted_step_cancelled_mid_flight_marks_run_cancelled() {
    let rt = test_runtime();
    rt.block_on(async {
        let (backend, hold) = ScriptedBackend::new("scripted")
            .text_encode(vec![Ok(TextEncodeResponse::new(
                ExecutionConditioning::new(
                    reimagine_inference::BackendTensorHandle::new(
                        Backend::new("scripted"),
                        reimagine_inference::BackendPayloadKey::new("emb-1"),
                        reimagine_core::model::TensorDType::F32,
                        reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                        "cpu",
                    ),
                    reimagine_inference::ConditioningMetadata::new(64, 64),
                ),
            ))])
            .with_hold();
        let backend = Arc::new(backend);
        let mut registry = NodeExecutorRegistry::default();
        registry
            .register(
                "test.scripted_text",
                Arc::new(ScriptedTextNode {
                    backend: backend.clone(),
                }),
            )
            .expect("register scripted text node");
        let service = RuntimeService::new(
            registry,
            Arc::new(NoopBackendInstanceRuntimeHooks::default()),
            Arc::new(VecRunEventSink::new()),
            Arc::new(FixedClock),
        );
        let handle = service
            .run(
                Arc::new(n_node_plan(&["a"])),
                Default::default(),
                RuntimeOptions::default(),
            )
            .expect("start run");

        // The scripted step enters the hold inside the running node;
        // cancel the run while it is parked there.
        wait_for_condition(Duration::from_secs(5), || hold.is_entered());
        service.cancel(handle.run_id()).expect("cancel run");
        run_to_completion(&service, &handle);

        let summary = service.summary(handle.run_id()).unwrap();
        assert_eq!(
            summary.state,
            reimagine_runtime::RunState::Cancelled,
            "cancelling the run while the scripted step is held must cancel it"
        );
        let snapshot = service.snapshot(handle.run_id()).unwrap();
        assert_eq!(
            snapshot.node_states.get(&NodeId::new("a")),
            Some(&reimagine_runtime::NodeState::Cancelled)
        );
        // The step was cancelled inside the gate, before it served a
        // scripted result, so no call was counted as served.
        assert_eq!(backend.total_calls(), 0);
    });
}
