//! Integration tests for [`ScriptedBackend`].
//!
//! These tests prove the scripted contract end to end: ordered
//! consumption of per-capability `Result` steps, deterministic exhaustion
//! behavior, and the cancellation-aware delay/hold primitives. The final
//! two tests drive the scripted backend through the real built-in
//! executors (router + registry) to prove error propagation through a
//! multi-node pipeline.

use std::sync::Arc;
use std::time::Duration;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{
    ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, SlotId,
    TensorDType, TensorShape, WorkflowId, WorkflowVersion,
};
use reimagine_inference::operation::InferenceCapability;
use reimagine_inference::{
    Backend, BackendPayloadKey, BackendTensorHandle, ConditioningMetadata, ExecutionConditioning,
    ExecutionValue, InferenceBackend, InferenceError, InferenceInvocation, LoadBundleResponse,
    NodeExecutionContext, NodeExecutorError, NodeExecutorRegistry, NodeInputs, NodeParams,
    NoopInferenceProgressSink, NoopNodeCancellation, RuntimeClipHandle, RuntimeModelHandle,
    RuntimeVaeHandle, ScriptedBackend, TextEncodeRequest, TextEncodeResponse,
    register_builtin_inference_executors,
};
use reimagine_inference::{
    ImageSourceResolver, ModelFormat, ModelResolver, RejectAllBridgePolicy, ResolvedImageSource,
    ResolvedInferenceModel, RouterRef,
};

// ── Shared scaffolding (mirrors e2e.rs) ─────────────────────────────

struct FakeResolver {
    path: std::path::PathBuf,
}

impl FakeResolver {
    fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait::async_trait]
impl ModelResolver for FakeResolver {
    async fn resolve(
        &self,
        model_ref: &ModelRef,
    ) -> Result<ResolvedInferenceModel, InferenceError> {
        Ok(ResolvedInferenceModel::new(
            model_ref.id().clone(),
            model_ref.model_series().clone(),
            model_ref.variant().clone(),
            model_ref.role(),
            &self.path,
            ModelFormat::SafeTensors,
        ))
    }
}

struct FakeImageSourceResolver {
    path: std::path::PathBuf,
}

impl FakeImageSourceResolver {
    fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl ImageSourceResolver for FakeImageSourceResolver {
    fn resolve(&self, _path: &std::path::Path) -> Result<ResolvedImageSource, NodeExecutorError> {
        Ok(ResolvedImageSource::new(
            &self.path,
            "image/png",
            self.path
                .file_name()
                .and_then(|n| n.to_str())
                .map(str::to_string),
        ))
    }
}

fn fake_image_resolver() -> Arc<dyn ImageSourceResolver> {
    Arc::new(FakeImageSourceResolver::new(std::path::PathBuf::from(
        "/workspace/input/cat.png",
    )))
}

fn model_ref() -> ModelRef {
    ModelRef::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    )
}

fn runtime_for_backend(backend: Arc<dyn reimagine_inference::InferenceBackend>) -> RouterRef {
    let mut registry = reimagine_inference::InferenceBackendRegistry::new();
    let backend_label = backend.backend_kind().clone();
    let descriptor = reimagine_inference::BackendInstanceDescriptor::new(
        reimagine_inference::BackendInstance::new("scripted:main"),
        backend_label,
    );
    registry.register(descriptor, backend);
    Arc::new(arc_swap::ArcSwap::from_pointee(
        reimagine_inference::InferenceRouter::new(
            Arc::new(registry),
            Arc::new(RejectAllBridgePolicy),
        ),
    ))
}

fn make_context(
    node_id: &str,
    type_id: &str,
    inputs: NodeInputs,
    params: NodeParams,
) -> NodeExecutionContext {
    let publisher: Arc<dyn reimagine_inference::ArtifactPublisher> =
        Arc::new(reimagine_inference::RecordingArtifactPublisher::new());
    let cancellation: Arc<dyn reimagine_inference::NodeCancellation> =
        Arc::new(NoopNodeCancellation::new());
    NodeExecutionContext::new(
        reimagine_core::model::RunId::new("run-scripted"),
        WorkflowId::new("wf-scripted"),
        WorkflowVersion::new(1),
        Some(CorrelationId::new("corr-scripted")),
        NodeId::new(node_id),
        reimagine_core::model::NodeTypeId::new(type_id),
        inputs,
        params,
        publisher,
        cancellation,
        Arc::new(NoopInferenceProgressSink),
        reimagine_core::event::Timestamp::new("2026-07-15T00:00:00Z"),
    )
}

fn make_load_bundle_response() -> LoadBundleResponse {
    let backend = Backend::new("scripted");
    let instance = reimagine_inference::BackendInstance::new("scripted:main");
    let model_id = ModelId::new("sdxl-base-1.0");
    LoadBundleResponse::new(
        RuntimeModelHandle::with_instance(
            model_id.clone(),
            ModelRole::DiffusionModel,
            backend.clone(),
            instance.clone(),
            "model-handle",
        ),
        RuntimeClipHandle::with_instance(
            model_id,
            backend.clone(),
            instance.clone(),
            "clip-handle",
        ),
        RuntimeVaeHandle::with_instance(
            ModelId::new("sdxl-base-1.0"),
            backend,
            instance,
            "vae-handle",
        ),
    )
}

fn conditioning(label: &str) -> ExecutionConditioning {
    ExecutionConditioning::new(
        BackendTensorHandle::new(
            Backend::new("scripted"),
            BackendPayloadKey::new(label),
            TensorDType::F32,
            TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        ConditioningMetadata::new(64, 64),
    )
}

fn text_request(node_id: &str) -> TextEncodeRequest {
    TextEncodeRequest::new(
        RuntimeClipHandle::new(
            ModelId::new("sdxl-base-1.0"),
            Backend::new("scripted"),
            "clip-handle",
        ),
        Arc::new(ExecutionValue::Param(ParamValue::String(
            "a prompt".to_owned(),
        ))),
        reimagine_core::model::RunId::new("run-scripted"),
        WorkflowId::new("wf-scripted"),
        WorkflowVersion::new(1),
        NodeId::new(node_id),
    )
}

// ── Direct backend contract tests ───────────────────────────────────

#[tokio::test]
async fn error_at_step_two_of_three_propagates_with_the_right_variant() {
    let backend = ScriptedBackend::new("scripted").text_encode(vec![
        Ok(TextEncodeResponse::new(conditioning("step-one"))),
        Err(InferenceError::TokenizationFailed {
            message: "emoticon-only prompt".to_owned(),
        }),
        Ok(TextEncodeResponse::new(conditioning("step-three"))),
    ]);

    let invocation = InferenceInvocation::new(
        reimagine_core::model::RunId::new("run-scripted"),
        NodeId::new("node-1"),
        None,
        Arc::new(NoopNodeCancellation::new()),
        Arc::new(NoopInferenceProgressSink),
    );

    let first = backend
        .text_encode_with_invocation(&invocation, text_request("node-1"))
        .await
        .expect("step 1 must succeed");
    assert_eq!(
        first.conditioning().text_embedding().payload_key().as_str(),
        "step-one"
    );

    let second = backend
        .text_encode_with_invocation(&invocation, text_request("node-1"))
        .await
        .expect_err("step 2 must fail");
    match &second {
        InferenceError::TokenizationFailed { message } => {
            assert!(message.contains("emoticon-only prompt"), "{message}");
        }
        other => panic!("expected TokenizationFailed, got {other:?}"),
    }
    assert_eq!(backend.total_calls(), 2);

    let third = backend
        .text_encode_with_invocation(&invocation, text_request("node-1"))
        .await
        .expect("step 3 must succeed");
    assert_eq!(
        third.conditioning().text_embedding().payload_key().as_str(),
        "step-three"
    );
    assert_eq!(backend.total_calls(), 3);
}

#[tokio::test]
async fn held_step_returns_cancelled_when_the_invocation_is_cancelled() {
    let (backend, hold) = ScriptedBackend::new("scripted")
        .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("held")))])
        .with_hold();
    let cancellation = Arc::new(NoopNodeCancellation::new());
    let invocation = InferenceInvocation::new(
        reimagine_core::model::RunId::new("run-scripted"),
        NodeId::new("node-1"),
        None,
        cancellation.clone(),
        Arc::new(NoopInferenceProgressSink),
    );

    let task = tokio::spawn(async move {
        backend
            .text_encode_with_invocation(&invocation, text_request("node-1"))
            .await
    });
    hold.wait_entered().await;
    assert!(hold.is_entered());
    cancellation.cancel();
    hold.release();

    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("held step must resolve promptly after cancel")
        .expect("held step must not panic");
    assert!(matches!(result, Err(InferenceError::Cancelled)));
}

#[tokio::test]
async fn delayed_step_is_interrupted_by_cancellation() {
    let backend = ScriptedBackend::new("scripted")
        .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("slow")))])
        .with_step_delay(Duration::from_secs(60));
    let cancellation = Arc::new(NoopNodeCancellation::new());
    let invocation = InferenceInvocation::new(
        reimagine_core::model::RunId::new("run-scripted"),
        NodeId::new("node-1"),
        None,
        cancellation.clone(),
        Arc::new(NoopInferenceProgressSink),
    );

    let task = tokio::spawn(async move {
        backend
            .text_encode_with_invocation(&invocation, text_request("node-1"))
            .await
    });
    tokio::task::yield_now().await;
    cancellation.cancel();

    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("cancelled delayed step must not wait out the full delay")
        .expect("task must not panic");
    assert!(matches!(result, Err(InferenceError::Cancelled)));
}

// ── Executor-level pipeline tests ───────────────────────────────────

#[tokio::test]
async fn mid_pipeline_failure_propagates_through_builtin_executors() {
    let backend = Arc::new(
        ScriptedBackend::new("scripted")
            .load_bundle(vec![Ok(make_load_bundle_response())])
            .text_encode(vec![Err(InferenceError::TokenizationFailed {
                message: "tokenizer rejected prompt".to_owned(),
            })]),
    );
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(
        &mut registry,
        runtime_for_backend(backend),
        resolver,
        fake_image_resolver(),
    )
    .expect("register executors");

    // Node 1: checkpoint loader — its scripted `load.bundle` succeeds.
    let loader = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.checkpoint_loader",
        ))
        .expect("checkpoint loader registered");
    let mut params = NodeParams::new();
    params.insert(SlotId::new("checkpoint"), ParamValue::ModelRef(model_ref()));
    let loader_out = loader
        .execute(make_context(
            "loader",
            "builtin.checkpoint_loader",
            NodeInputs::new(),
            params,
        ))
        .await
        .expect("load.bundle step must succeed");
    let model_value = loader_out
        .iter()
        .find(|output| output.slot_id() == &SlotId::new("clip"))
        .map(reimagine_inference::ExecutionOutput::value)
        .expect("loader must emit a model value")
        .clone();

    // Node 2: clip text encoder — its scripted `text.encode` fails with
    // the exact scripted variant, mapped to a node failure.
    let encoder = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.clip_text_encode",
        ))
        .expect("clip text encode registered");
    let mut inputs = NodeInputs::new();
    inputs.insert(SlotId::new("clip"), model_value);
    inputs.insert(
        SlotId::new("text"),
        Arc::new(ExecutionValue::Param(ParamValue::String(
            "a prompt".to_owned(),
        ))),
    );
    let error = encoder
        .execute(make_context(
            "encoder",
            "builtin.clip_text_encode",
            inputs,
            NodeParams::new(),
        ))
        .await
        .expect_err("text.encode step 2 must fail the node");
    match &error {
        NodeExecutorError::Failed { message } => {
            assert!(message.contains("tokenizer rejected prompt"), "{message}");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[tokio::test]
async fn out_of_memory_variant_maps_to_a_failed_node_not_cancelled() {
    let backend = Arc::new(
        ScriptedBackend::new("scripted")
            .load_bundle(vec![Ok(make_load_bundle_response())])
            .text_encode(vec![Err(InferenceError::OutOfMemory {
                requested: Some(1 << 30),
                available: None,
            })]),
    );
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(
        &mut registry,
        runtime_for_backend(backend),
        resolver,
        fake_image_resolver(),
    )
    .expect("register executors");

    let loader = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.checkpoint_loader",
        ))
        .expect("checkpoint loader registered");
    let mut params = NodeParams::new();
    params.insert(SlotId::new("checkpoint"), ParamValue::ModelRef(model_ref()));
    let loader_out = loader
        .execute(make_context(
            "loader",
            "builtin.checkpoint_loader",
            NodeInputs::new(),
            params,
        ))
        .await
        .expect("load.bundle step must succeed");

    let encoder = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.clip_text_encode",
        ))
        .expect("clip text encode registered");
    let model_value = loader_out
        .iter()
        .find(|output| output.slot_id() == &SlotId::new("clip"))
        .map(reimagine_inference::ExecutionOutput::value)
        .expect("loader must emit a model value")
        .clone();
    let mut inputs = NodeInputs::new();
    inputs.insert(SlotId::new("clip"), model_value);
    inputs.insert(
        SlotId::new("text"),
        Arc::new(ExecutionValue::Param(ParamValue::String(
            "a prompt".to_owned(),
        ))),
    );
    let error = encoder
        .execute(make_context(
            "encoder",
            "builtin.clip_text_encode",
            inputs,
            NodeParams::new(),
        ))
        .await
        .expect_err("scripted OutOfMemory must fail the node");
    assert!(
        matches!(error, NodeExecutorError::Failed { .. }),
        "OutOfMemory must map to a Failed node, got {error:?}"
    );
    assert!(error.to_string().contains("out of memory"));
    assert!(!matches!(error, NodeExecutorError::Cancelled));
}

#[test]
fn scripted_backend_kind_and_scripted_capabilities_are_visible() {
    let backend = ScriptedBackend::new("my-scripted")
        .text_encode(vec![Ok(TextEncodeResponse::new(conditioning("x")))]);
    assert_eq!(backend.backend_kind().as_str(), "my-scripted");
    let caps = backend.capabilities();
    assert!(caps.supports_capability(InferenceCapability::TextEncode));
    assert!(!caps.supports_capability(InferenceCapability::DiffusionSample));
}

#[tokio::test]
async fn scripted_backend_reports_call_counts() {
    let backend = ScriptedBackend::new("scripted").text_encode(vec![
        Ok(TextEncodeResponse::new(conditioning("a"))),
        Ok(TextEncodeResponse::new(conditioning("b"))),
    ]);
    assert_eq!(backend.total_calls(), 0);
    let invocation = InferenceInvocation::new(
        reimagine_core::model::RunId::new("run-scripted"),
        NodeId::new("node-1"),
        None,
        Arc::new(NoopNodeCancellation::new()),
        Arc::new(NoopInferenceProgressSink),
    );
    backend
        .text_encode_with_invocation(&invocation, text_request("node-1"))
        .await
        .expect("step 1");
    backend
        .text_encode_with_invocation(&invocation, text_request("node-1"))
        .await
        .expect("step 2");
    assert_eq!(backend.total_calls(), 2);
}
