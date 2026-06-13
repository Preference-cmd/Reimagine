//! Integration tests for the inference layer.
//!
//! These tests exercise the full executor path: register V1 built-in
//! executors with a [`FakeBackend`], drive them through a synthetic
//! [`NodeExecutionContext`], and assert the output slot names, order
//! independence, error propagation, and validation invariants.

use std::sync::Arc;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{
    ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, ParamValue, SlotId,
};
use reimagine_runtime::{
    CancellationToken, NodeExecutionContext, NodeExecutorError, NodeExecutorRegistry, NodeInputs,
    NodeParams, RuntimeValue, VecRunEventSink,
};

use reimagine_inference::operation::*;
use reimagine_inference::testing::fake_output;
use reimagine_inference::{
    FakeBackend, InferenceError, ModelFormat, ModelResolver, ResolvedInferenceModel,
    register_builtin_inference_executors,
};

// ── Fake model resolver ────────────────────────────────────────────

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

// ── Test helpers ───────────────────────────────────────────────────

fn model_ref() -> ModelRef {
    ModelRef::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    )
}

fn fake_checkpoint_outputs() -> Vec<(SlotId, Arc<RuntimeValue>)> {
    vec![
        fake_output("model", ParamValue::String("model-handle".to_string())),
        fake_output("clip", ParamValue::String("clip-handle".to_string())),
        fake_output("vae", ParamValue::String("vae-handle".to_string())),
    ]
}

fn make_context(
    node_id: &str,
    type_id: &str,
    inputs: NodeInputs,
    params: reimagine_runtime::NodeParams,
) -> NodeExecutionContext {
    NodeExecutionContext::new(
        reimagine_core::model::RunId::new("run-test"),
        reimagine_core::model::WorkflowId::new("wf-test"),
        reimagine_core::model::WorkflowVersion::new(1),
        Some(CorrelationId::new("corr-test")),
        reimagine_core::model::NodeId::new(node_id),
        reimagine_core::model::NodeTypeId::new(type_id),
        inputs,
        params,
        make_artifact_capability(reimagine_core::model::NodeId::new(node_id)),
        CancellationToken::new(),
        reimagine_core::event::Timestamp::new("2026-06-13T00:00:00Z"),
    )
}

fn make_artifact_capability(
    node_id: reimagine_core::model::NodeId,
) -> reimagine_runtime::NodeArtifactCapability {
    use reimagine_runtime::{Clock, RunEventSink, SystemClock};
    use std::sync::Arc;

    let store = Arc::new(tokio::sync::Mutex::new(
        reimagine_runtime::ArtifactStore::new(),
    ));
    let sink: Arc<dyn RunEventSink> = Arc::new(VecRunEventSink::new());
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    reimagine_runtime::NodeArtifactCapability::new(
        reimagine_core::model::RunId::new("run-test"),
        reimagine_core::model::WorkflowId::new("wf-test"),
        reimagine_core::model::WorkflowVersion::new(1),
        node_id,
        store,
        sink,
        clock,
        CancellationToken::new(),
    )
}

// ── Tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn checkpoint_loader_multi_output_by_slot_id() {
    let backend = Arc::new(
        FakeBackend::new("fake").with_operation(OP_MODEL_LOAD_BUNDLE, fake_checkpoint_outputs()),
    );
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.checkpoint_loader",
        ))
        .expect("executor registered");

    let mut params = reimagine_runtime::NodeParams::new();
    params.insert(SlotId::new("checkpoint"), ParamValue::ModelRef(model_ref()));
    let context = make_context(
        "loader",
        "builtin.checkpoint_loader",
        NodeInputs::new(),
        params,
    );

    let result = executor.execute(context).await.expect("execute ok");
    assert_eq!(result.len(), 3);

    let slot_names: Vec<&str> = result.iter().map(|(s, _)| s.as_str()).collect();
    assert!(slot_names.contains(&"model"));
    assert!(slot_names.contains(&"clip"));
    assert!(slot_names.contains(&"vae"));
}

#[tokio::test]
async fn checkpoint_loader_slot_order_is_irrelevant() {
    // Return outputs in reversed order (vae, clip, model) — should
    // still pass validation because slot order must not matter.
    let reversed_outputs = vec![
        fake_output("vae", ParamValue::String("vae-handle".to_string())),
        fake_output("clip", ParamValue::String("clip-handle".to_string())),
        fake_output("model", ParamValue::String("model-handle".to_string())),
    ];
    let backend =
        Arc::new(FakeBackend::new("fake").with_operation(OP_MODEL_LOAD_BUNDLE, reversed_outputs));
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.checkpoint_loader",
        ))
        .expect("executor registered");

    let mut params = reimagine_runtime::NodeParams::new();
    params.insert(SlotId::new("checkpoint"), ParamValue::ModelRef(model_ref()));
    let context = make_context(
        "loader",
        "builtin.checkpoint_loader",
        NodeInputs::new(),
        params,
    );

    let result = executor.execute(context).await.expect("execute ok");
    // The output order from the executor matches the backend's
    // returned order (the validator preserves order), but the runtime
    // doesn't depend on it. Assert all three are present.
    let slot_names: Vec<&str> = result.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(slot_names.len(), 3);
    assert!(slot_names.contains(&"model"), "model slot missing");
    assert!(slot_names.contains(&"clip"), "clip slot missing");
    assert!(slot_names.contains(&"vae"), "vae slot missing");
}

#[tokio::test]
async fn checkpoint_loader_duplicate_output_slot_is_error() {
    // Return two "model" slots — should be rejected by validation.
    let dup_outputs = vec![
        fake_output("model", ParamValue::String("m1".to_string())),
        fake_output("model", ParamValue::String("m2".to_string())),
        fake_output("clip", ParamValue::String("c".to_string())),
        fake_output("vae", ParamValue::String("v".to_string())),
    ];
    let backend =
        Arc::new(FakeBackend::new("fake").with_operation(OP_MODEL_LOAD_BUNDLE, dup_outputs));
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.checkpoint_loader",
        ))
        .expect("executor registered");

    let mut params = reimagine_runtime::NodeParams::new();
    params.insert(SlotId::new("checkpoint"), ParamValue::ModelRef(model_ref()));
    let context = make_context(
        "loader",
        "builtin.checkpoint_loader",
        NodeInputs::new(),
        params,
    );

    let err = executor.execute(context).await.expect_err("should fail");
    assert!(
        err.to_string().contains("duplicate"),
        "expected duplicate slot error, got: {err}"
    );
}

#[tokio::test]
async fn checkpoint_loader_undeclared_output_slot_is_error() {
    // Return an extra "noise" slot not declared by the node.
    let extra_outputs = vec![
        fake_output("model", ParamValue::String("m".to_string())),
        fake_output("clip", ParamValue::String("c".to_string())),
        fake_output("vae", ParamValue::String("v".to_string())),
        fake_output("noise", ParamValue::String("n".to_string())),
    ];
    let backend =
        Arc::new(FakeBackend::new("fake").with_operation(OP_MODEL_LOAD_BUNDLE, extra_outputs));
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.checkpoint_loader",
        ))
        .expect("executor registered");

    let mut params = reimagine_runtime::NodeParams::new();
    params.insert(SlotId::new("checkpoint"), ParamValue::ModelRef(model_ref()));
    let context = make_context(
        "loader",
        "builtin.checkpoint_loader",
        NodeInputs::new(),
        params,
    );

    let err = executor.execute(context).await.expect_err("should fail");
    assert!(
        err.to_string().contains("undeclared"),
        "expected undeclared slot error, got: {err}"
    );
}

#[tokio::test]
async fn unregistered_operation_returns_backend_not_implemented() {
    // Register no operations — every call should fail with
    // BackendNotImplemented.
    let backend = Arc::new(FakeBackend::new("empty"));
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.checkpoint_loader",
        ))
        .expect("executor registered");

    let mut params = reimagine_runtime::NodeParams::new();
    params.insert(SlotId::new("checkpoint"), ParamValue::ModelRef(model_ref()));
    let context = make_context(
        "loader",
        "builtin.checkpoint_loader",
        NodeInputs::new(),
        params,
    );

    let err = executor.execute(context).await.expect_err("should fail");
    assert!(
        err.to_string().contains("does not implement"),
        "expected backend-not-implemented, got: {err}"
    );
}

#[tokio::test]
async fn string_executor_passthrough() {
    // builtin.string does not call the backend at all.
    let backend = Arc::new(FakeBackend::new("unused"));
    let resolver = Arc::new(FakeResolver::new("/unused"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new("builtin.string"))
        .expect("executor registered");

    let mut params = reimagine_runtime::NodeParams::new();
    params.insert(
        SlotId::new("value"),
        ParamValue::String("hello world".to_string()),
    );
    let context = make_context("str", "builtin.string", NodeInputs::new(), params);

    let result = executor.execute(context).await.expect("execute ok");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0.as_str(), "value");
    match result[0].1.as_ref() {
        RuntimeValue::Param(ParamValue::String(s)) => assert_eq!(s, "hello world"),
        other => panic!("expected Param(String), got {other:?}"),
    }
}

#[tokio::test]
async fn all_v1_executors_register_successfully() {
    let backend = Arc::new(FakeBackend::new("fake"));
    let resolver = Arc::new(FakeResolver::new("/models/test.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    // All 8 V1 built-ins should be registered.
    for type_id in &[
        "builtin.string",
        "builtin.checkpoint_loader",
        "builtin.clip_text_encode",
        "builtin.empty_latent_image",
        "builtin.ksampler",
        "builtin.vae_decode",
        "builtin.save_image",
        "builtin.preview_image",
    ] {
        assert!(
            registry
                .get(&reimagine_core::model::NodeTypeId::new(*type_id))
                .is_some(),
            "executor for `{type_id}` should be registered"
        );
    }
}

#[tokio::test]
async fn checkpoint_loader_missing_model_ref_is_error() {
    // Omit the checkpoint param entirely.
    let backend = Arc::new(
        FakeBackend::new("fake").with_operation(OP_MODEL_LOAD_BUNDLE, fake_checkpoint_outputs()),
    );
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.checkpoint_loader",
        ))
        .expect("executor registered");

    let context = make_context(
        "loader",
        "builtin.checkpoint_loader",
        NodeInputs::new(),
        reimagine_runtime::NodeParams::new(),
    );

    let err = executor.execute(context).await.expect_err("should fail");
    assert!(
        matches!(err, NodeExecutorError::MissingInput { .. }),
        "expected MissingInput, got: {err:?}"
    );
}

#[test]
fn inference_error_boundary_is_explicit() {
    // Verify that InferenceError does NOT have a broad From impl for
    // NodeExecutorError — only the explicit `into_executor_error()`.
    let err = InferenceError::BackendNotImplemented {
        operation_id: "test".to_string(),
        backend_kind: "fake".to_string(),
    };
    let exec_err = err.into_executor_error();
    assert!(exec_err.to_string().contains("does not implement"));
}
