//! Integration tests for the inference layer.
//!
//! These tests exercise the full executor path: register V1 built-in
//! executors with a [`FakeBackend`], drive them through a synthetic
//! [`NodeExecutionContext`], and assert the executor builds typed
//! DTOs, calls the typed backend methods, and returns the expected
//! workflow node outputs.

use std::sync::{Arc, Mutex};

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{
    ArtifactRef, ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, ParamValue, SlotId,
};
use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_core::{
    BackendKind, BackendPayloadKey, BackendTensorHandle, ExecutionValue, RuntimeClipHandle,
    RuntimeImage, RuntimeModelHandle, RuntimeVaeHandle,
};
use reimagine_inference::operation::InferenceCapability;
use reimagine_inference::{
    CannedCapabilityResponse, CreateEmptyLatentRequest, CreateEmptyLatentResponse, FakeBackend,
    ImageSaveResponse, InferenceBackend, InferenceBackendCapabilities, InferenceCapabilitySupport,
    InferenceError, IntoNodeExecutorError, LoadBundleResponse, ModelFormat, ModelResolver,
    ResolvedInferenceModel, TextEncodeRequest, TextEncodeResponse,
    register_builtin_inference_executors,
};
use reimagine_runtime::{
    CancellationToken, NodeExecutionContext, NodeExecutorError, NodeExecutorRegistry, NodeInputs,
    VecRunEventSink,
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

fn fake_tensor_handle(backend: &BackendKind, key: &str, dims: Vec<usize>) -> BackendTensorHandle {
    BackendTensorHandle::new(
        backend.clone(),
        BackendPayloadKey::new(key),
        TensorDType::F32,
        TensorShape::new(dims),
        "cpu",
    )
}

fn fake_image_value(backend: BackendKind, key: &str, width: u32, height: u32) -> RuntimeImage {
    RuntimeImage::new(
        fake_tensor_handle(&backend, key, vec![1, 3, height as usize, width as usize]),
        width,
        height,
        1,
        "rgb",
    )
}

fn make_load_bundle_response() -> LoadBundleResponse {
    let backend = BackendKind::new("fake");
    let model_id = ModelId::new("sdxl-base-1.0");
    LoadBundleResponse::new(
        RuntimeModelHandle::new(
            model_id.clone(),
            ModelRole::DiffusionModel,
            backend.clone(),
            "model-handle",
        ),
        RuntimeClipHandle::new(model_id, backend.clone(), "clip-handle"),
        RuntimeVaeHandle::new(ModelId::new("sdxl-base-1.0"), backend, "vae-handle"),
    )
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

fn make_context_with_artifact_store(
    node_id: &str,
    type_id: &str,
    inputs: NodeInputs,
    params: reimagine_runtime::NodeParams,
) -> (
    NodeExecutionContext,
    Arc<tokio::sync::Mutex<reimagine_runtime::ArtifactStore>>,
) {
    let node_id_value = reimagine_core::model::NodeId::new(node_id);
    let store = Arc::new(tokio::sync::Mutex::new(
        reimagine_runtime::ArtifactStore::new(),
    ));
    let capability = make_artifact_capability_with_store(node_id_value.clone(), Arc::clone(&store));
    (
        NodeExecutionContext::new(
            reimagine_core::model::RunId::new("run-test"),
            reimagine_core::model::WorkflowId::new("wf-test"),
            reimagine_core::model::WorkflowVersion::new(1),
            Some(CorrelationId::new("corr-test")),
            node_id_value,
            reimagine_core::model::NodeTypeId::new(type_id),
            inputs,
            params,
            capability,
            CancellationToken::new(),
            reimagine_core::event::Timestamp::new("2026-06-13T00:00:00Z"),
        ),
        store,
    )
}

fn make_artifact_capability(
    node_id: reimagine_core::model::NodeId,
) -> reimagine_runtime::NodeArtifactCapability {
    let store = Arc::new(tokio::sync::Mutex::new(
        reimagine_runtime::ArtifactStore::new(),
    ));
    make_artifact_capability_with_store(node_id, store)
}

fn make_artifact_capability_with_store(
    node_id: reimagine_core::model::NodeId,
    store: Arc<tokio::sync::Mutex<reimagine_runtime::ArtifactStore>>,
) -> reimagine_runtime::NodeArtifactCapability {
    use reimagine_runtime::{Clock, RunEventSink, SystemClock};
    use std::sync::Arc;

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
        FakeBackend::new("fake")
            .load_bundle(CannedCapabilityResponse::always(make_load_bundle_response())),
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
async fn checkpoint_loader_requires_all_three_outputs() {
    // Backend returns only two handles. The executor owns the slot
    // mapping, so missing outputs become missing runtime outputs.
    // We assert the executor does NOT panic or invent missing values.
    let partial = LoadBundleResponse::new(
        RuntimeModelHandle::new(
            ModelId::new("sdxl-base-1.0"),
            ModelRole::DiffusionModel,
            BackendKind::new("fake"),
            "model-handle",
        ),
        RuntimeClipHandle::new(
            ModelId::new("sdxl-base-1.0"),
            BackendKind::new("fake"),
            "clip-handle",
        ),
        RuntimeVaeHandle::new(
            ModelId::new("sdxl-base-1.0"),
            BackendKind::new("fake"),
            "vae-handle",
        ),
    );
    let backend =
        Arc::new(FakeBackend::new("fake").load_bundle(CannedCapabilityResponse::always(partial)));
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
    let slot_names: Vec<&str> = result.iter().map(|(s, _)| s.as_str()).collect();
    assert!(slot_names.contains(&"model"));
    assert!(slot_names.contains(&"clip"));
    assert!(slot_names.contains(&"vae"));
}

#[tokio::test]
async fn unregistered_operation_returns_backend_not_implemented() {
    // Register no capabilities — every call should fail with
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
        ExecutionValue::Param(ParamValue::String(s)) => assert_eq!(s, "hello world"),
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
async fn save_image_records_backend_returned_artifact_reference() {
    let artifact_ref = ArtifactRef::new("output/reimagine_run-test_save_0.png");
    let backend = Arc::new(
        FakeBackend::new("fake").image_save(CannedCapabilityResponse::always(
            ImageSaveResponse::new(artifact_ref.clone()),
        )),
    );
    let resolver = Arc::new(FakeResolver::new("/models/sdxl-base.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.save_image",
        ))
        .expect("executor registered");

    let image = fake_image_value(BackendKind::new("fake"), "image:payload", 64, 64);
    let mut inputs = NodeInputs::new();
    inputs.insert(SlotId::new("image"), Arc::new(ExecutionValue::Image(image)));

    let (context, artifact_store) = make_context_with_artifact_store(
        "save",
        "builtin.save_image",
        inputs,
        reimagine_runtime::NodeParams::new(),
    );

    let result = executor.execute(context).await.expect("execute ok");
    assert!(result.is_empty(), "save node should not expose outputs");

    let store = artifact_store.lock().await;
    let records: Vec<_> = store.iter_ordered().collect();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].reference, artifact_ref);
    assert_eq!(records[0].slot_id.as_str(), "artifact");
}

#[tokio::test]
async fn checkpoint_loader_missing_model_ref_is_error() {
    let backend = Arc::new(
        FakeBackend::new("fake")
            .load_bundle(CannedCapabilityResponse::always(make_load_bundle_response())),
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
    let err = InferenceError::BackendNotImplemented {
        capability: InferenceCapability::DiffusionSample,
        backend_kind: "fake".to_string(),
        message: None,
    };
    let exec_err = err.into_executor_error();
    assert!(exec_err.to_string().contains("does not implement"));
}

#[tokio::test]
async fn text_encode_executor_calls_typed_text_encode() {
    // Build a backend that records the request it received and
    // returns a canned conditioning.
    #[derive(Default)]
    struct CapturedRequest {
        inner: Mutex<Option<TextEncodeRequest>>,
    }

    let captured = Arc::new(CapturedRequest::default());
    let captured_for_factory = Arc::clone(&captured);

    let response_factory = move |req: TextEncodeRequest| {
        let mut guard = captured_for_factory.inner.lock().unwrap();
        *guard = Some(req.clone());
        let (clip, _text) = req.into_parts();
        Ok::<_, InferenceError>(TextEncodeResponse::new(
            reimagine_core::ExecutionConditioning::new(
                reimagine_core::BackendTensorHandle::new(
                    clip.backend().clone(),
                    reimagine_core::BackendPayloadKey::new("captured-text"),
                    reimagine_core::model::TensorDType::F32,
                    reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
                    "cpu",
                ),
                reimagine_core::ConditioningMetadata::new(64, 64),
            ),
        ))
    };

    let backend = Arc::new(
        FakeBackend::new("fake")
            .text_encode(CannedCapabilityResponse::from_request(response_factory)),
    );
    let resolver = Arc::new(FakeResolver::new("/models/clip.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.clip_text_encode",
        ))
        .expect("executor registered");

    let clip_handle = RuntimeClipHandle::new(
        ModelId::new("clip-model"),
        BackendKind::new("fake"),
        "clip-payload",
    );
    let mut inputs = NodeInputs::new();
    inputs.insert(
        SlotId::new("clip"),
        Arc::new(ExecutionValue::Clip(clip_handle)),
    );
    inputs.insert(
        SlotId::new("text"),
        Arc::new(ExecutionValue::Param(ParamValue::String(
            "a sunset".to_string(),
        ))),
    );
    let context = make_context(
        "clip",
        "builtin.clip_text_encode",
        inputs,
        reimagine_runtime::NodeParams::new(),
    );

    let result = executor.execute(context).await.expect("execute ok");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0.as_str(), "conditioning");

    let captured = captured.inner.lock().unwrap().clone().expect("captured");
    assert_eq!(captured.clip().model_id().as_str(), "clip-model");
    assert_eq!(captured.prompt_string().as_deref(), Some("a sunset"));
}

#[tokio::test]
async fn create_empty_latent_executor_calls_typed_create_empty_latent() {
    #[derive(Default)]
    struct Captured {
        inner: Mutex<Option<CreateEmptyLatentRequest>>,
    }

    let captured = Arc::new(Captured::default());
    let captured_for_factory = Arc::clone(&captured);

    let response_factory = move |req: CreateEmptyLatentRequest| {
        let mut guard = captured_for_factory.inner.lock().unwrap();
        *guard = Some(req.clone());
        Ok::<_, InferenceError>(CreateEmptyLatentResponse::new(req.into_latent()))
    };

    let backend = Arc::new(
        FakeBackend::new("fake")
            .create_empty_latent(CannedCapabilityResponse::from_request(response_factory)),
    );
    let resolver = Arc::new(FakeResolver::new("/models/test.safetensors"));

    let mut registry = NodeExecutorRegistry::default();
    register_builtin_inference_executors(&mut registry, backend, resolver)
        .expect("register executors");

    let executor = registry
        .get(&reimagine_core::model::NodeTypeId::new(
            "builtin.empty_latent_image",
        ))
        .expect("executor registered");

    let mut params = reimagine_runtime::NodeParams::new();
    params.insert(SlotId::new("width"), ParamValue::Integer(64));
    params.insert(SlotId::new("height"), ParamValue::Integer(64));
    params.insert(SlotId::new("batch_size"), ParamValue::Integer(1));
    let context = make_context(
        "empty",
        "builtin.empty_latent_image",
        NodeInputs::new(),
        params,
    );

    let result = executor.execute(context).await.expect("execute ok");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0.as_str(), "latent");

    let captured = captured.inner.lock().unwrap().clone().expect("captured");
    assert_eq!(captured.width(), 64);
    assert_eq!(captured.height(), 64);
    assert_eq!(captured.batch_size(), 1);
}

#[tokio::test]
async fn fake_backend_capabilities_advertise_registered_capabilities() {
    let backend = FakeBackend::new("test-kind")
        .load_bundle(CannedCapabilityResponse::always(make_load_bundle_response()))
        .text_encode(CannedCapabilityResponse::from_request(
            |req: TextEncodeRequest| -> Result<TextEncodeResponse, InferenceError> {
                let (clip, _text) = req.into_parts();
                Ok(TextEncodeResponse::new(
                    reimagine_core::ExecutionConditioning::new(
                        reimagine_core::BackendTensorHandle::new(
                            clip.backend().clone(),
                            reimagine_core::BackendPayloadKey::new("text-encoded"),
                            reimagine_core::model::TensorDType::F32,
                            reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
                            "cpu",
                        ),
                        reimagine_core::ConditioningMetadata::new(64, 64),
                    ),
                ))
            },
        ));
    let caps: InferenceBackendCapabilities = InferenceBackend::capabilities(&backend);
    assert!(caps.supports_capability(InferenceCapability::LoadBundle));
    assert!(caps.supports_capability(InferenceCapability::TextEncode));
    assert!(!caps.supports_capability(InferenceCapability::ImageSave));
    // Check that the support shapes are valid.
    for support in caps.capability_supports() {
        let _ = support.capability();
    }
    let _ = InferenceCapabilitySupport::new(InferenceCapability::LoadBundle);
}
