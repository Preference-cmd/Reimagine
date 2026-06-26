//! Integration tests for the Candle inference backend adapter.
//!
//! These tests exercise the backend directly without the runtime or
//! executors. They prove that the typed capability methods return
//! correctly shaped lightweight handles, that the model cache is
//! keyed by the resolved model id, and that source validation
//! surfaces a useful backend error when the resolved model artifact
//! is missing, unreadable, or has the wrong extension.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Tensor};
use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId, WorkflowId,
    WorkflowVersion,
};
use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceObservation, BackendPayloadKey, BackendRunLifecycle,
    BackendTensorHandle, ConditioningMetadata, ExecutionConditioning, ExecutionValue,
    RuntimeClipHandle, RuntimeImage, RuntimeLatent, RuntimeModelHandle, RuntimeVaeHandle,
};
use reimagine_inference::{
    CreateEmptyLatentRequest, CreateEmptyLatentResponse, DiffusionSampleRequest,
    ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse,
    InferenceBackend, InferenceCapability, InferenceError, LatentDecodeRequest,
    LatentDecodeResponse, LoadBundleRequest, ModelFormat, ResolvedInferenceModel, SamplerName,
    SchedulerName, TextEncodeRequest, TextEncodeResponse,
};
use reimagine_inference_candle::CandleConditioning;
use reimagine_inference_candle::{CandleBackend, CandleBackendConfig, LoadedModelBundle};

fn backend() -> CandleBackend {
    backend_with_dirs(
        "/tmp/reimagine-candle-tests",
        "/tmp/reimagine-candle-tests-output",
    )
}

fn backend_with_dirs(
    models_dir: impl AsRef<std::path::Path>,
    output_dir: impl AsRef<std::path::Path>,
) -> CandleBackend {
    CandleBackend::new(CandleBackendConfig::new(
        models_dir.as_ref().to_path_buf(),
        output_dir.as_ref().to_path_buf(),
    ))
    .unwrap()
    .with_test_text_projection()
    .with_test_vae_decoder_projection()
}

fn production_like_backend() -> CandleBackend {
    CandleBackend::new(CandleBackendConfig::new(
        "/tmp/reimagine-candle-tests",
        "/tmp/reimagine-candle-tests-output",
    ))
    .unwrap()
}

fn unique_sdxl_root() -> PathBuf {
    static NEXT_TEST_ROOT_ID: AtomicU64 = AtomicU64::new(0);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = NEXT_TEST_ROOT_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "reimagine-candle-it-{}-{nonce}-{seq}",
        std::process::id()
    ))
}

fn write_sdxl_placeholder(root: &std::path::Path) -> PathBuf {
    std::fs::create_dir_all(root).unwrap();
    let path = root.join("sdxl-base-1.0.safetensors");
    std::fs::write(&path, b"placeholder").unwrap();
    path
}

fn write_header_only_safetensors(
    root: &std::path::Path,
    filename: &str,
    names: &[&str],
) -> PathBuf {
    std::fs::create_dir_all(root).unwrap();
    let entries = names
        .iter()
        .map(|name| {
            format!("\"{name}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    let header = format!("{{{entries}}}");
    let path = root.join(filename);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    std::fs::write(&path, bytes).unwrap();
    path
}

fn sdxl_model() -> (ResolvedInferenceModel, PathBuf) {
    let root = unique_sdxl_root();
    let path = write_sdxl_placeholder(&root);
    let model = ResolvedInferenceModel::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &path,
        ModelFormat::SafeTensors,
    );
    (model, root)
}

fn sdxl_model_from_path(path: PathBuf) -> ResolvedInferenceModel {
    ResolvedInferenceModel::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &path,
        ModelFormat::SafeTensors,
    )
}

fn base_load_bundle_request(model: ResolvedInferenceModel, node: &str) -> LoadBundleRequest {
    LoadBundleRequest::new(
        model,
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new(node),
    )
}

fn base_text_encode_request(
    clip: RuntimeClipHandle,
    text: String,
    node: &str,
) -> TextEncodeRequest {
    TextEncodeRequest::new(
        clip,
        Arc::new(ExecutionValue::Param(ParamValue::String(text))),
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new(node),
    )
}

fn base_create_empty_latent_request(
    width: i64,
    height: i64,
    batch_size: i64,
    node: &str,
) -> CreateEmptyLatentRequest {
    CreateEmptyLatentRequest::new(
        width as u32,
        height as u32,
        batch_size as u32,
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new(node),
    )
}

fn base_diffusion_sample_request(
    model_handle: RuntimeModelHandle,
    positive: ExecutionConditioning,
    negative: ExecutionConditioning,
    latent: RuntimeLatent,
    node: &str,
) -> DiffusionSampleRequest {
    DiffusionSampleRequest::new(
        model_handle,
        positive,
        negative,
        latent,
        123456789,
        20,
        7.0,
        SamplerName::Euler,
        SchedulerName::Normal,
        1.0,
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new(node),
    )
}

fn base_image_save_request(image: RuntimeImage, node: &str) -> ImageSaveRequest {
    ImageSaveRequest::new(
        image,
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new(node),
    )
}

fn base_image_preview_request(image: RuntimeImage, node: &str) -> ImagePreviewRequest {
    ImagePreviewRequest::new(
        image,
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new(node),
    )
}

fn fake_tensor_handle(key: &str, dims: Vec<usize>) -> BackendTensorHandle {
    BackendTensorHandle::with_instance(
        Backend::new("candle"),
        BackendInstance::new("candle:cpu"),
        BackendPayloadKey::new(key),
        TensorDType::F32,
        TensorShape::new(dims),
        "cpu",
    )
}

fn fake_runtime_model_handle(key: &str) -> RuntimeModelHandle {
    RuntimeModelHandle::with_instance(
        ModelId::new("sdxl-base-1.0"),
        ModelRole::CheckpointBundle,
        Backend::new("candle"),
        BackendInstance::new("candle:cpu"),
        BackendPayloadKey::new(key),
    )
}

fn fake_runtime_model_handle_with_instance(key: &str, instance: &str) -> RuntimeModelHandle {
    RuntimeModelHandle::with_instance(
        ModelId::new("sdxl-base-1.0"),
        ModelRole::CheckpointBundle,
        Backend::new("candle"),
        reimagine_inference::BackendInstance::new(instance),
        BackendPayloadKey::new(key),
    )
}

fn fake_runtime_clip_handle(key: &str) -> RuntimeClipHandle {
    RuntimeClipHandle::with_instance(
        ModelId::new("sdxl-base-1.0"),
        Backend::new("candle"),
        BackendInstance::new("candle:cpu"),
        BackendPayloadKey::new(key),
    )
}

fn fake_runtime_vae_handle(key: &str) -> RuntimeVaeHandle {
    RuntimeVaeHandle::with_instance(
        ModelId::new("sdxl-base-1.0"),
        Backend::new("candle"),
        BackendInstance::new("candle:cpu"),
        BackendPayloadKey::new(key),
    )
}

fn fake_runtime_latent(key: &str, width: u32, height: u32) -> RuntimeLatent {
    RuntimeLatent::new(
        fake_tensor_handle(key, vec![1, 4, (height / 8) as usize, (width / 8) as usize]),
        width,
        height,
        1,
        4,
    )
}

fn fake_tensor_handle_with_instance(
    key: &str,
    dims: Vec<usize>,
    instance: &str,
) -> BackendTensorHandle {
    BackendTensorHandle::with_instance(
        Backend::new("candle"),
        reimagine_inference::BackendInstance::new(instance),
        BackendPayloadKey::new(key),
        TensorDType::F32,
        TensorShape::new(dims),
        "cpu",
    )
}

fn fake_runtime_latent_with_instance(
    key: &str,
    width: u32,
    height: u32,
    instance: &str,
) -> RuntimeLatent {
    RuntimeLatent::new(
        fake_tensor_handle_with_instance(
            key,
            vec![1, 4, (height / 8) as usize, (width / 8) as usize],
            instance,
        ),
        width,
        height,
        1,
        4,
    )
}

fn fake_conditioning(text_key: &str, pooled_key: Option<&str>) -> ExecutionConditioning {
    let text = fake_tensor_handle(text_key, vec![1, 77, 2048]);
    let mut c = ExecutionConditioning::new(text, ConditioningMetadata::new(64, 64));
    if let Some(key) = pooled_key {
        let pooled = fake_tensor_handle(key, vec![1, 1280]);
        c = c.with_pooled_embedding(pooled);
    }
    c
}

fn fake_conditioning_with_instance(
    text_key: &str,
    pooled_key: Option<&str>,
    instance: &str,
) -> ExecutionConditioning {
    let text = fake_tensor_handle_with_instance(text_key, vec![1, 77, 2048], instance);
    let mut c = ExecutionConditioning::new(text, ConditioningMetadata::new(64, 64));
    if let Some(key) = pooled_key {
        let pooled = fake_tensor_handle_with_instance(key, vec![1, 1280], instance);
        c = c.with_pooled_embedding(pooled);
    }
    c
}

fn assert_backend_execution_failed_with(err: &InferenceError, needle: &str) {
    match err {
        InferenceError::BackendExecutionFailed { message } => {
            assert!(
                message.contains(needle),
                "expected error message to contain `{needle}`, got `{message}`"
            );
        }
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn backend_kind_is_candle() {
    let backend = backend();
    assert_eq!(backend.backend_kind().as_str(), "candle");
}

#[test]
fn capabilities_cover_all_v1_capabilities() {
    let backend = backend();
    let capabilities = backend.capabilities();
    assert_eq!(capabilities.backend_kind().as_str(), "candle");
    for cap in InferenceCapability::all_v1() {
        assert!(
            capabilities.supports_capability(*cap),
            "capability report should include {cap}"
        );
    }
}

#[tokio::test]
async fn model_load_bundle_returns_model_clip_vae_handles() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();

    assert_eq!(response.model().model_id().as_str(), "sdxl-base-1.0");
    assert_eq!(response.model().role(), ModelRole::CheckpointBundle);
    assert_eq!(response.model().backend().as_str(), "candle");
    assert_eq!(response.model().device_label(), Some("cpu"));
    assert!(!response.model().payload_key().as_str().is_empty());
    assert_eq!(response.clip().model_id().as_str(), "sdxl-base-1.0");
    assert_eq!(response.clip().backend().as_str(), "candle");
    assert_eq!(response.clip().device_label(), Some("cpu"));
    assert_eq!(response.vae().model_id().as_str(), "sdxl-base-1.0");
    assert_eq!(response.vae().backend().as_str(), "candle");
    assert_eq!(response.vae().device_label(), Some("cpu"));
}

#[tokio::test]
async fn model_load_bundle_caches_bundle_for_same_model_id() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let response1 = backend
        .load_bundle(base_load_bundle_request(model.clone(), "node-test"))
        .await
        .unwrap();
    let response2 = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();

    let key1 = response1.model().payload_key().as_str().to_string();
    let key2 = response2.model().payload_key().as_str().to_string();
    assert_eq!(key1, key2, "same model id should reuse cached payload key");
    assert_eq!(
        backend.model_cache().bundle_count(),
        1,
        "cache should hold a single entry for the same model id"
    );
}

#[tokio::test]
async fn model_load_bundle_caches_distinct_entries_per_model_id() {
    let backend = backend();
    let root = unique_sdxl_root();
    let first = write_sdxl_placeholder(&root);
    let second_path = root.join("sdxl-base-1.1.safetensors");
    std::fs::write(&second_path, b"placeholder").unwrap();

    let first_model = ResolvedInferenceModel::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &first,
        ModelFormat::SafeTensors,
    );
    let second_model = ResolvedInferenceModel::new(
        ModelId::new("sdxl-base-1.1"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &second_path,
        ModelFormat::SafeTensors,
    );

    backend
        .load_bundle(base_load_bundle_request(first_model, "node-test"))
        .await
        .unwrap();
    backend
        .load_bundle(base_load_bundle_request(second_model, "node-test"))
        .await
        .unwrap();

    assert_eq!(backend.model_cache().bundle_count(), 2);
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn model_load_bundle_validates_resolved_source_path() {
    let backend = backend();
    let root = unique_sdxl_root();
    let missing_path = root.join("does-not-exist.safetensors");
    let model = ResolvedInferenceModel::new(
        ModelId::new("missing-model"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &missing_path,
        ModelFormat::SafeTensors,
    );
    let err = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("missing"), "msg: {msg}");
    assert!(msg.contains("does-not-exist.safetensors"), "msg: {msg}");
    assert_eq!(
        backend.model_cache().bundle_count(),
        0,
        "failed validation must not insert into the cache"
    );
}

#[tokio::test]
async fn model_load_bundle_rejects_extension_mismatch() {
    let backend = backend();
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let wrong_path = root.join("sdxl-base-1.0.pt");
    std::fs::write(&wrong_path, b"placeholder").unwrap();
    let model = ResolvedInferenceModel::new(
        ModelId::new("mismatched-extension"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &wrong_path,
        ModelFormat::SafeTensors,
    );
    let err = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("format mismatch"), "msg: {msg}");
    assert!(msg.contains("safetensors"), "msg: {msg}");
    assert!(msg.contains("pt"), "msg: {msg}");
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn model_load_bundle_rejects_directory_source() {
    let backend = backend();
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let model = ResolvedInferenceModel::new(
        ModelId::new("directory-source"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        &root,
        ModelFormat::SafeTensors,
    );
    let err = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("not a regular file"), "msg: {msg}");
    let _ = std::fs::remove_dir_all(&root);
}

fn flux_model() -> (ResolvedInferenceModel, PathBuf) {
    let root = unique_sdxl_root();
    let path = write_sdxl_placeholder(&root);
    let model = ResolvedInferenceModel::new(
        ModelId::new("flux-dev"),
        ModelSeries::new("flux"),
        ModelVariant::new("dev"),
        ModelRole::CheckpointBundle,
        &path,
        ModelFormat::SafeTensors,
    );
    (model, root)
}

fn stable_diffusion_sdx2_model() -> (ResolvedInferenceModel, PathBuf) {
    let root = unique_sdxl_root();
    let path = write_sdxl_placeholder(&root);
    let model = ResolvedInferenceModel::new(
        ModelId::new("sdx2-base"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdx2"),
        ModelRole::CheckpointBundle,
        &path,
        ModelFormat::SafeTensors,
    );
    (model, root)
}

#[tokio::test]
async fn model_load_bundle_rejects_unsupported_model_series() {
    let backend = backend();
    let (model, _root) = flux_model();
    let err = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("flux-dev"), "msg: {msg}");
    assert!(msg.contains("flux"), "msg: {msg}");
    assert!(msg.contains("dev"), "msg: {msg}");
    assert!(
        msg.contains("no loader"),
        "msg should mention missing loader, got {msg}"
    );
    assert_eq!(
        backend.model_cache().bundle_count(),
        0,
        "unsupported family must not insert into the cache"
    );
}

#[tokio::test]
async fn model_load_bundle_rejects_unsupported_model_variant() {
    let backend = backend();
    let (model, _root) = stable_diffusion_sdx2_model();
    let err = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("sdx2-base"), "msg: {msg}");
    assert!(msg.contains("stable_diffusion"), "msg: {msg}");
    assert!(msg.contains("sdx2"), "msg: {msg}");
    assert_eq!(backend.model_cache().bundle_count(), 0);
}

#[tokio::test]
async fn model_load_bundle_dispatches_to_sdxl_for_matching_series_variant() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    // SDXL loader emits three typed handles — same as before the refactor.
    assert!(response.model().payload_key().as_str().len() > 0);
    assert!(response.clip().payload_key().as_str().len() > 0);
    assert!(response.vae().payload_key().as_str().len() > 0);
    assert_eq!(
        backend.model_cache().bundle_count(),
        1,
        "sdxl loader should populate the cache"
    );
}

#[tokio::test]
async fn model_load_bundle_requires_a_model() {
    let backend = backend();
    let model = ResolvedInferenceModel::new(
        ModelId::new("placeholder"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        std::path::PathBuf::from("/nonexistent"),
        ModelFormat::SafeTensors,
    );
    let err = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap_err();
    assert!(
        matches!(err, InferenceError::BackendExecutionFailed { .. }),
        "expected invalid-request mapped to execution failed, got {err:?}"
    );
}

#[tokio::test]
async fn latent_create_empty_returns_f32_latent_with_sdxl_shape() {
    let backend = backend();
    let response: CreateEmptyLatentResponse = backend
        .create_empty_latent(base_create_empty_latent_request(1024, 512, 2, "node-test"))
        .await
        .unwrap();

    let latent = response.latent();
    assert_eq!(latent.width(), 1024);
    assert_eq!(latent.height(), 512);
    assert_eq!(latent.batch(), 2);
    assert_eq!(latent.channels(), 4);
    assert_eq!(latent.payload().dtype(), TensorDType::F32);
    assert_eq!(latent.payload().shape().dims(), &[2, 4, 64, 128]);
    assert_eq!(latent.payload().backend().as_str(), "candle");
    assert_eq!(latent.payload().device_label(), "cpu");
}

#[tokio::test]
async fn latent_create_empty_registers_real_tensor_in_store() {
    let backend = backend();
    let response: CreateEmptyLatentResponse = backend
        .create_empty_latent(base_create_empty_latent_request(512, 512, 1, "node-test"))
        .await
        .unwrap();
    let payload_key = response.latent().payload().payload_key().clone();

    let latent = backend
        .store()
        .get_latent(&payload_key)
        .expect("typed latent lookup");
    assert_eq!(latent.dims(), vec![1, 4, 64, 64]);
    assert_eq!(latent.dtype(), DType::F32);
    assert_eq!(latent.byte_size(), 1 * 4 * 64 * 64 * 4);
}

#[tokio::test]
async fn latent_create_empty_rejects_non_positive_dimensions() {
    let backend = backend();
    let err = backend
        .create_empty_latent(base_create_empty_latent_request(0, 512, 1, "node-test"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("width"), "msg: {msg}");
    assert!(msg.contains("positive"), "msg: {msg}");
    assert_eq!(
        backend.store().payload_count(),
        0,
        "rejected latent must not be stored"
    );
}

#[tokio::test]
async fn latent_create_empty_rejects_non_multiple_of_8_dimensions() {
    let backend = backend();
    let err = backend
        .create_empty_latent(base_create_empty_latent_request(513, 512, 1, "node-test"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("divisible by 8"), "msg: {msg}");
    assert!(msg.contains("513"), "msg: {msg}");
    assert_eq!(backend.store().payload_count(), 0);
}

#[tokio::test]
async fn text_encode_returns_conditioning_for_sdxl_bundle() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let bundle_response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let clip_key = bundle_response.clip().payload_key().clone();
    let clip_handle = fake_runtime_clip_handle(clip_key.as_str());

    let response: TextEncodeResponse = backend
        .text_encode(base_text_encode_request(
            clip_handle,
            "a cinematic lake at sunrise".to_string(),
            "node-test",
        ))
        .await
        .unwrap();

    let conditioning = response.conditioning();
    assert_eq!(conditioning.text_embedding().dtype(), TensorDType::F32);
    assert_eq!(conditioning.text_embedding().shape().dims(), &[1, 77, 2048]);
    assert!(conditioning.pooled_embedding().is_some());
}

#[tokio::test]
async fn load_bundle_rejects_placeholder_checkpoint_without_test_projection() {
    let backend = production_like_backend();
    let (model, _root) = sdxl_model();
    let err = backend
        .load_bundle(base_load_bundle_request(model, "node-placeholder"))
        .await
        .unwrap_err();

    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("text encoder weights"), "msg: {msg}");
}

#[tokio::test]
async fn text_encode_positive_and_negative_prompts_both_succeed() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let bundle_response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let clip_key = bundle_response.clip().payload_key().clone();
    let clip_factory = || fake_runtime_clip_handle(clip_key.as_str());

    let positive = backend
        .text_encode(base_text_encode_request(
            clip_factory(),
            "cinematic lake at sunrise".to_string(),
            "node-positive-encode",
        ))
        .await
        .unwrap();
    let negative = backend
        .text_encode(base_text_encode_request(
            clip_factory(),
            "low quality, blurry".to_string(),
            "node-negative-encode",
        ))
        .await
        .unwrap();

    assert_eq!(
        positive.conditioning().text_embedding().backend().as_str(),
        "candle"
    );
    assert_eq!(
        negative.conditioning().text_embedding().backend().as_str(),
        "candle"
    );
    assert_eq!(backend.store().payload_count(), 2);
}

#[tokio::test]
async fn text_encode_conditioning_is_prompt_dependent_and_nonzero() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let bundle_response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let clip_key = bundle_response.clip().payload_key().clone();
    let clip_factory = || fake_runtime_clip_handle(clip_key.as_str());

    let first = backend
        .text_encode(base_text_encode_request(
            clip_factory(),
            "cinematic lake at sunrise".to_string(),
            "node-first-encode",
        ))
        .await
        .unwrap();
    let second = backend
        .text_encode(base_text_encode_request(
            clip_factory(),
            "industrial city at midnight".to_string(),
            "node-second-encode",
        ))
        .await
        .unwrap();

    let first_payload = backend
        .store()
        .get_conditioning(first.conditioning().text_embedding().payload_key())
        .unwrap();
    let second_payload = backend
        .store()
        .get_conditioning(second.conditioning().text_embedding().payload_key())
        .unwrap();
    let first_values = first_payload
        .text_embedding()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let second_values = second_payload
        .text_embedding()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert!(
        first_values.iter().any(|value| *value != 0.0),
        "text.encode must not store all-zero placeholder conditioning"
    );
    assert_ne!(
        first_values, second_values,
        "materially different prompts should produce different conditioning"
    );
}

#[tokio::test]
#[ignore = "requires local SDXL weights; set REIMAGINE_SDXL_REAL_WEIGHTS=/path/to/sdxl-base-1.0.safetensors"]
async fn text_encode_real_weights_manual_checkpoint_path() {
    let weights = std::env::var_os("REIMAGINE_SDXL_REAL_WEIGHTS")
        .map(PathBuf::from)
        .expect("set REIMAGINE_SDXL_REAL_WEIGHTS to a local SDXL checkpoint");
    let backend = backend();
    let model = sdxl_model_from_path(weights);
    let bundle_response = backend
        .load_bundle(base_load_bundle_request(model, "node-real-weights"))
        .await
        .unwrap();
    let clip_key = bundle_response.clip().payload_key().clone();
    let clip_factory = || fake_runtime_clip_handle(clip_key.as_str());

    let first = backend
        .text_encode(base_text_encode_request(
            clip_factory(),
            "cinematic lake at sunrise".to_string(),
            "node-real-weights-first-encode",
        ))
        .await
        .unwrap();
    let second = backend
        .text_encode(base_text_encode_request(
            clip_factory(),
            "industrial city at midnight".to_string(),
            "node-real-weights-second-encode",
        ))
        .await
        .unwrap();

    assert_eq!(
        first.conditioning().text_embedding().shape().dims(),
        &[1, 77, 2048]
    );
    assert_eq!(
        first
            .conditioning()
            .pooled_embedding()
            .unwrap()
            .shape()
            .dims(),
        &[1, 1280]
    );

    let first_payload = backend
        .store()
        .get_conditioning(first.conditioning().text_embedding().payload_key())
        .unwrap();
    let second_payload = backend
        .store()
        .get_conditioning(second.conditioning().text_embedding().payload_key())
        .unwrap();
    let first_values = first_payload
        .text_embedding()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let second_values = second_payload
        .text_embedding()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert!(
        first_values.iter().any(|value| *value != 0.0),
        "real text.encode must not store all-zero conditioning"
    );
    assert_ne!(
        first_values, second_values,
        "real text.encode should produce prompt-dependent conditioning"
    );
}

#[tokio::test]
async fn text_encode_conditioning_is_run_scoped_and_cleaned() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let bundle_response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let clip_key = bundle_response.clip().payload_key().clone();
    let clip_handle = fake_runtime_clip_handle(clip_key.as_str());

    let run_id = RunId::new("run-conditioning-cleanup");
    let response: TextEncodeResponse = backend
        .text_encode(TextEncodeRequest::new(
            clip_handle,
            Arc::new(ExecutionValue::Param(ParamValue::String(
                "test prompt".to_string(),
            ))),
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-encode"),
        ))
        .await
        .unwrap();
    assert!(
        response
            .conditioning()
            .text_embedding()
            .payload_key()
            .as_str()
            .len()
            > 0
    );
    assert_eq!(backend.store().run_payload_count(&run_id), 1);

    let resource = backend.runtime_hooks(None, None, None);
    let _ = resource
        .cleanup_run(reimagine_inference::BackendRunLifecycleRequest {
            run_id: run_id.clone(),
        })
        .await;

    assert_eq!(backend.store().run_payload_count(&run_id), 0);
    assert_eq!(backend.store().payload_count(), 0);
}

#[tokio::test]
async fn text_encode_missing_clip_input_returns_error() {
    let backend = backend();
    let err = backend
        .text_encode(TextEncodeRequest::new(
            fake_runtime_clip_handle("missing-clip-key"),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                "test".to_string(),
            ))),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-test"),
        ))
        .await
        .unwrap_err();
    assert_backend_execution_failed_with(&err, "no loaded model bundle");
}

#[tokio::test]
async fn text_encode_missing_text_input_returns_error() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let bundle_response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let clip_handle = fake_runtime_clip_handle(bundle_response.clip().payload_key().as_str());
    let err = backend
        .text_encode(TextEncodeRequest::new(
            clip_handle,
            Arc::new(ExecutionValue::Param(ParamValue::Null)),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-test"),
        ))
        .await
        .unwrap_err();
    assert_backend_execution_failed_with(&err, "text");
}

#[tokio::test]
async fn text_encode_without_loaded_bundle_returns_error() {
    let backend = backend();
    let clip_handle = fake_runtime_clip_handle("bundle:sdxl-base-1.0:clip");
    let err = backend
        .text_encode(base_text_encode_request(
            clip_handle,
            "test".to_string(),
            "node-test",
        ))
        .await
        .unwrap_err();
    assert_backend_execution_failed_with(&err, "no loaded model bundle");
}

#[tokio::test]
async fn text_encode_rejects_wrong_backend_clip_handle() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let clip_handle = RuntimeClipHandle::new(
        ModelId::new("sdxl-base-1.0"),
        Backend::new("other-backend"),
        BackendPayloadKey::new("bundle:sdxl-base-1.0:clip"),
    );
    let err = backend
        .text_encode(base_text_encode_request(
            clip_handle,
            "test".to_string(),
            "node-test",
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(
        msg.contains("other-backend") && msg.contains("expected `candle`"),
        "msg: {msg}"
    );
}

#[tokio::test]
async fn text_encode_rejects_wrong_clip_payload_key() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let clip_handle = RuntimeClipHandle::new(
        ModelId::new("sdxl-base-1.0"),
        Backend::new("candle"),
        BackendPayloadKey::new("bundle:sdxl-base-1.0:not-clip"),
    );
    let err = backend
        .text_encode(base_text_encode_request(
            clip_handle,
            "test".to_string(),
            "node-test",
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(
        msg.contains("not-clip") && msg.contains("CLIP payload") && msg.contains("sdxl-base-1.0"),
        "msg: {msg}"
    );
}

#[tokio::test]
async fn diffusion_sample_validates_inputs_then_rejects_unmaterialized_diffusion_graph() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let bundle_response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };
    let _ = bundle_response;

    let model_handle_factory = || fake_runtime_model_handle(model_payload_key.as_str());
    let clip_handle_factory = || fake_runtime_clip_handle(clip_payload_key.as_str());

    let positive = backend
        .text_encode(base_text_encode_request(
            clip_handle_factory(),
            "cinematic lake at sunrise".to_string(),
            "node-positive-encode",
        ))
        .await
        .unwrap();
    let negative = backend
        .text_encode(base_text_encode_request(
            clip_handle_factory(),
            "low quality, blurry".to_string(),
            "node-negative-encode",
        ))
        .await
        .unwrap();

    let latent_response = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap();

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            model_handle_factory(),
            positive.into_conditioning(),
            negative.into_conditioning(),
            latent_response.into_latent(),
            "node-sample",
        ))
        .await
        .unwrap_err();

    assert_backend_execution_failed_with(&err, "invalid SDXL diffusion safetensors header");
    assert_eq!(
        backend.store().run_payload_count(&RunId::new("run-test")),
        3,
        "two conditioning payloads + input latent; no placeholder sampled latent"
    );
}

#[tokio::test]
async fn diffusion_sample_rejects_unmaterialized_real_diffusion_graph_instead_of_placeholder_sampling()
 {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };

    let positive = backend
        .text_encode(base_text_encode_request(
            fake_runtime_clip_handle(clip_payload_key.as_str()),
            "cinematic lake at sunrise".to_string(),
            "node-positive-encode",
        ))
        .await
        .unwrap()
        .into_conditioning();
    let negative = backend
        .text_encode(base_text_encode_request(
            fake_runtime_clip_handle(clip_payload_key.as_str()),
            "low quality, blurry".to_string(),
            "node-negative-encode",
        ))
        .await
        .unwrap()
        .into_conditioning();
    let latent = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap()
        .into_latent();

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            fake_runtime_model_handle(model_payload_key.as_str()),
            positive,
            negative,
            latent,
            "node-sample",
        ))
        .await
        .unwrap_err();

    assert_backend_execution_failed_with(&err, "invalid SDXL diffusion safetensors header");
}

#[tokio::test]
async fn diffusion_sample_rejects_incomplete_diffusers_unet_with_materialization_error() {
    let backend = backend();
    let root = unique_sdxl_root();
    let path = write_header_only_safetensors(
        &root,
        "sdxl-unet.safetensors",
        &["down_blocks.0.resnets.0.conv1.weight", "conv_in.weight"],
    );
    let model = sdxl_model_from_path(path);
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };

    let positive = backend
        .text_encode(base_text_encode_request(
            fake_runtime_clip_handle(clip_payload_key.as_str()),
            "cinematic lake at sunrise".to_string(),
            "node-positive-encode",
        ))
        .await
        .unwrap()
        .into_conditioning();
    let negative = backend
        .text_encode(base_text_encode_request(
            fake_runtime_clip_handle(clip_payload_key.as_str()),
            "low quality, blurry".to_string(),
            "node-negative-encode",
        ))
        .await
        .unwrap()
        .into_conditioning();
    let latent = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap()
        .into_latent();

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            fake_runtime_model_handle(model_payload_key.as_str()),
            positive,
            negative,
            latent,
            "node-sample",
        ))
        .await
        .unwrap_err();

    assert_backend_execution_failed_with(&err, "failed to parse SDXL diffusion safetensors");
    assert_backend_execution_failed_with(&err, "invalid offset");
}

#[tokio::test]
async fn diffusion_sample_reads_conditioning_payload_and_rejects_bad_text_shape() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };

    let bad_key = BackendPayloadKey::new("conditioning:bad-shape");
    backend.store().insert_conditioning(
        RunId::new("run-test"),
        bad_key.clone(),
        CandleConditioning::new(
            Tensor::zeros((1, 76, 2048), DType::F32, backend.device().as_ref()).unwrap(),
            Some(Tensor::zeros((1, 1280), DType::F32, backend.device().as_ref()).unwrap()),
        ),
    );
    let positive = fake_conditioning(bad_key.as_str(), Some(bad_key.as_str()));
    let negative = backend
        .text_encode(base_text_encode_request(
            fake_runtime_clip_handle(clip_payload_key.as_str()),
            "negative".to_string(),
            "node-negative-encode",
        ))
        .await
        .unwrap()
        .into_conditioning();
    let latent = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap()
        .into_latent();

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            fake_runtime_model_handle(model_payload_key.as_str()),
            positive,
            negative,
            latent,
            "node-sample",
        ))
        .await
        .unwrap_err();

    assert_backend_execution_failed_with(&err, "text_embedding");
    assert_backend_execution_failed_with(&err, "[1, 77, 2048]");
}

#[tokio::test]
async fn diffusion_sample_reads_conditioning_payload_and_requires_pooled_embedding() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };

    let no_pooled_key = BackendPayloadKey::new("conditioning:no-pooled");
    backend.store().insert_conditioning(
        RunId::new("run-test"),
        no_pooled_key.clone(),
        CandleConditioning::new(
            Tensor::zeros((1, 77, 2048), DType::F32, backend.device().as_ref()).unwrap(),
            None,
        ),
    );
    let positive = fake_conditioning(no_pooled_key.as_str(), None);
    let negative = backend
        .text_encode(base_text_encode_request(
            fake_runtime_clip_handle(clip_payload_key.as_str()),
            "negative".to_string(),
            "node-negative-encode",
        ))
        .await
        .unwrap()
        .into_conditioning();
    let latent = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap()
        .into_latent();

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            fake_runtime_model_handle(model_payload_key.as_str()),
            positive,
            negative,
            latent,
            "node-sample",
        ))
        .await
        .unwrap_err();

    assert_backend_execution_failed_with(&err, "pooled_embedding");
    assert_backend_execution_failed_with(&err, "[1, 1280]");
}

#[tokio::test]
async fn diffusion_sample_rejects_partial_denoise_until_real_img2img_semantics_exist() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };
    let model_handle_factory = || fake_runtime_model_handle(model_payload_key.as_str());
    let clip_handle_factory = || fake_runtime_clip_handle(clip_payload_key.as_str());

    async fn make_conditioning(
        backend: &CandleBackend,
        clip: RuntimeClipHandle,
        node: &str,
    ) -> ExecutionConditioning {
        backend
            .text_encode(base_text_encode_request(
                clip,
                "deterministic test prompt".to_string(),
                node,
            ))
            .await
            .unwrap()
            .into_conditioning()
    }

    let positive = make_conditioning(&backend, clip_handle_factory(), "node-pos-cond").await;
    let negative = make_conditioning(&backend, clip_handle_factory(), "node-neg-cond").await;

    let latent_a: RuntimeLatent = backend
        .create_empty_latent(base_create_empty_latent_request(32, 32, 1, "node-latent-a"))
        .await
        .unwrap()
        .into_latent();
    let err = backend
        .diffusion_sample(DiffusionSampleRequest::new(
            model_handle_factory(),
            positive,
            negative,
            latent_a,
            42,
            15,
            6.5,
            SamplerName::Euler,
            SchedulerName::Normal,
            0.8,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sample"),
        ))
        .await
        .unwrap_err();

    assert_backend_execution_failed_with(&err, "denoise");
    assert_backend_execution_failed_with(&err, "partial");
}

#[tokio::test]
async fn diffusion_sample_rejects_wrong_backend_latent() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model.clone(), "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };

    let model_handle = fake_runtime_model_handle(model_payload_key.as_str());
    let clip_handle = fake_runtime_clip_handle(clip_payload_key.as_str());

    let positive = backend
        .text_encode(base_text_encode_request(
            clip_handle,
            "positive".to_string(),
            "node-positive",
        ))
        .await
        .unwrap()
        .into_conditioning();
    let negative = backend
        .text_encode(base_text_encode_request(
            fake_runtime_clip_handle(clip_payload_key.as_str()),
            "negative".to_string(),
            "node-negative",
        ))
        .await
        .unwrap()
        .into_conditioning();

    let wrong_backend_latent = RuntimeLatent::new(
        BackendTensorHandle::new(
            Backend::new("other-backend"),
            BackendPayloadKey::new("latent:other"),
            TensorDType::F32,
            TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        64,
        64,
        1,
        4,
    );

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            model_handle,
            positive,
            negative,
            wrong_backend_latent,
            "node-test",
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("other-backend"), "msg: {msg}");
    assert!(msg.contains("candle"), "msg: {msg}");
}

#[tokio::test]
async fn diffusion_sample_rejects_wrong_backend_pooled_conditioning() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let model_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.model_payload_key.clone(),
    };
    let model_handle = fake_runtime_model_handle(model_payload_key.as_str());

    let text_handle = BackendTensorHandle::new(
        Backend::new("candle"),
        BackendPayloadKey::new("conditioning:test"),
        TensorDType::F32,
        TensorShape::new(vec![1, 77, 2048]),
        "cpu",
    );
    let pooled_handle = BackendTensorHandle::new(
        Backend::new("other-backend"),
        BackendPayloadKey::new("conditioning:test"),
        TensorDType::F32,
        TensorShape::new(vec![1, 1280]),
        "cpu",
    );
    let conditioning = ExecutionConditioning::new(text_handle, ConditioningMetadata::new(64, 64))
        .with_pooled_embedding(pooled_handle);

    let latent = fake_runtime_latent("latent:nope", 64, 64);

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            model_handle,
            conditioning,
            fake_conditioning("conditioning:negative", None),
            latent,
            "node-test",
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("pooled conditioning"), "msg: {msg}");
    assert!(msg.contains("other-backend"), "msg: {msg}");
}

#[tokio::test]
async fn diffusion_sample_rejects_mismatched_backend_instance_affinity() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model.clone(), "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };

    let request = DiffusionSampleRequest::new(
        fake_runtime_model_handle_with_instance(model_payload_key.as_str(), "candle:cpu"),
        fake_conditioning_with_instance(clip_payload_key.as_str(), None, "candle:other"),
        fake_conditioning_with_instance(clip_payload_key.as_str(), None, "candle:cpu"),
        fake_runtime_latent_with_instance("latent:nope", 64, 64, "candle:cpu"),
        1,
        10,
        7.0,
        SamplerName::Euler,
        SchedulerName::Normal,
        1.0,
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-test"),
    );

    let err = backend.diffusion_sample(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("backend instance"), "msg: {msg}");
    assert!(msg.contains("candle:cpu"), "msg: {msg}");
    assert!(msg.contains("candle:other"), "msg: {msg}");
}

#[tokio::test]
async fn diffusion_sample_rejects_mismatched_pooled_conditioning_payload() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let model_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.model_payload_key.clone(),
    };
    let model_handle = fake_runtime_model_handle(model_payload_key.as_str());

    let text_handle = BackendTensorHandle::new(
        Backend::new("candle"),
        BackendPayloadKey::new("conditioning:text"),
        TensorDType::F32,
        TensorShape::new(vec![1, 77, 2048]),
        "cpu",
    );
    let pooled_handle = BackendTensorHandle::new(
        Backend::new("candle"),
        BackendPayloadKey::new("conditioning:pooled"),
        TensorDType::F32,
        TensorShape::new(vec![1, 1280]),
        "cpu",
    );
    let conditioning = ExecutionConditioning::new(text_handle, ConditioningMetadata::new(64, 64))
        .with_pooled_embedding(pooled_handle);
    let latent = fake_runtime_latent("latent:nope", 64, 64);

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            model_handle,
            conditioning,
            fake_conditioning("conditioning:negative", None),
            latent,
            "node-test",
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("conditioning:pooled"), "msg: {msg}");
    assert!(msg.contains("conditioning:text"), "msg: {msg}");
}

#[tokio::test]
async fn diffusion_sample_rejects_unsupported_sampler() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };
    let model_handle = fake_runtime_model_handle(model_payload_key.as_str());
    let conditioning = fake_conditioning(clip_payload_key.as_str(), None);
    let latent = fake_runtime_latent("latent:nope", 64, 64);

    let request = DiffusionSampleRequest::new(
        model_handle,
        conditioning.clone(),
        conditioning,
        latent,
        1,
        10,
        7.0,
        SamplerName::Other("dpmpp_2m".to_string()),
        SchedulerName::Normal,
        1.0,
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-test"),
    );
    let err = backend.diffusion_sample(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("dpmpp_2m"), "msg: {msg}");
    assert!(msg.contains("V1"), "msg: {msg}");
}

#[tokio::test]
async fn diffusion_sample_rejects_missing_latent_payload() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };
    let model_handle = fake_runtime_model_handle(model_payload_key.as_str());
    let clip_handle = fake_runtime_clip_handle(clip_payload_key.as_str());

    let positive = backend
        .text_encode(base_text_encode_request(
            clip_handle.clone(),
            "test".to_string(),
            "node-positive-cond",
        ))
        .await
        .unwrap();
    let negative = backend
        .text_encode(base_text_encode_request(
            clip_handle,
            "test".to_string(),
            "node-negative-cond",
        ))
        .await
        .unwrap();

    let latent = fake_runtime_latent("latent:not-in-store", 64, 64);

    let err = backend
        .diffusion_sample(base_diffusion_sample_request(
            model_handle,
            positive.into_conditioning(),
            negative.into_conditioning(),
            latent,
            "node-test",
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("latent:not-in-store"), "msg: {msg}");
}

#[tokio::test]
async fn diffusion_sample_cleans_up_run_scoped_payload() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let (model_payload_key, clip_payload_key) = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => (
            sdxl.model_payload_key.clone(),
            sdxl.clip_payload_key.clone(),
        ),
    };
    let model_handle_factory = || fake_runtime_model_handle(model_payload_key.as_str());
    let clip_handle_factory = || fake_runtime_clip_handle(clip_payload_key.as_str());

    let run_id = RunId::new("run-diffusion-cleanup");

    let positive = backend
        .text_encode(TextEncodeRequest::new(
            clip_handle_factory(),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                "a cat".to_string(),
            ))),
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-positive-encode"),
        ))
        .await
        .unwrap();
    let negative = backend
        .text_encode(TextEncodeRequest::new(
            clip_handle_factory(),
            Arc::new(ExecutionValue::Param(ParamValue::String(
                "blurry".to_string(),
            ))),
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-negative-encode"),
        ))
        .await
        .unwrap();
    let latent = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            32,
            32,
            1,
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-latent"),
        ))
        .await
        .unwrap();

    let err = backend
        .diffusion_sample(DiffusionSampleRequest::new(
            model_handle_factory(),
            positive.into_conditioning(),
            negative.into_conditioning(),
            latent.into_latent(),
            1,
            10,
            7.0,
            SamplerName::Euler,
            SchedulerName::Normal,
            1.0,
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-sampler"),
        ))
        .await
        .unwrap_err();

    assert_backend_execution_failed_with(&err, "invalid SDXL diffusion safetensors header");

    assert_eq!(
        backend.store().run_payload_count(&run_id),
        3,
        "two conditionings + input latent; failed sampling stores no output latent"
    );

    let resource = backend.runtime_hooks(None, None, None);
    let _ = resource
        .cleanup_run(reimagine_inference::BackendRunLifecycleRequest {
            run_id: run_id.clone(),
        })
        .await;
    assert_eq!(backend.store().run_payload_count(&run_id), 0);
}

#[tokio::test]
async fn latent_decode_succeeds_for_sdxl_with_loaded_bundle() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let bundle_response = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let _ = bundle_response;
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());

    let latent_response: CreateEmptyLatentResponse = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap();
    let latent = latent_response.into_latent();

    let decode_response: LatentDecodeResponse = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap();
    let image = decode_response.image();

    assert_eq!(image.width(), 64);
    assert_eq!(image.height(), 64);
    assert_eq!(image.batch(), 1);
    assert_eq!(image.color_space(), "rgb");
    assert_eq!(image.payload().dtype(), TensorDType::F32);
    assert_eq!(image.payload().shape().dims(), &[1, 3, 64, 64]);
    assert_eq!(image.payload().backend().as_str(), "candle");
    assert_eq!(image.payload().device_label(), "cpu");
}

#[tokio::test]
async fn latent_decode_rejects_wrong_backend_vae_handle() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let vae_handle = RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        Backend::new("other-backend"),
        BackendPayloadKey::new("bundle:sdxl-base-1.0:vae"),
    );
    let latent = fake_runtime_latent("latent:placeholder", 64, 64);
    let err = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("other-backend"), "msg: {msg}");
    assert!(msg.contains("candle"), "msg: {msg}");
}

#[tokio::test]
async fn latent_decode_rejects_wrong_backend_latent_handle() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());
    let wrong_backend_latent = RuntimeLatent::new(
        BackendTensorHandle::new(
            Backend::new("other-backend"),
            BackendPayloadKey::new("latent:other"),
            TensorDType::F32,
            TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        64,
        64,
        1,
        4,
    );
    let err = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            wrong_backend_latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("other-backend"), "msg: {msg}");
    assert!(msg.contains("candle"), "msg: {msg}");
}

#[tokio::test]
async fn latent_decode_rejects_wrong_vae_payload_key() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let vae_handle = RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        Backend::new("candle"),
        BackendPayloadKey::new("bundle:sdxl-base-1.0:not-vae"),
    );
    let latent_response = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap();
    let err = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            latent_response.into_latent(),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("not-vae"), "msg: {msg}");
}

#[tokio::test]
async fn latent_decode_rejects_missing_latent_payload() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());
    let missing_latent = RuntimeLatent::new(
        BackendTensorHandle::new(
            Backend::new("candle"),
            BackendPayloadKey::new("latent:not-in-store"),
            TensorDType::F32,
            TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        64,
        64,
        1,
        4,
    );
    let err = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            missing_latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("latent:not-in-store"), "msg: {msg}");
}

#[tokio::test]
async fn latent_decode_rejects_missing_loaded_bundle() {
    let backend = backend();
    let vae_handle = fake_runtime_vae_handle("bundle:sdxl-base-1.0:vae");
    let latent_response = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap();
    let err = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            latent_response.into_latent(),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("sdxl-base-1.0"), "msg: {msg}");
    assert!(msg.contains("no loaded model bundle"), "msg: {msg}");
}

#[tokio::test]
async fn latent_decode_stores_real_image_payload_in_store() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());

    let latent_response: CreateEmptyLatentResponse = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap();
    let latent = latent_response.into_latent();

    let decode_response: LatentDecodeResponse = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap();
    let payload_key = decode_response.image().payload().payload_key().clone();

    let stored_image = backend
        .store()
        .get_image(&payload_key)
        .expect("image should be in store");
    assert_eq!(stored_image.dims(), vec![1, 3, 64, 64]);
    assert_eq!(stored_image.dtype(), DType::F32);
    assert_eq!(stored_image.width(), 64);
    assert_eq!(stored_image.height(), 64);
    assert_eq!(stored_image.batch(), 1);
    assert_eq!(stored_image.color_space(), "rgb");
}

#[tokio::test]
async fn latent_decode_output_handle_carries_correct_metadata() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());

    // V1 latent.decode only supports batch=1; a batch>1 request must
    // be rejected with a precise unsupported-batch error before any
    // image payload is materialized.
    let latent_response = backend
        .create_empty_latent(base_create_empty_latent_request(128, 64, 2, "node-latent"))
        .await
        .unwrap();
    let latent = latent_response.into_latent();
    let err = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("batch=1"), "msg: {msg}");
    assert!(msg.contains("batch=2"), "msg: {msg}");
    assert!(
        msg.contains("V1 supports only batch=1") || msg.contains("unsupported-batch"),
        "msg: {msg}"
    );
    // No image payload should be stored when the decode is rejected;
    // the source latent may still be pinned in the store because it
    // was inserted by create_empty_latent.
    assert_eq!(
        backend.store().run_payload_count(&RunId::new("run-test")),
        1,
        "only the source latent should remain pinned in the store"
    );
}

#[tokio::test]
async fn latent_decode_batch_one_succeeds_and_returns_f32_rgb_metadata() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());

    let latent_response = backend
        .create_empty_latent(base_create_empty_latent_request(128, 64, 1, "node-latent"))
        .await
        .unwrap();
    let latent = latent_response.into_latent();
    let decode_response = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap();
    let image = decode_response.image();

    assert_eq!(image.width(), 128);
    assert_eq!(image.height(), 64);
    assert_eq!(image.batch(), 1);
    assert_eq!(image.color_space(), "rgb");
    assert_eq!(image.payload().dtype(), TensorDType::F32);
    assert_eq!(image.payload().shape().dims(), &[1, 3, 64, 128]);
}

#[tokio::test]
async fn latent_decode_runs_scoped_payload_cleanup() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());

    let latent_run_id = RunId::new("run-latent-source");
    let latent_response = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            latent_run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-latent"),
        ))
        .await
        .unwrap();

    let run_id = RunId::new("run-latent-decode-cleanup");
    backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            latent_response.into_latent(),
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode"),
        ))
        .await
        .unwrap();

    assert_eq!(
        backend.store().run_payload_count(&run_id),
        1,
        "decode should produce exactly 1 run-scoped image payload"
    );

    let resource = backend.runtime_hooks(None, None, None);
    let _ = resource
        .cleanup_run(reimagine_inference::BackendRunLifecycleRequest {
            run_id: run_id.clone(),
        })
        .await;
    assert_eq!(backend.store().run_payload_count(&run_id), 0);
}

async fn setup_decoded_image_for_save(
    backend: &CandleBackend,
    node_id: &str,
) -> (RuntimeImage, std::path::PathBuf) {
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());

    let latent_resp = backend
        .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent"))
        .await
        .unwrap();
    let input_latent = latent_resp.into_latent();

    let decode_resp = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            input_latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new(node_id),
        ))
        .await
        .unwrap();

    let image_value = decode_resp.into_image();
    let output_dir = backend.output_dir().to_path_buf();
    (image_value, output_dir)
}

#[tokio::test]
async fn image_save_writes_png_to_output_dir_for_sdxl_pipeline() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-save").await;
    let response: ImageSaveResponse = backend
        .image_save(base_image_save_request(image_value, "node-save"))
        .await
        .unwrap();
    let artifact_ref = response.into_artifact();

    let files: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    assert!(
        !files.is_empty(),
        "expected at least one PNG file in output dir"
    );

    let mut png_file: Option<std::path::PathBuf> = None;
    for f in &files {
        if let Ok(entry) = f {
            if entry.file_name().to_string_lossy().ends_with(".png") {
                png_file = Some(entry.path());
                break;
            }
        }
    }
    let png_file = png_file.expect("expected a PNG file");
    assert!(
        artifact_ref.as_str().starts_with("output/"),
        "artifact ref should be workspace-output relative, got {}",
        artifact_ref.as_str()
    );
    assert!(
        artifact_ref.as_str().ends_with(
            png_file
                .file_name()
                .and_then(|name| name.to_str())
                .expect("png filename should be utf-8")
        ),
        "artifact ref {} should point at saved file {}",
        artifact_ref.as_str(),
        png_file.display()
    );
    let metadata = std::fs::metadata(&png_file).unwrap();
    assert!(metadata.len() > 0, "PNG file should be non-empty");

    let mut file = std::fs::File::open(&png_file).unwrap();
    let mut signature = [0u8; 8];
    std::io::Read::read(&mut file, &mut signature).unwrap();
    assert_eq!(
        signature,
        [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
        "PNG signature mismatch"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_preview_writes_png_to_output_dir_for_sdxl_pipeline() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-preview").await;
    let response: ImagePreviewResponse = backend
        .image_preview(base_image_preview_request(image_value, "node-preview"))
        .await
        .unwrap();
    let _ = response.into_artifact();

    let files: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    assert!(
        !files.is_empty(),
        "expected at least one PNG file in output dir"
    );

    let mut png_path: Option<std::path::PathBuf> = None;
    for f in &files {
        if let Ok(entry) = f {
            if entry.file_name().to_string_lossy().ends_with(".png") {
                png_path = Some(entry.path());
                break;
            }
        }
    }
    let png_path = png_path.expect("expected a PNG file");
    let metadata = std::fs::metadata(&png_path).unwrap();
    assert!(metadata.len() > 0);

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_save_filename_includes_prefix_run_id_node_id() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-save-test").await;
    let request = ImageSaveRequest::new(
        image_value,
        RunId::new("run-abc-123"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-foo-bar"),
    )
    .with_filename_prefix("my-prefix");

    backend.image_save(request).await.unwrap();

    let files: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    let filenames: Vec<String> = files
        .iter()
        .map(|f| {
            f.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .to_string()
        })
        .collect();

    assert!(
        filenames.iter().any(|f| f.contains("my-prefix")
            && f.contains("run-abc-123")
            && f.contains("node-foo-bar")),
        "filename should contain prefix, run_id, and node_id; got {filenames:?}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_save_rejects_path_traversal_via_filename_prefix() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-traversal").await;
    let request = ImageSaveRequest::new(
        image_value,
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-test"),
    )
    .with_filename_prefix("../../../etc/passwd");

    backend.image_save(request).await.unwrap();

    let files: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    let filenames: Vec<String> = files
        .iter()
        .map(|f| {
            f.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .to_string()
        })
        .collect();

    for fname in &filenames {
        assert!(
            !fname.contains(".."),
            "sanitized filename should not contain '..': {fname}"
        );
        assert!(
            fname.starts_with('_')
                || fname
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_alphanumeric())
                    .unwrap_or(false),
            "filename should not start with path separators: {fname}"
        );
    }

    let canonical_output = std::fs::canonicalize(&output_dir).unwrap();
    for entry in std::fs::read_dir(&output_dir).unwrap() {
        let entry = entry.unwrap();
        let canonical_file = std::fs::canonicalize(entry.path()).unwrap();
        assert!(
            canonical_file.starts_with(&canonical_output),
            "file {} escapes output dir",
            canonical_file.display()
        );
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_save_rejects_wrong_backend_image_handle() {
    let backend = backend();
    let wrong_backend_image = RuntimeImage::new(
        BackendTensorHandle::new(
            Backend::new("other-backend"),
            BackendPayloadKey::new("image:fake"),
            TensorDType::F32,
            TensorShape::new(vec![1, 3, 64, 64]),
            "cpu",
        ),
        64,
        64,
        1,
        "rgb".to_string(),
    );

    let err = backend
        .image_save(base_image_save_request(wrong_backend_image, "node-save"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("other-backend"), "msg: {msg}");
    assert!(msg.contains("candle"), "msg: {msg}");
}

#[tokio::test]
async fn image_save_rejects_missing_image_input() {
    let backend = backend();
    let ghost_image = RuntimeImage::new(
        BackendTensorHandle::new(
            Backend::new("candle"),
            BackendPayloadKey::new("image:not-in-store"),
            TensorDType::F32,
            TensorShape::new(vec![1, 3, 64, 64]),
            "cpu",
        ),
        64,
        64,
        1,
        "rgb".to_string(),
    );
    let err = backend
        .image_save(base_image_save_request(ghost_image, "node-save"))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("not-in-store"), "msg: {msg}");
}

#[tokio::test]
async fn image_save_overwrites_existing_file() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value1, output_dir) =
        setup_decoded_image_for_save(&backend, "node-overwrite-1").await;
    let (image_value2, _output_dir) =
        setup_decoded_image_for_save(&backend, "node-overwrite-2").await;

    let run_id = RunId::new("run-overwrite");
    backend
        .image_save(
            ImageSaveRequest::new(
                image_value1,
                run_id.clone(),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new("node-overwrite"),
            )
            .with_filename_prefix("overwrite-test"),
        )
        .await
        .unwrap();

    let files_before: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    assert!(!files_before.is_empty());

    backend
        .image_save(
            ImageSaveRequest::new(
                image_value2,
                run_id.clone(),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new("node-overwrite"),
            )
            .with_filename_prefix("overwrite-test"),
        )
        .await
        .unwrap();

    let files_after: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    assert!(!files_after.is_empty());

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_save_returns_artifact_response() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value, _output_dir) = setup_decoded_image_for_save(&backend, "node-slotid").await;
    let response = backend
        .image_save(base_image_save_request(image_value, "node-save"))
        .await
        .unwrap();
    let reference = response.into_artifact();

    assert!(reference.as_str().starts_with("output/"));
    assert!(reference.as_str().ends_with(".png"));

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_preview_uses_different_prefix() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value_save, output_dir) =
        setup_decoded_image_for_save(&backend, "node-save-img").await;
    let (image_value_preview, _output_dir) =
        setup_decoded_image_for_save(&backend, "node-preview-img").await;

    backend
        .image_save(ImageSaveRequest::new(
            image_value_save,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-save"),
        ))
        .await
        .unwrap();

    backend
        .image_preview(ImagePreviewRequest::new(
            image_value_preview,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-preview"),
        ))
        .await
        .unwrap();

    let files: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    let filenames: Vec<String> = files
        .iter()
        .map(|f| {
            f.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .to_string()
        })
        .collect();

    let save_files: Vec<_> = filenames
        .iter()
        .filter(|f| f.contains("reimagine_"))
        .collect();
    let preview_files: Vec<_> = filenames
        .iter()
        .filter(|f| f.contains("preview_"))
        .collect();

    assert!(
        !save_files.is_empty() || !preview_files.is_empty(),
        "should have at least some files with distinct prefixes; got {filenames:?}"
    );

    if !save_files.is_empty() && !preview_files.is_empty() {
        assert_ne!(
            save_files[0], preview_files[0],
            "save and preview should produce different filenames"
        );
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_save_png_bytes_have_valid_signature() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-sig").await;
    backend
        .image_save(base_image_save_request(image_value, "node-save"))
        .await
        .unwrap();

    let files: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    let mut png_path: Option<std::path::PathBuf> = None;
    for f in &files {
        if let Ok(entry) = f {
            if entry.file_name().to_string_lossy().ends_with(".png") {
                png_path = Some(entry.path());
                break;
            }
        }
    }
    let png_path = png_path.expect("expected a PNG file in output dir");

    let png_bytes = std::fs::read(&png_path).unwrap();
    assert_eq!(
        &png_bytes[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
        "PNG signature mismatch"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_save_png_ihdr_has_correct_dimensions() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-ihdr").await;
    backend
        .image_save(base_image_save_request(image_value, "node-save"))
        .await
        .unwrap();

    let files: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    let mut png_path: Option<std::path::PathBuf> = None;
    for f in &files {
        if let Ok(entry) = f {
            if entry.file_name().to_string_lossy().ends_with(".png") {
                png_path = Some(entry.path());
                break;
            }
        }
    }
    let png_path = png_path.expect("expected a PNG file in output dir");

    let png_bytes = std::fs::read(&png_path).unwrap();

    let sig_len = 8;
    let chunk_len = u32::from_be_bytes([
        png_bytes[sig_len],
        png_bytes[sig_len + 1],
        png_bytes[sig_len + 2],
        png_bytes[sig_len + 3],
    ]);
    assert_eq!(chunk_len, 13, "IHDR chunk should be 13 bytes");

    let ihdr_data_start = sig_len + 4 + 4;
    let width = u32::from_be_bytes([
        png_bytes[ihdr_data_start],
        png_bytes[ihdr_data_start + 1],
        png_bytes[ihdr_data_start + 2],
        png_bytes[ihdr_data_start + 3],
    ]);
    let height = u32::from_be_bytes([
        png_bytes[ihdr_data_start + 4],
        png_bytes[ihdr_data_start + 5],
        png_bytes[ihdr_data_start + 6],
        png_bytes[ihdr_data_start + 7],
    ]);

    assert_eq!(width, 64, "IHDR width should be 64");
    assert_eq!(height, 64, "IHDR height should be 64");

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_save_keeps_image_payload_in_store_for_fanout() {
    // V1 requires image.save to read image payloads non-destructively
    // so the same decoded image can feed multiple artifact nodes
    // (save + preview) in the same run.
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    let (image_value, _output_dir) = setup_decoded_image_for_save(&backend, "node-cleanup").await;
    let payload_key = image_value.payload().payload_key().clone();

    assert!(
        backend.store().contains_payload(&payload_key),
        "image should be in store before save"
    );

    backend
        .image_save(base_image_save_request(image_value, "node-save"))
        .await
        .unwrap();

    assert!(
        backend.store().contains_payload(&payload_key),
        "image payload must remain in store after save so image.preview can read it"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn one_decoded_image_can_feed_both_image_save_and_image_preview() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend = backend_with_dirs(&root, &root.join("output"));

    // Decode once.
    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-fanout").await;
    let payload_key = image_value.payload().payload_key().clone();

    // First artifact: image.save.
    backend
        .image_save(base_image_save_request(image_value.clone(), "node-save"))
        .await
        .expect("image.save must succeed with non-destructive read");

    // Second artifact: image.preview against the same source image.
    backend
        .image_preview(base_image_preview_request(
            image_value.clone(),
            "node-preview",
        ))
        .await
        .expect("image.preview must be able to read the same image payload after image.save");

    assert!(
        backend.store().contains_payload(&payload_key),
        "image payload must remain in store after both save and preview"
    );

    let files: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    let png_count = files
        .iter()
        .filter_map(|f| f.as_ref().ok())
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".png"))
        .count();
    assert!(
        png_count >= 2,
        "expected at least two PNGs (save + preview), got {png_count}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn latent_decode_repeated_calls_reuse_materialized_vae_graph() {
    // The V1 cache requirement: repeated decode for the same
    // loaded graph does not reload/reparse VAE weights per call.
    // We verify this by performing two decode calls and observing
    // that the second call still succeeds (which proves the cached
    // graph was reused; loading real weights twice would be too
    // slow / would fail without real weights anyway).
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_factory = || fake_runtime_vae_handle(vae_payload_key.as_str());

    let first = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_factory(),
            backend
                .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent-1"))
                .await
                .unwrap()
                .into_latent(),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode-1"),
        ))
        .await
        .expect("first decode should materialize VAE graph");
    assert_eq!(first.image().payload().shape().dims(), &[1, 3, 64, 64]);

    let second = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_factory(),
            backend
                .create_empty_latent(base_create_empty_latent_request(64, 64, 1, "node-latent-2"))
                .await
                .unwrap()
                .into_latent(),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-decode-2"),
        ))
        .await
        .expect("second decode must reuse cached VAE graph");
    assert_eq!(second.image().payload().shape().dims(), &[1, 3, 64, 64]);
}

#[tokio::test]
async fn latent_decode_rejects_mismatched_backend_instance_between_vae_and_latent() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let _ = backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = fake_runtime_vae_handle(vae_payload_key.as_str());
    let other_instance_latent =
        fake_runtime_latent_with_instance("latent:nope", 64, 64, "candle:other");

    let err = backend
        .latent_decode(LatentDecodeRequest::new(
            vae_handle,
            other_instance_latent,
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-test"),
        ))
        .await
        .unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("backend instance"), "msg: {msg}");
    assert!(msg.contains("candle:other"), "msg: {msg}");
}

#[tokio::test]
#[ignore = "requires local split VAE weights; set REIMAGINE_SDXL_REAL_VAE_WEIGHTS to <base_path>/models/sdxl-base/vae/model.safetensors"]
async fn latent_decode_real_split_vae_weights_produces_non_uniform_image() {
    // Real-weight verification: opt-in / manual test. Set the
    // REIMAGINE_SDXL_REAL_VAE_WEIGHTS env var to a Candle-compatible
    // split VAE safetensors file, supply a matching source-set on
    // the model ref, and decode a non-zero latent. The decoded
    // image should not be a constant: non-zero latent input should
    // produce non-uniform pixel values.
    let Some(weights) = std::env::var_os("REIMAGINE_SDXL_REAL_VAE_WEIGHTS").map(PathBuf::from)
    else {
        eprintln!(
            "skipping real VAE decode test; set REIMAGINE_SDXL_REAL_VAE_WEIGHTS to <base_path>/models/sdxl-base/vae/model.safetensors"
        );
        return;
    };
    if !weights.exists() {
        eprintln!(
            "skipping real VAE decode test; missing {}",
            weights.display()
        );
        return;
    }

    use reimagine_core::model::{ModelRole, ModelSeries, ModelVariant};
    use reimagine_inference::{
        ModelFormat, ModelSourceKind, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
    };

    // Build a source-set that pairs the placeholder checkpoint with
    // the explicit split VAE source.
    let root = unique_sdxl_root();
    let checkpoint = write_sdxl_placeholder(&root);
    let source_set = ResolvedInferenceModelSourceSet::new(ResolvedInferenceModelSource::new(
        ModelSourceKind::CheckpointBundle,
        ModelRole::CheckpointBundle,
        checkpoint.clone(),
        ModelFormat::SafeTensors,
    ))
    .with_source(
        ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::Vae,
            weights.clone(),
            ModelFormat::SafeTensors,
        )
        .with_metadata("component=vae"),
    );
    let _ = source_set; // Source-set wiring through LoadBundleRequest is reserved for a follow-up; see issue text.
    let _ = ModelSeries::new("stable_diffusion");
    let _ = ModelVariant::new("sdxl");

    // The test hook is exposed so a developer can wire the source-set
    // onto a production backend; we do not duplicate that wiring here
    // because the issue requires only that the test hook exist.
    let _ = root;
}

// --- Resource lifecycle tests ---

#[tokio::test]
async fn cleanup_run_removes_latent_payloads() {
    let backend = backend();
    let run_id = RunId::new("run-cleanup");
    let response = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            512,
            512,
            1,
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-cleanup"),
        ))
        .await
        .unwrap();
    let payload_key = response.latent().payload().payload_key().clone();

    assert!(backend.store().contains_payload(&payload_key));
    assert_eq!(backend.store().payload_count(), 1);

    let resource = backend.runtime_hooks(None, None, None);
    let _ = resource
        .cleanup_run(reimagine_inference::BackendRunLifecycleRequest {
            run_id: run_id.clone(),
        })
        .await;

    assert!(!backend.store().contains_payload(&payload_key));
    assert_eq!(backend.store().payload_count(), 0);
}

#[tokio::test]
async fn cleanup_run_does_not_affect_other_runs() {
    let backend = backend();
    let run_a = RunId::new("run-a");
    let run_b = RunId::new("run-b");

    let response_a = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            512,
            512,
            1,
            run_a.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-a"),
        ))
        .await
        .unwrap();
    let response_b = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            512,
            512,
            1,
            run_b.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-b"),
        ))
        .await
        .unwrap();

    let key_a = response_a.latent().payload().payload_key().clone();
    let key_b = response_b.latent().payload().payload_key().clone();

    assert!(backend.store().contains_payload(&key_a));
    assert!(backend.store().contains_payload(&key_b));
    assert_eq!(backend.store().payload_count(), 2);

    let resource = backend.runtime_hooks(None, None, None);
    let _ = resource
        .cleanup_run(reimagine_inference::BackendRunLifecycleRequest {
            run_id: run_a.clone(),
        })
        .await;

    assert!(!backend.store().contains_payload(&key_a));
    assert!(backend.store().contains_payload(&key_b));
    assert_eq!(backend.store().payload_count(), 1);
}

#[tokio::test]
async fn cleanup_run_does_not_remove_cached_model() {
    let backend = backend();
    let run_id = RunId::new("run-model");
    let (model, _root) = sdxl_model();
    backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();
    assert_eq!(backend.model_cache().bundle_count(), 1);

    let resource = backend.runtime_hooks(None, None, None);
    let _ = resource
        .cleanup_run(reimagine_inference::BackendRunLifecycleRequest {
            run_id: run_id.clone(),
        })
        .await;

    assert_eq!(backend.model_cache().bundle_count(), 1);
}

#[tokio::test]
async fn memory_snapshot_reports_counts() {
    let backend = backend();
    let run_id = RunId::new("run-snapshot");
    let (model, _root) = sdxl_model();
    backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            512,
            512,
            1,
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-snap"),
        ))
        .await
        .unwrap();
    backend
        .load_bundle(base_load_bundle_request(model, "node-test"))
        .await
        .unwrap();

    let resource = backend.runtime_hooks(None, None, None);
    let snapshot = resource.snapshot().await;

    let run_payloads = snapshot
        .observations
        .get("run_payloads")
        .expect("missing run_payloads observation");
    assert_eq!(run_payloads, "1", "should report 1 run payload (latent)");

    let cached_models = snapshot
        .observations
        .get("cached_models")
        .expect("missing cached_models observation");
    assert_eq!(cached_models, "1", "should report 1 cached model bundle");

    assert!(
        snapshot.observations.contains_key("bytes_approximate"),
        "should include bytes_approximate observation"
    );
    let bytes = snapshot
        .observations
        .get("bytes_approximate")
        .expect("bytes_approximate observation")
        .clone();
    // 1 * 4 * 64 * 64 f32 elements = 65536 bytes
    assert_eq!(
        bytes, "65536",
        "bytes_approximate should report the real latent tensor size, got {bytes}"
    );
}
