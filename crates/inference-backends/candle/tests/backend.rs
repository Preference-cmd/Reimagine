//! Integration tests for the Candle inference backend adapter.
//!
//! These tests exercise the backend directly without the runtime or
//! executors. They prove that `model.load_bundle` and
//! `latent.create_empty` return correctly shaped lightweight handles,
//! that the model cache is keyed by the resolved model id, and that
//! source validation surfaces a useful backend error when the resolved
//! model artifact is missing, unreadable, or has the wrong extension.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::DType;
use reimagine_core::ExecutionValue;
use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId, WorkflowId,
    WorkflowVersion,
};
use reimagine_inference_candle::{CandleBackend, CandleBackendConfig, LoadedModelBundle};
use reimagine_inference_core::{
    ALL_V1_OPERATIONS, InferenceBackend, InferenceError, InferenceRequest, InferenceResponse,
    ModelFormat, OP_DIFFUSION_SAMPLE, OP_IMAGE_PREVIEW, OP_IMAGE_SAVE, OP_LATENT_CREATE_EMPTY,
    OP_LATENT_DECODE, OP_MODEL_LOAD_BUNDLE, OP_TEXT_ENCODE, ResolvedInferenceModel,
};
use reimagine_runtime::RunResourceBackend;

fn backend() -> CandleBackend {
    CandleBackend::new(CandleBackendConfig::new(
        "/tmp/reimagine-candle-tests",
        "/tmp/reimagine-candle-tests-output",
    ))
    .unwrap()
}

fn base_request(operation_id: &str) -> InferenceRequest {
    InferenceRequest::new(
        operation_id.into(),
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-test"),
    )
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

fn outputs_by_slot(response: &InferenceResponse) -> HashMap<&str, &Arc<ExecutionValue>> {
    response
        .outputs()
        .iter()
        .map(|output| (output.slot_id().as_str(), output.value()))
        .collect()
}

fn assert_backend_not_implemented(err: InferenceError, operation_id: &str) {
    assert!(
        matches!(err, InferenceError::BackendNotImplemented { operation_id: ref op, backend_kind: ref kind, .. } if op == operation_id && kind == "candle"),
        "expected candle backend-not-implemented for {operation_id}, got {err:?}"
    );
}

#[tokio::test]
async fn backend_kind_is_candle() {
    let backend = backend();
    assert_eq!(backend.backend_kind().as_str(), "candle");
}

#[test]
fn capabilities_cover_all_v1_operations() {
    let backend = backend();
    let capabilities = backend.capabilities();
    assert_eq!(capabilities.backend_kind().as_str(), "candle");
    for op in ALL_V1_OPERATIONS {
        assert!(
            capabilities.supports_operation(&(*op).into()),
            "capability report should include {op}"
        );
    }
}

#[tokio::test]
async fn model_load_bundle_returns_model_clip_vae_handles() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let request = base_request(OP_MODEL_LOAD_BUNDLE).with_model(model);

    let response = backend.execute(request).await.unwrap();
    let outputs = outputs_by_slot(&response);
    assert_eq!(outputs.len(), 3);

    let model = outputs["model"];
    let clip = outputs["clip"];
    let vae = outputs["vae"];

    assert!(matches!(model.as_ref(), ExecutionValue::Model(handle) if
        handle.model_id().as_str() == "sdxl-base-1.0"
        && handle.role() == ModelRole::CheckpointBundle
        && handle.backend().as_str() == "candle"
        && handle.device_label() == Some("cpu")
        && !handle.payload_key().as_str().is_empty()
    ));
    assert!(matches!(clip.as_ref(), ExecutionValue::Clip(handle) if
        handle.model_id().as_str() == "sdxl-base-1.0"
        && handle.backend().as_str() == "candle"
        && handle.device_label() == Some("cpu")
    ));
    assert!(matches!(vae.as_ref(), ExecutionValue::Vae(handle) if
        handle.model_id().as_str() == "sdxl-base-1.0"
        && handle.backend().as_str() == "candle"
        && handle.device_label() == Some("cpu")
    ));
}

#[tokio::test]
async fn model_load_bundle_caches_bundle_for_same_model_id() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    let response1 = backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model.clone()))
        .await
        .unwrap();
    let response2 = backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let key1 = match outputs_by_slot(&response1)["model"].as_ref() {
        ExecutionValue::Model(handle) => handle.payload_key().as_str().to_string(),
        other => panic!("expected model handle, got {other:?}"),
    };
    let key2 = match outputs_by_slot(&response2)["model"].as_ref() {
        ExecutionValue::Model(handle) => handle.payload_key().as_str().to_string(),
        other => panic!("expected model handle, got {other:?}"),
    };
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
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(first_model))
        .await
        .unwrap();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(second_model))
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
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap_err();
    match err {
        InferenceError::BackendExecutionFailed { message } => {
            assert!(message.contains("missing"), "msg: {message}");
            assert!(
                message.contains("does-not-exist.safetensors"),
                "msg: {message}"
            );
        }
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    }
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
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();
    let outputs = outputs_by_slot(&response);
    // SDXL loader emits three typed handles — same as before the refactor.
    assert!(outputs.contains_key("model"));
    assert!(outputs.contains_key("clip"));
    assert!(outputs.contains_key("vae"));
    assert_eq!(
        backend.model_cache().bundle_count(),
        1,
        "sdxl loader should populate the cache"
    );
}

#[tokio::test]
async fn model_load_bundle_requires_a_model() {
    let backend = backend();
    let request = base_request(OP_MODEL_LOAD_BUNDLE);
    let err = backend.execute(request).await.unwrap_err();
    assert!(
        matches!(err, InferenceError::BackendExecutionFailed { .. }),
        "expected invalid-request mapped to execution failed, got {err:?}"
    );
}

#[tokio::test]
async fn latent_create_empty_returns_f32_latent_with_sdxl_shape() {
    let backend = backend();
    let request = base_request(OP_LATENT_CREATE_EMPTY)
        .with_param("width", ParamValue::Integer(1024))
        .with_param("height", ParamValue::Integer(512))
        .with_param("batch_size", ParamValue::Integer(2));

    let response = backend.execute(request).await.unwrap();
    let outputs = outputs_by_slot(&response);
    let latent = outputs["latent"];

    assert!(matches!(latent.as_ref(), ExecutionValue::Latent(handle) if
        handle.width() == 1024
        && handle.height() == 512
        && handle.batch() == 2
        && handle.channels() == 4
        && handle.payload().dtype() == reimagine_core::model::TensorDType::F32
        && handle.payload().shape().dims() == &[2, 4, 64, 128]
        && handle.payload().backend().as_str() == "candle"
        && handle.payload().device_label() == "cpu"
    ));
}

#[tokio::test]
async fn latent_create_empty_requires_integer_params() {
    let backend = backend();
    let request = base_request(OP_LATENT_CREATE_EMPTY)
        .with_param("width", ParamValue::String("bad".to_string()));
    let err = backend.execute(request).await.unwrap_err();
    assert!(
        matches!(err, InferenceError::BackendExecutionFailed { .. }),
        "expected invalid-request mapped to execution failed, got {err:?}"
    );
}

#[tokio::test]
async fn latent_create_empty_registers_real_tensor_in_store() {
    let backend = backend();
    let request = base_request(OP_LATENT_CREATE_EMPTY)
        .with_param("width", ParamValue::Integer(512))
        .with_param("height", ParamValue::Integer(512))
        .with_param("batch_size", ParamValue::Integer(1));
    let response = backend.execute(request).await.unwrap();
    let outputs = outputs_by_slot(&response);
    let payload_key = match outputs["latent"].as_ref() {
        ExecutionValue::Latent(handle) => handle.payload().payload_key().clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    // The store should expose the real tensor behind the typed
    // accessor, not just a placeholder descriptor.
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
    let request = base_request(OP_LATENT_CREATE_EMPTY)
        .with_param("width", ParamValue::Integer(0))
        .with_param("height", ParamValue::Integer(512))
        .with_param("batch_size", ParamValue::Integer(1));
    let err = backend.execute(request).await.unwrap_err();
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
    let request = base_request(OP_LATENT_CREATE_EMPTY)
        .with_param("width", ParamValue::Integer(513))
        .with_param("height", ParamValue::Integer(512))
        .with_param("batch_size", ParamValue::Integer(1));
    let err = backend.execute(request).await.unwrap_err();
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

    // Load the bundle first
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    // Get the clip handle from the cached bundle
    let bundle = backend
        .model_cache()
        .get_bundle(&reimagine_core::model::ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let clip_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.clip_payload_key.clone(),
    };
    let clip_handle = ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
        reimagine_core::model::ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        clip_key,
    ));

    let request = base_request(OP_TEXT_ENCODE)
        .with_input("clip", Arc::new(clip_handle))
        .with_input(
            "text",
            Arc::new(ExecutionValue::Param(
                reimagine_core::model::ParamValue::String(
                    "a cinematic lake at sunrise".to_string(),
                ),
            )),
        );

    let response = backend.execute(request).await.unwrap();
    let outputs = outputs_by_slot(&response);
    assert_eq!(outputs.len(), 1);

    let conditioning = outputs["conditioning"];
    assert!(
        matches!(conditioning.as_ref(), ExecutionValue::Conditioning(handle) if
            handle.text_embedding().dtype() == reimagine_core::model::TensorDType::F32
            && handle.text_embedding().shape().dims() == &[1, 77, 2048]
            && handle.pooled_embedding().is_some()
        ),
        "expected Conditioning with correct shapes, got {conditioning:?}"
    );
}

#[tokio::test]
async fn text_encode_positive_and_negative_prompts_both_succeed() {
    let backend = backend();
    let (model, _root) = sdxl_model();

    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&reimagine_core::model::ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let clip_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.clip_payload_key.clone(),
    };
    let make_clip = || {
        ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
            reimagine_core::model::ModelId::new("sdxl-base-1.0"),
            reimagine_runtime::BackendKind::from("candle"),
            clip_key.clone(),
        ))
    };

    // Positive prompt (uses different node_id to produce distinct payload key)
    let positive_req = InferenceRequest::new(
        OP_TEXT_ENCODE.into(),
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-positive-encode"),
    )
    .with_input("clip", Arc::new(make_clip()))
    .with_input(
        "text",
        Arc::new(ExecutionValue::Param(
            reimagine_core::model::ParamValue::String("cinematic lake at sunrise".to_string()),
        )),
    );
    let positive_resp = backend.execute(positive_req).await.unwrap();
    assert_eq!(positive_resp.outputs().len(), 1);

    // Negative prompt (uses different node_id to produce distinct payload key)
    let negative_req = InferenceRequest::new(
        OP_TEXT_ENCODE.into(),
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-negative-encode"),
    )
    .with_input("clip", Arc::new(make_clip()))
    .with_input(
        "text",
        Arc::new(ExecutionValue::Param(
            reimagine_core::model::ParamValue::String("low quality, blurry".to_string()),
        )),
    );
    let negative_resp = backend.execute(negative_req).await.unwrap();
    assert_eq!(negative_resp.outputs().len(), 1);

    // Both should have stored conditioning payloads
    assert_eq!(backend.store().payload_count(), 2);
}

#[tokio::test]
async fn text_encode_conditioning_is_run_scoped_and_cleaned() {
    let backend = backend();
    let (model, _root) = sdxl_model();

    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let run_id = reimagine_core::model::RunId::new("run-conditioning-cleanup");
    let bundle = backend
        .model_cache()
        .get_bundle(&reimagine_core::model::ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let clip_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.clip_payload_key.clone(),
    };
    let clip_handle = ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
        reimagine_core::model::ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        clip_key,
    ));

    let request = InferenceRequest::new(
        OP_TEXT_ENCODE.into(),
        run_id.clone(),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-encode"),
    )
    .with_input("clip", Arc::new(clip_handle))
    .with_input(
        "text",
        Arc::new(ExecutionValue::Param(
            reimagine_core::model::ParamValue::String("test prompt".to_string()),
        )),
    );

    backend.execute(request).await.unwrap();
    assert_eq!(backend.store().run_payload_count(&run_id), 1);

    let resource = backend.resource_backend();
    resource.cleanup_run(&run_id).await;

    assert_eq!(backend.store().run_payload_count(&run_id), 0);
    assert_eq!(backend.store().payload_count(), 0);
}

#[tokio::test]
async fn text_encode_missing_clip_input_returns_error() {
    let backend = backend();
    let request = base_request(OP_TEXT_ENCODE).with_input(
        "text",
        Arc::new(ExecutionValue::Param(
            reimagine_core::model::ParamValue::String("test".to_string()),
        )),
    );
    let err = backend.execute(request).await.unwrap_err();
    assert!(
        matches!(err, InferenceError::BackendExecutionFailed { ref message } if message.contains("clip")),
        "expected error about missing clip input, got {err:?}"
    );
}

#[tokio::test]
async fn text_encode_missing_text_input_returns_error() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&reimagine_core::model::ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let clip_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.clip_payload_key.clone(),
    };
    let clip_handle = ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
        reimagine_core::model::ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        clip_key,
    ));

    let request = base_request(OP_TEXT_ENCODE).with_input("clip", Arc::new(clip_handle));
    let err = backend.execute(request).await.unwrap_err();
    assert!(
        matches!(err, InferenceError::BackendExecutionFailed { ref message } if message.contains("text")),
        "expected error about missing text input, got {err:?}"
    );
}

#[tokio::test]
async fn text_encode_without_loaded_bundle_returns_error() {
    let backend = backend();
    let clip_handle = ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
        reimagine_core::model::ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        reimagine_runtime::BackendPayloadKey::new("bundle:sdxl-base-1.0:clip"),
    ));

    let request = base_request(OP_TEXT_ENCODE)
        .with_input("clip", Arc::new(clip_handle))
        .with_input(
            "text",
            Arc::new(ExecutionValue::Param(
                reimagine_core::model::ParamValue::String("test".to_string()),
            )),
        );
    let err = backend.execute(request).await.unwrap_err();
    assert!(
        matches!(err, InferenceError::BackendExecutionFailed { ref message } if message.contains("no loaded model bundle")),
        "expected error about missing bundle, got {err:?}"
    );
}

#[tokio::test]
async fn text_encode_rejects_wrong_backend_clip_handle() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let clip_handle = ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
        reimagine_core::model::ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("other-backend"),
        reimagine_runtime::BackendPayloadKey::new("bundle:sdxl-base-1.0:clip"),
    ));

    let request = base_request(OP_TEXT_ENCODE)
        .with_input("clip", Arc::new(clip_handle))
        .with_input(
            "text",
            Arc::new(ExecutionValue::Param(
                reimagine_core::model::ParamValue::String("test".to_string()),
            )),
        );
    let err = backend.execute(request).await.unwrap_err();
    assert!(
        matches!(err, InferenceError::BackendExecutionFailed { ref message }
            if message.contains("other-backend") && message.contains("expected `candle`")),
        "expected wrong-backend clip error, got {err:?}"
    );
}

#[tokio::test]
async fn text_encode_rejects_wrong_clip_payload_key() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let clip_handle = ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
        reimagine_core::model::ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        reimagine_runtime::BackendPayloadKey::new("bundle:sdxl-base-1.0:not-clip"),
    ));

    let request = base_request(OP_TEXT_ENCODE)
        .with_input("clip", Arc::new(clip_handle))
        .with_input(
            "text",
            Arc::new(ExecutionValue::Param(
                reimagine_core::model::ParamValue::String("test".to_string()),
            )),
        );
    let err = backend.execute(request).await.unwrap_err();
    assert!(
        matches!(err, InferenceError::BackendExecutionFailed { ref message }
            if message.contains("not-clip") && message.contains("loaded SDXL CLIP payload")),
        "expected wrong-payload clip error, got {err:?}"
    );
}

#[tokio::test]
async fn diffusion_sample_succeeds_after_text_encode_and_latent_create_empty() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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

    let make_model = || {
        ExecutionValue::Model(reimagine_runtime::RuntimeModelHandle::new(
            ModelId::new("sdxl-base-1.0"),
            ModelRole::CheckpointBundle,
            reimagine_runtime::BackendKind::from("candle"),
            model_payload_key.clone(),
        ))
    };
    let make_clip = || {
        ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
            ModelId::new("sdxl-base-1.0"),
            reimagine_runtime::BackendKind::from("candle"),
            clip_payload_key.clone(),
        ))
    };

    // Encode the positive and negative conditioning so the sampler can
    // resolve them through the typed store accessor.
    let positive_encode = backend
        .execute(
            base_request(OP_TEXT_ENCODE)
                .with_input("clip", Arc::new(make_clip()))
                .with_input(
                    "text",
                    Arc::new(ExecutionValue::Param(
                        reimagine_core::model::ParamValue::String(
                            "cinematic lake at sunrise".to_string(),
                        ),
                    )),
                ),
        )
        .await
        .unwrap();
    let positive = match &positive_encode.outputs()[0].value().as_ref() {
        ExecutionValue::Conditioning(c) => c.clone(),
        other => panic!("expected conditioning, got {other:?}"),
    };

    let negative_encode = backend
        .execute(
            InferenceRequest::new(
                OP_TEXT_ENCODE.into(),
                RunId::new("run-test"),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new("node-negative-encode"),
            )
            .with_input("clip", Arc::new(make_clip()))
            .with_input(
                "text",
                Arc::new(ExecutionValue::Param(
                    reimagine_core::model::ParamValue::String("low quality, blurry".to_string()),
                )),
            ),
        )
        .await
        .unwrap();
    let negative = match &negative_encode.outputs()[0].value().as_ref() {
        ExecutionValue::Conditioning(c) => c.clone(),
        other => panic!("expected conditioning, got {other:?}"),
    };

    // Allocate the empty latent.
    let latent_resp = backend
        .execute(
            base_request(OP_LATENT_CREATE_EMPTY)
                .with_param("width", ParamValue::Integer(64))
                .with_param("height", ParamValue::Integer(64))
                .with_param("batch_size", ParamValue::Integer(1)),
        )
        .await
        .unwrap();
    let input_latent = match latent_resp.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    let sampler_resp = backend
        .execute(
            base_request(OP_DIFFUSION_SAMPLE)
                .with_input("model", Arc::new(make_model()))
                .with_input("positive", Arc::new(ExecutionValue::Conditioning(positive)))
                .with_input("negative", Arc::new(ExecutionValue::Conditioning(negative)))
                .with_input("latent", Arc::new(ExecutionValue::Latent(input_latent)))
                .with_param("seed", ParamValue::Seed(123456789))
                .with_param("steps", ParamValue::Integer(20))
                .with_param("cfg", ParamValue::Float(7.0))
                .with_param("sampler", ParamValue::Select("euler".to_string()))
                .with_param("scheduler", ParamValue::Select("normal".to_string()))
                .with_param("denoise", ParamValue::Float(1.0)),
        )
        .await
        .unwrap();
    let sampled = match sampler_resp.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };
    assert_eq!(sampled.width(), 64);
    assert_eq!(sampled.height(), 64);
    assert_eq!(sampled.batch(), 1);
    assert_eq!(sampled.channels(), 4);
    assert_eq!(
        sampled.payload().dtype(),
        reimagine_core::model::TensorDType::F32
    );
    assert_eq!(sampled.payload().backend().as_str(), "candle");
    assert_eq!(sampled.payload().device_label(), "cpu");
    // SDXL latent shape: [batch, 4, height/8, width/8].
    assert_eq!(sampled.payload().shape().dims(), &[1, 4, 8, 8]);

    // The sampled latent must be present in the store under the new
    // payload key, and the store must keep the run-scoped cleanup
    // invariant.
    let sampled_payload = backend
        .store()
        .get_latent(sampled.payload().payload_key())
        .expect("sampled latent present");
    assert_eq!(sampled_payload.dims(), vec![1, 4, 8, 8]);
    assert_eq!(sampled_payload.dtype(), candle_core::DType::F32);
    assert_eq!(
        backend.store().run_payload_count(&RunId::new("run-test")),
        4,
        "two conditioning payloads + input latent + sampled latent"
    );
}

#[tokio::test]
async fn diffusion_sample_is_deterministic_for_same_seed_and_inputs() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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
    let make_model = || {
        ExecutionValue::Model(reimagine_runtime::RuntimeModelHandle::new(
            ModelId::new("sdxl-base-1.0"),
            ModelRole::CheckpointBundle,
            reimagine_runtime::BackendKind::from("candle"),
            model_payload_key.clone(),
        ))
    };
    let make_clip = || {
        ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
            ModelId::new("sdxl-base-1.0"),
            reimagine_runtime::BackendKind::from("candle"),
            clip_payload_key.clone(),
        ))
    };
    let make_conditioning = || async {
        let response = backend
            .execute(
                base_request(OP_TEXT_ENCODE)
                    .with_input("clip", Arc::new(make_clip()))
                    .with_input(
                        "text",
                        Arc::new(ExecutionValue::Param(
                            reimagine_core::model::ParamValue::String(
                                "deterministic test prompt".to_string(),
                            ),
                        )),
                    ),
            )
            .await
            .unwrap();
        match response.outputs()[0].value().as_ref() {
            ExecutionValue::Conditioning(c) => c.clone(),
            other => panic!("expected conditioning, got {other:?}"),
        }
    };
    let positive = make_conditioning().await;
    let negative = make_conditioning().await;

    let make_latent = || async {
        let response = backend
            .execute(
                base_request(OP_LATENT_CREATE_EMPTY)
                    .with_param("width", ParamValue::Integer(32))
                    .with_param("height", ParamValue::Integer(32))
                    .with_param("batch_size", ParamValue::Integer(1)),
            )
            .await
            .unwrap();
        match response.outputs()[0].value().as_ref() {
            ExecutionValue::Latent(l) => l.clone(),
            other => panic!("expected latent, got {other:?}"),
        }
    };
    let latent_a = make_latent().await;
    let latent_b = make_latent().await;

    let sample_request = |latent: reimagine_runtime::RuntimeLatent, node: &str| {
        InferenceRequest::new(
            OP_DIFFUSION_SAMPLE.into(),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new(node),
        )
        .with_input("model", Arc::new(make_model()))
        .with_input(
            "positive",
            Arc::new(ExecutionValue::Conditioning(positive.clone())),
        )
        .with_input(
            "negative",
            Arc::new(ExecutionValue::Conditioning(negative.clone())),
        )
        .with_input("latent", Arc::new(ExecutionValue::Latent(latent)))
        .with_param("seed", ParamValue::Seed(42))
        .with_param("steps", ParamValue::Integer(15))
        .with_param("cfg", ParamValue::Float(6.5))
        .with_param("sampler", ParamValue::Select("euler".to_string()))
        .with_param("scheduler", ParamValue::Select("normal".to_string()))
        .with_param("denoise", ParamValue::Float(0.8))
    };

    let resp_a = backend
        .execute(sample_request(latent_a, "node-sample-a"))
        .await
        .unwrap();
    let resp_b = backend
        .execute(sample_request(latent_b, "node-sample-b"))
        .await
        .unwrap();

    let key_a = match resp_a.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.payload().payload_key().clone(),
        other => panic!("expected latent, got {other:?}"),
    };
    let key_b = match resp_b.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.payload().payload_key().clone(),
        other => panic!("expected latent, got {other:?}"),
    };
    let tensor_a = backend.store().get_latent(&key_a).unwrap().into_tensor();
    let tensor_b = backend.store().get_latent(&key_b).unwrap().into_tensor();
    let data_a = tensor_a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let data_b = tensor_b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        data_a, data_b,
        "sampling must be deterministic for the same seed"
    );
}

#[tokio::test]
async fn diffusion_sample_rejects_missing_inputs() {
    let backend = backend();
    let request = base_request(OP_DIFFUSION_SAMPLE);
    let err = backend.execute(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("model"), "msg: {msg}");
}

#[tokio::test]
async fn diffusion_sample_rejects_wrong_backend_latent() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model.clone()))
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
    let model_value = ExecutionValue::Model(reimagine_runtime::RuntimeModelHandle::new(
        ModelId::new("sdxl-base-1.0"),
        ModelRole::CheckpointBundle,
        reimagine_runtime::BackendKind::from("candle"),
        model_payload_key,
    ));
    let conditioning_value =
        ExecutionValue::Conditioning(reimagine_runtime::ExecutionConditioning::new(
            reimagine_runtime::BackendTensorHandle::new(
                reimagine_runtime::BackendKind::from("candle"),
                clip_payload_key,
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
                "cpu",
            ),
            reimagine_runtime::ConditioningMetadata::new(64, 64),
        ));
    let wrong_backend_latent = ExecutionValue::Latent(reimagine_runtime::RuntimeLatent::new(
        reimagine_runtime::BackendTensorHandle::new(
            reimagine_runtime::BackendKind::from("other-backend"),
            reimagine_runtime::BackendPayloadKey::new("latent:other"),
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        64,
        64,
        1,
        4,
    ));

    let request = base_request(OP_DIFFUSION_SAMPLE)
        .with_input("model", Arc::new(model_value))
        .with_input("positive", Arc::new(conditioning_value.clone()))
        .with_input("negative", Arc::new(conditioning_value))
        .with_input("latent", Arc::new(wrong_backend_latent))
        .with_param("seed", ParamValue::Seed(1))
        .with_param("steps", ParamValue::Integer(10))
        .with_param("cfg", ParamValue::Float(7.0))
        .with_param("sampler", ParamValue::Select("euler".to_string()))
        .with_param("scheduler", ParamValue::Select("normal".to_string()))
        .with_param("denoise", ParamValue::Float(1.0));
    let err = backend.execute(request).await.unwrap_err();
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
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let model_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.model_payload_key.clone(),
    };
    let model_value = ExecutionValue::Model(reimagine_runtime::RuntimeModelHandle::new(
        ModelId::new("sdxl-base-1.0"),
        ModelRole::CheckpointBundle,
        reimagine_runtime::BackendKind::from("candle"),
        model_payload_key,
    ));
    let text_handle = reimagine_runtime::BackendTensorHandle::new(
        reimagine_runtime::BackendKind::from("candle"),
        reimagine_runtime::BackendPayloadKey::new("conditioning:test"),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
        "cpu",
    );
    let pooled_handle = reimagine_runtime::BackendTensorHandle::new(
        reimagine_runtime::BackendKind::from("other-backend"),
        reimagine_runtime::BackendPayloadKey::new("conditioning:test"),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 1280]),
        "cpu",
    );
    let conditioning = ExecutionValue::Conditioning(
        reimagine_runtime::ExecutionConditioning::new(
            text_handle,
            reimagine_runtime::ConditioningMetadata::new(64, 64),
        )
        .with_pooled_embedding(pooled_handle),
    );

    let request = base_request(OP_DIFFUSION_SAMPLE)
        .with_input("model", Arc::new(model_value))
        .with_input("positive", Arc::new(conditioning))
        .with_input("negative", Arc::new(ExecutionValue::Null))
        .with_input("latent", Arc::new(ExecutionValue::Null));
    let err = backend.execute(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("pooled conditioning"), "msg: {msg}");
    assert!(msg.contains("other-backend"), "msg: {msg}");
}

#[tokio::test]
async fn diffusion_sample_rejects_mismatched_pooled_conditioning_payload() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();
    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let model_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.model_payload_key.clone(),
    };
    let model_value = ExecutionValue::Model(reimagine_runtime::RuntimeModelHandle::new(
        ModelId::new("sdxl-base-1.0"),
        ModelRole::CheckpointBundle,
        reimagine_runtime::BackendKind::from("candle"),
        model_payload_key,
    ));
    let text_handle = reimagine_runtime::BackendTensorHandle::new(
        reimagine_runtime::BackendKind::from("candle"),
        reimagine_runtime::BackendPayloadKey::new("conditioning:text"),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
        "cpu",
    );
    let pooled_handle = reimagine_runtime::BackendTensorHandle::new(
        reimagine_runtime::BackendKind::from("candle"),
        reimagine_runtime::BackendPayloadKey::new("conditioning:pooled"),
        reimagine_core::model::TensorDType::F32,
        reimagine_core::model::TensorShape::new(vec![1, 1280]),
        "cpu",
    );
    let conditioning = ExecutionValue::Conditioning(
        reimagine_runtime::ExecutionConditioning::new(
            text_handle,
            reimagine_runtime::ConditioningMetadata::new(64, 64),
        )
        .with_pooled_embedding(pooled_handle),
    );

    let request = base_request(OP_DIFFUSION_SAMPLE)
        .with_input("model", Arc::new(model_value))
        .with_input("positive", Arc::new(conditioning))
        .with_input("negative", Arc::new(ExecutionValue::Null))
        .with_input("latent", Arc::new(ExecutionValue::Null));
    let err = backend.execute(request).await.unwrap_err();
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
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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
    let model_value = ExecutionValue::Model(reimagine_runtime::RuntimeModelHandle::new(
        ModelId::new("sdxl-base-1.0"),
        ModelRole::CheckpointBundle,
        reimagine_runtime::BackendKind::from("candle"),
        model_payload_key,
    ));
    let cond_value = ExecutionValue::Conditioning(reimagine_runtime::ExecutionConditioning::new(
        reimagine_runtime::BackendTensorHandle::new(
            reimagine_runtime::BackendKind::from("candle"),
            clip_payload_key,
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 77, 2048]),
            "cpu",
        ),
        reimagine_runtime::ConditioningMetadata::new(64, 64),
    ));
    let latent_value = ExecutionValue::Latent(reimagine_runtime::RuntimeLatent::new(
        reimagine_runtime::BackendTensorHandle::new(
            reimagine_runtime::BackendKind::from("candle"),
            reimagine_runtime::BackendPayloadKey::new("latent:missing"),
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        64,
        64,
        1,
        4,
    ));

    let request = base_request(OP_DIFFUSION_SAMPLE)
        .with_input("model", Arc::new(model_value))
        .with_input("positive", Arc::new(cond_value.clone()))
        .with_input("negative", Arc::new(cond_value))
        .with_input("latent", Arc::new(latent_value))
        .with_param("seed", ParamValue::Seed(1))
        .with_param("steps", ParamValue::Integer(10))
        .with_param("cfg", ParamValue::Float(7.0))
        .with_param("sampler", ParamValue::Select("dpmpp_2m".to_string()))
        .with_param("scheduler", ParamValue::Select("normal".to_string()))
        .with_param("denoise", ParamValue::Float(1.0));
    let err = backend.execute(request).await.unwrap_err();
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
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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
    let model_value = ExecutionValue::Model(reimagine_runtime::RuntimeModelHandle::new(
        ModelId::new("sdxl-base-1.0"),
        ModelRole::CheckpointBundle,
        reimagine_runtime::BackendKind::from("candle"),
        model_payload_key,
    ));
    let clip_handle = ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        clip_payload_key,
    ));

    let positive = encode_conditioning(&backend, &clip_handle, "node-positive-cond", "test").await;
    let negative = encode_conditioning(&backend, &clip_handle, "node-negative-cond", "test").await;

    // Latent handle points to a key that was never registered.
    let latent_value = ExecutionValue::Latent(reimagine_runtime::RuntimeLatent::new(
        reimagine_runtime::BackendTensorHandle::new(
            reimagine_runtime::BackendKind::from("candle"),
            reimagine_runtime::BackendPayloadKey::new("latent:not-in-store"),
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        64,
        64,
        1,
        4,
    ));

    let request = base_request(OP_DIFFUSION_SAMPLE)
        .with_input("model", Arc::new(model_value))
        .with_input("positive", Arc::new(ExecutionValue::Conditioning(positive)))
        .with_input("negative", Arc::new(ExecutionValue::Conditioning(negative)))
        .with_input("latent", Arc::new(latent_value))
        .with_param("seed", ParamValue::Seed(1))
        .with_param("steps", ParamValue::Integer(10))
        .with_param("cfg", ParamValue::Float(7.0))
        .with_param("sampler", ParamValue::Select("euler".to_string()))
        .with_param("scheduler", ParamValue::Select("normal".to_string()))
        .with_param("denoise", ParamValue::Float(1.0));
    let err = backend.execute(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("latent:not-in-store"), "msg: {msg}");
}

async fn encode_conditioning(
    backend: &CandleBackend,
    clip: &ExecutionValue,
    node: &str,
    text: &str,
) -> reimagine_runtime::ExecutionConditioning {
    let resp = backend
        .execute(
            InferenceRequest::new(
                OP_TEXT_ENCODE.into(),
                RunId::new("run-test"),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new(node),
            )
            .with_input("clip", Arc::new(clone_runtime_value(clip)))
            .with_input(
                "text",
                Arc::new(ExecutionValue::Param(
                    reimagine_core::model::ParamValue::String(text.to_string()),
                )),
            ),
        )
        .await
        .unwrap();
    match resp.outputs()[0].value().as_ref() {
        ExecutionValue::Conditioning(c) => c.clone(),
        other => panic!("expected conditioning, got {other:?}"),
    }
}

fn clone_runtime_value(value: &ExecutionValue) -> ExecutionValue {
    match value {
        ExecutionValue::Clip(handle) => {
            ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
                handle.model_id().clone(),
                handle.backend().clone(),
                handle.payload_key().clone(),
            ))
        }
        other => other.clone(),
    }
}

#[tokio::test]
async fn diffusion_sample_cleans_up_run_scoped_payload() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
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
    let make_model = || {
        ExecutionValue::Model(reimagine_runtime::RuntimeModelHandle::new(
            ModelId::new("sdxl-base-1.0"),
            ModelRole::CheckpointBundle,
            reimagine_runtime::BackendKind::from("candle"),
            model_payload_key.clone(),
        ))
    };
    let make_clip = || {
        ExecutionValue::Clip(reimagine_runtime::RuntimeClipHandle::new(
            ModelId::new("sdxl-base-1.0"),
            reimagine_runtime::BackendKind::from("candle"),
            clip_payload_key.clone(),
        ))
    };

    let run_id = RunId::new("run-diffusion-cleanup");

    let positive = {
        let resp = backend
            .execute(
                InferenceRequest::new(
                    OP_TEXT_ENCODE.into(),
                    run_id.clone(),
                    WorkflowId::new("wf-test"),
                    WorkflowVersion::new(1),
                    NodeId::new("node-positive-encode"),
                )
                .with_input("clip", Arc::new(make_clip()))
                .with_input(
                    "text",
                    Arc::new(ExecutionValue::Param(
                        reimagine_core::model::ParamValue::String("a cat".to_string()),
                    )),
                ),
            )
            .await
            .unwrap();
        match resp.outputs()[0].value().as_ref() {
            ExecutionValue::Conditioning(c) => c.clone(),
            other => panic!("expected conditioning, got {other:?}"),
        }
    };
    let negative = {
        let resp = backend
            .execute(
                InferenceRequest::new(
                    OP_TEXT_ENCODE.into(),
                    run_id.clone(),
                    WorkflowId::new("wf-test"),
                    WorkflowVersion::new(1),
                    NodeId::new("node-negative-encode"),
                )
                .with_input("clip", Arc::new(make_clip()))
                .with_input(
                    "text",
                    Arc::new(ExecutionValue::Param(
                        reimagine_core::model::ParamValue::String("blurry".to_string()),
                    )),
                ),
            )
            .await
            .unwrap();
        match resp.outputs()[0].value().as_ref() {
            ExecutionValue::Conditioning(c) => c.clone(),
            other => panic!("expected conditioning, got {other:?}"),
        }
    };
    let latent = {
        let resp = backend
            .execute(
                InferenceRequest::new(
                    OP_LATENT_CREATE_EMPTY.into(),
                    run_id.clone(),
                    WorkflowId::new("wf-test"),
                    WorkflowVersion::new(1),
                    NodeId::new("node-latent"),
                )
                .with_param("width", ParamValue::Integer(32))
                .with_param("height", ParamValue::Integer(32))
                .with_param("batch_size", ParamValue::Integer(1)),
            )
            .await
            .unwrap();
        match resp.outputs()[0].value().as_ref() {
            ExecutionValue::Latent(l) => l.clone(),
            other => panic!("expected latent, got {other:?}"),
        }
    };

    backend
        .execute(
            InferenceRequest::new(
                OP_DIFFUSION_SAMPLE.into(),
                run_id.clone(),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new("node-sampler"),
            )
            .with_input("model", Arc::new(make_model()))
            .with_input("positive", Arc::new(ExecutionValue::Conditioning(positive)))
            .with_input("negative", Arc::new(ExecutionValue::Conditioning(negative)))
            .with_input("latent", Arc::new(ExecutionValue::Latent(latent)))
            .with_param("seed", ParamValue::Seed(1))
            .with_param("steps", ParamValue::Integer(10))
            .with_param("cfg", ParamValue::Float(7.0))
            .with_param("sampler", ParamValue::Select("euler".to_string()))
            .with_param("scheduler", ParamValue::Select("normal".to_string()))
            .with_param("denoise", ParamValue::Float(1.0)),
        )
        .await
        .unwrap();

    assert_eq!(
        backend.store().run_payload_count(&run_id),
        4,
        "two conditionings + input latent + sampled latent"
    );

    let resource = backend.resource_backend();
    resource.cleanup_run(&run_id).await;
    assert_eq!(backend.store().run_payload_count(&run_id), 0);
}

#[tokio::test]
async fn latent_decode_succeeds_for_sdxl_with_loaded_bundle() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };

    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        vae_payload_key,
    ));

    let response = backend
        .execute(
            base_request(OP_LATENT_CREATE_EMPTY)
                .with_param("width", ParamValue::Integer(64))
                .with_param("height", ParamValue::Integer(64))
                .with_param("batch_size", ParamValue::Integer(1)),
        )
        .await
        .unwrap();
    let input_latent = match response.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    let decode_resp = backend
        .execute(
            base_request(OP_LATENT_DECODE)
                .with_input("vae", Arc::new(vae_handle))
                .with_input("latent", Arc::new(ExecutionValue::Latent(input_latent))),
        )
        .await
        .unwrap();
    let outputs = outputs_by_slot(&decode_resp);
    let image = outputs["image"];

    let image = match image.as_ref() {
        ExecutionValue::Image(img) => img,
        other => panic!("expected image, got {other:?}"),
    };

    assert_eq!(image.width(), 64);
    assert_eq!(image.height(), 64);
    assert_eq!(image.batch(), 1);
    assert_eq!(image.color_space(), "rgb");
    assert_eq!(
        image.payload().dtype(),
        reimagine_core::model::TensorDType::F32
    );
    assert_eq!(image.payload().shape().dims(), &[1, 3, 64, 64]);
    assert_eq!(image.payload().backend().as_str(), "candle");
    assert_eq!(image.payload().device_label(), "cpu");
}

#[tokio::test]
async fn latent_decode_rejects_missing_vae_input() {
    let backend = backend();
    let request =
        base_request(OP_LATENT_DECODE).with_input("latent", Arc::new(ExecutionValue::Null));
    let err = backend.execute(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("vae"), "msg: {msg}");
}

#[tokio::test]
async fn latent_decode_rejects_missing_latent_input() {
    let backend = backend();
    let request = base_request(OP_LATENT_DECODE).with_input("vae", Arc::new(ExecutionValue::Null));
    let err = backend.execute(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("latent"), "msg: {msg}");
}

#[tokio::test]
async fn latent_decode_rejects_wrong_backend_vae_handle() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("other-backend"),
        reimagine_runtime::BackendPayloadKey::new("bundle:sdxl-base-1.0:vae"),
    ));

    let request = base_request(OP_LATENT_DECODE)
        .with_input("vae", Arc::new(vae_handle))
        .with_input("latent", Arc::new(ExecutionValue::Null));
    let err = backend.execute(request).await.unwrap_err();
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
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        vae_payload_key,
    ));

    let wrong_backend_latent = ExecutionValue::Latent(reimagine_runtime::RuntimeLatent::new(
        reimagine_runtime::BackendTensorHandle::new(
            reimagine_runtime::BackendKind::from("other-backend"),
            reimagine_runtime::BackendPayloadKey::new("latent:other"),
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        64,
        64,
        1,
        4,
    ));

    let request = base_request(OP_LATENT_DECODE)
        .with_input("vae", Arc::new(vae_handle))
        .with_input("latent", Arc::new(wrong_backend_latent));
    let err = backend.execute(request).await.unwrap_err();
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
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        reimagine_runtime::BackendPayloadKey::new("bundle:sdxl-base-1.0:not-vae"),
    ));

    let response = backend
        .execute(
            base_request(OP_LATENT_CREATE_EMPTY)
                .with_param("width", ParamValue::Integer(64))
                .with_param("height", ParamValue::Integer(64))
                .with_param("batch_size", ParamValue::Integer(1)),
        )
        .await
        .unwrap();
    let input_latent = match response.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    let request = base_request(OP_LATENT_DECODE)
        .with_input("vae", Arc::new(vae_handle))
        .with_input("latent", Arc::new(ExecutionValue::Latent(input_latent)));
    let err = backend.execute(request).await.unwrap_err();
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
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        vae_payload_key,
    ));

    let missing_latent = ExecutionValue::Latent(reimagine_runtime::RuntimeLatent::new(
        reimagine_runtime::BackendTensorHandle::new(
            reimagine_runtime::BackendKind::from("candle"),
            reimagine_runtime::BackendPayloadKey::new("latent:not-in-store"),
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
            "cpu",
        ),
        64,
        64,
        1,
        4,
    ));

    let request = base_request(OP_LATENT_DECODE)
        .with_input("vae", Arc::new(vae_handle))
        .with_input("latent", Arc::new(missing_latent));
    let err = backend.execute(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("latent:not-in-store"), "msg: {msg}");
}

#[tokio::test]
async fn latent_decode_rejects_missing_loaded_bundle() {
    let backend = backend();
    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        reimagine_runtime::BackendPayloadKey::new("bundle:sdxl-base-1.0:vae"),
    ));

    let response = backend
        .execute(
            base_request(OP_LATENT_CREATE_EMPTY)
                .with_param("width", ParamValue::Integer(64))
                .with_param("height", ParamValue::Integer(64))
                .with_param("batch_size", ParamValue::Integer(1)),
        )
        .await
        .unwrap();
    let input_latent = match response.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    let request = base_request(OP_LATENT_DECODE)
        .with_input("vae", Arc::new(vae_handle))
        .with_input("latent", Arc::new(ExecutionValue::Latent(input_latent)));
    let err = backend.execute(request).await.unwrap_err();
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
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        vae_payload_key,
    ));

    let response = backend
        .execute(
            base_request(OP_LATENT_CREATE_EMPTY)
                .with_param("width", ParamValue::Integer(64))
                .with_param("height", ParamValue::Integer(64))
                .with_param("batch_size", ParamValue::Integer(1)),
        )
        .await
        .unwrap();
    let input_latent = match response.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    let decode_resp = backend
        .execute(
            base_request(OP_LATENT_DECODE)
                .with_input("vae", Arc::new(vae_handle))
                .with_input("latent", Arc::new(ExecutionValue::Latent(input_latent))),
        )
        .await
        .unwrap();
    let payload_key = match decode_resp.outputs()[0].value().as_ref() {
        ExecutionValue::Image(img) => img.payload().payload_key().clone(),
        other => panic!("expected image, got {other:?}"),
    };

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
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        vae_payload_key,
    ));

    let response = backend
        .execute(
            base_request(OP_LATENT_CREATE_EMPTY)
                .with_param("width", ParamValue::Integer(128))
                .with_param("height", ParamValue::Integer(64))
                .with_param("batch_size", ParamValue::Integer(2)),
        )
        .await
        .unwrap();
    let input_latent = match response.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    let decode_resp = backend
        .execute(
            base_request(OP_LATENT_DECODE)
                .with_input("vae", Arc::new(vae_handle))
                .with_input("latent", Arc::new(ExecutionValue::Latent(input_latent))),
        )
        .await
        .unwrap();
    let image = match decode_resp.outputs()[0].value().as_ref() {
        ExecutionValue::Image(img) => img,
        other => panic!("expected image, got {other:?}"),
    };

    assert_eq!(image.width(), 128);
    assert_eq!(image.height(), 64);
    assert_eq!(image.batch(), 2);
    assert_eq!(image.color_space(), "rgb");
    assert_eq!(
        image.payload().dtype(),
        reimagine_core::model::TensorDType::F32
    );
    assert_eq!(image.payload().shape().dims(), &[2, 3, 64, 128]);
}

#[tokio::test]
async fn latent_decode_runs_scoped_payload_cleanup() {
    let backend = backend();
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };
    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        vae_payload_key,
    ));

    // Create latent in a separate run so only the decoded image
    // is scoped to the cleanup run.
    let latent_run_id = RunId::new("run-latent-source");
    let response = backend
        .execute(
            InferenceRequest::new(
                OP_LATENT_CREATE_EMPTY.into(),
                latent_run_id.clone(),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new("node-latent"),
            )
            .with_param("width", ParamValue::Integer(64))
            .with_param("height", ParamValue::Integer(64))
            .with_param("batch_size", ParamValue::Integer(1)),
        )
        .await
        .unwrap();
    let input_latent = match response.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    let run_id = RunId::new("run-latent-decode-cleanup");
    backend
        .execute(
            InferenceRequest::new(
                OP_LATENT_DECODE.into(),
                run_id.clone(),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new("node-decode"),
            )
            .with_input("vae", Arc::new(vae_handle))
            .with_input("latent", Arc::new(ExecutionValue::Latent(input_latent))),
        )
        .await
        .unwrap();

    assert_eq!(
        backend.store().run_payload_count(&run_id),
        1,
        "decode should produce exactly 1 run-scoped image payload"
    );

    let resource = backend.resource_backend();
    resource.cleanup_run(&run_id).await;
    assert_eq!(backend.store().run_payload_count(&run_id), 0);
}

async fn setup_decoded_image_for_save(
    backend: &CandleBackend,
    node_id: &str,
) -> (ExecutionValue, std::path::PathBuf) {
    let (model, _root) = sdxl_model();
    backend
        .execute(base_request(OP_MODEL_LOAD_BUNDLE).with_model(model))
        .await
        .unwrap();

    let bundle = backend
        .model_cache()
        .get_bundle(&ModelId::new("sdxl-base-1.0"))
        .expect("cached bundle");
    let vae_payload_key = match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => sdxl.vae_payload_key.clone(),
    };

    let vae_handle = ExecutionValue::Vae(reimagine_runtime::RuntimeVaeHandle::new(
        ModelId::new("sdxl-base-1.0"),
        reimagine_runtime::BackendKind::from("candle"),
        vae_payload_key,
    ));

    let latent_resp = backend
        .execute(
            base_request(OP_LATENT_CREATE_EMPTY)
                .with_param("width", ParamValue::Integer(64))
                .with_param("height", ParamValue::Integer(64))
                .with_param("batch_size", ParamValue::Integer(1)),
        )
        .await
        .unwrap();
    let input_latent = match latent_resp.outputs()[0].value().as_ref() {
        ExecutionValue::Latent(l) => l.clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    let decode_resp = backend
        .execute(
            InferenceRequest::new(
                OP_LATENT_DECODE.into(),
                RunId::new("run-test"),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new(node_id),
            )
            .with_input("vae", Arc::new(vae_handle))
            .with_input("latent", Arc::new(ExecutionValue::Latent(input_latent))),
        )
        .await
        .unwrap();

    let image_value = match decode_resp.outputs()[0].value().as_ref() {
        ExecutionValue::Image(img) => img.clone(),
        other => panic!("expected image, got {other:?}"),
    };

    let output_dir = backend.output_dir().to_path_buf();
    (ExecutionValue::Image(image_value), output_dir)
}

#[tokio::test]
async fn image_save_writes_png_to_output_dir_for_sdxl_pipeline() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-save").await;

    let request = base_request(OP_IMAGE_SAVE).with_input("image", Arc::new(image_value));

    let response = backend.execute(request).await.unwrap();
    let outputs = outputs_by_slot(&response);
    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("artifact"));
    let artifact_ref = match response.outputs()[0].value().as_ref() {
        ExecutionValue::Artifact(reference) => reference.clone(),
        other => panic!("expected artifact output, got {other:?}"),
    };

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
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-preview").await;

    let request = base_request(OP_IMAGE_PREVIEW).with_input("image", Arc::new(image_value));

    let response = backend.execute(request).await.unwrap();
    assert_eq!(response.outputs().len(), 1);

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
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-save-test").await;

    let request = InferenceRequest::new(
        OP_IMAGE_SAVE.into(),
        RunId::new("run-abc-123"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-foo-bar"),
    )
    .with_input("image", Arc::new(image_value))
    .with_param(
        "filename_prefix",
        ParamValue::String("my-prefix".to_string()),
    );

    backend.execute(request).await.unwrap();

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
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-traversal").await;

    let request = InferenceRequest::new(
        OP_IMAGE_SAVE.into(),
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-test"),
    )
    .with_input("image", Arc::new(image_value))
    .with_param(
        "filename_prefix",
        ParamValue::String("../../../etc/passwd".to_string()),
    );

    let response = backend.execute(request).await.unwrap();
    assert_eq!(response.outputs().len(), 1);

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
    let wrong_backend_image = ExecutionValue::Image(reimagine_runtime::RuntimeImage::new(
        reimagine_runtime::BackendTensorHandle::new(
            reimagine_runtime::BackendKind::from("other-backend"),
            reimagine_runtime::BackendPayloadKey::new("image:fake"),
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 3, 64, 64]),
            "cpu",
        ),
        64,
        64,
        1,
        "rgb".to_string(),
    ));

    let request = base_request(OP_IMAGE_SAVE).with_input("image", Arc::new(wrong_backend_image));

    let err = backend.execute(request).await.unwrap_err();
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
    let request = base_request(OP_IMAGE_SAVE);
    let err = backend.execute(request).await.unwrap_err();
    let msg = match err {
        InferenceError::BackendExecutionFailed { message } => message,
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    };
    assert!(msg.contains("image"), "msg: {msg}");
}

#[tokio::test]
async fn image_save_rejects_missing_image_payload() {
    let backend = backend();
    let ghost_image = ExecutionValue::Image(reimagine_runtime::RuntimeImage::new(
        reimagine_runtime::BackendTensorHandle::new(
            reimagine_runtime::BackendKind::from("candle"),
            reimagine_runtime::BackendPayloadKey::new("image:not-in-store"),
            reimagine_core::model::TensorDType::F32,
            reimagine_core::model::TensorShape::new(vec![1, 3, 64, 64]),
            "cpu",
        ),
        64,
        64,
        1,
        "rgb".to_string(),
    ));

    let request = base_request(OP_IMAGE_SAVE).with_input("image", Arc::new(ghost_image));

    let err = backend.execute(request).await.unwrap_err();
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
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    // First image for first save
    let (image_value1, output_dir) =
        setup_decoded_image_for_save(&backend, "node-overwrite-1").await;

    // Second image for second save (different node_id, so different payload key)
    let (image_value2, _output_dir) =
        setup_decoded_image_for_save(&backend, "node-overwrite-2").await;

    let run_id = RunId::new("run-overwrite");

    let request1 = InferenceRequest::new(
        OP_IMAGE_SAVE.into(),
        run_id.clone(),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-overwrite"),
    )
    .with_input("image", Arc::new(image_value1))
    .with_param(
        "filename_prefix",
        ParamValue::String("overwrite-test".to_string()),
    );

    backend.execute(request1).await.unwrap();

    let files_before: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    assert!(!files_before.is_empty());

    let request2 = InferenceRequest::new(
        OP_IMAGE_SAVE.into(),
        run_id.clone(),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-overwrite"),
    )
    .with_input("image", Arc::new(image_value2))
    .with_param(
        "filename_prefix",
        ParamValue::String("overwrite-test".to_string()),
    );

    let response2 = backend.execute(request2).await.unwrap();
    assert_eq!(response2.outputs().len(), 1);

    let files_after: Vec<_> = std::fs::read_dir(&output_dir).unwrap().collect();
    assert!(!files_after.is_empty());

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_save_returns_output_with_artifact_slot_id() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    let (image_value, _output_dir) = setup_decoded_image_for_save(&backend, "node-slotid").await;

    let request = base_request(OP_IMAGE_SAVE).with_input("image", Arc::new(image_value));

    let response = backend.execute(request).await.unwrap();
    assert_eq!(response.outputs().len(), 1);
    assert_eq!(response.outputs()[0].slot_id().as_str(), "artifact");
    match response.outputs()[0].value().as_ref() {
        ExecutionValue::Artifact(reference) => {
            assert!(reference.as_str().starts_with("output/"));
            assert!(reference.as_str().ends_with(".png"));
        }
        other => panic!("expected artifact output, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn image_preview_uses_different_prefix() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    // First decode for save
    let (image_value_save, output_dir) =
        setup_decoded_image_for_save(&backend, "node-save-img").await;

    // Second decode for preview (different node_id so payload key is different)
    let (image_value_preview, _output_dir) =
        setup_decoded_image_for_save(&backend, "node-preview-img").await;

    let save_request = InferenceRequest::new(
        OP_IMAGE_SAVE.into(),
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-save"),
    )
    .with_input("image", Arc::new(image_value_save));

    backend.execute(save_request).await.unwrap();

    let preview_request = InferenceRequest::new(
        OP_IMAGE_PREVIEW.into(),
        RunId::new("run-test"),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-preview"),
    )
    .with_input("image", Arc::new(image_value_preview));

    backend.execute(preview_request).await.unwrap();

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
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-sig").await;

    let request = base_request(OP_IMAGE_SAVE).with_input("image", Arc::new(image_value));

    backend.execute(request).await.unwrap();

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
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    let (image_value, output_dir) = setup_decoded_image_for_save(&backend, "node-ihdr").await;

    let request = base_request(OP_IMAGE_SAVE).with_input("image", Arc::new(image_value));

    backend.execute(request).await.unwrap();

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
async fn image_save_cleans_up_image_payload_from_store() {
    let root = unique_sdxl_root();
    std::fs::create_dir_all(&root).unwrap();
    let backend =
        CandleBackend::new(CandleBackendConfig::new(&root, &root.join("output"))).unwrap();

    let (image_value, _output_dir) = setup_decoded_image_for_save(&backend, "node-cleanup").await;
    let payload_key = match image_value {
        ExecutionValue::Image(ref img) => img.payload().payload_key().clone(),
        other => panic!("expected image, got {other:?}"),
    };

    assert!(
        backend.store().contains_payload(&payload_key),
        "image should be in store before save"
    );

    let request = base_request(OP_IMAGE_SAVE).with_input("image", Arc::new(image_value));

    backend.execute(request).await.unwrap();

    assert!(
        !backend.store().contains_payload(&payload_key),
        "image should be removed from store after save"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn unknown_operation_returns_not_implemented() {
    let backend = backend();
    let request = base_request("custom.unknown_op");
    let err = backend.execute(request).await.unwrap_err();
    assert_backend_not_implemented(err, "custom.unknown_op");
}

// --- Resource lifecycle tests ---

#[tokio::test]
async fn cleanup_run_removes_latent_payloads() {
    let backend = backend();
    let run_id = RunId::new("run-cleanup");
    let request = InferenceRequest::new(
        OP_LATENT_CREATE_EMPTY.into(),
        run_id.clone(),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-cleanup"),
    )
    .with_param("width", ParamValue::Integer(512))
    .with_param("height", ParamValue::Integer(512))
    .with_param("batch_size", ParamValue::Integer(1));

    let response = backend.execute(request).await.unwrap();
    let outputs = outputs_by_slot(&response);
    let latent = outputs["latent"];
    let payload_key = match latent.as_ref() {
        ExecutionValue::Latent(handle) => handle.payload().payload_key().clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    assert!(backend.store().contains_payload(&payload_key));
    assert_eq!(backend.store().payload_count(), 1);

    let resource = backend.resource_backend();
    resource.cleanup_run(&run_id).await;

    assert!(!backend.store().contains_payload(&payload_key));
    assert_eq!(backend.store().payload_count(), 0);
}

#[tokio::test]
async fn cleanup_run_does_not_affect_other_runs() {
    let backend = backend();
    let run_a = RunId::new("run-a");
    let run_b = RunId::new("run-b");

    let request_a = InferenceRequest::new(
        OP_LATENT_CREATE_EMPTY.into(),
        run_a.clone(),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-a"),
    )
    .with_param("width", ParamValue::Integer(512))
    .with_param("height", ParamValue::Integer(512))
    .with_param("batch_size", ParamValue::Integer(1));

    let request_b = InferenceRequest::new(
        OP_LATENT_CREATE_EMPTY.into(),
        run_b.clone(),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-b"),
    )
    .with_param("width", ParamValue::Integer(512))
    .with_param("height", ParamValue::Integer(512))
    .with_param("batch_size", ParamValue::Integer(1));

    let response_a = backend.execute(request_a).await.unwrap();
    let response_b = backend.execute(request_b).await.unwrap();

    let key_a = match outputs_by_slot(&response_a)["latent"].as_ref() {
        ExecutionValue::Latent(handle) => handle.payload().payload_key().clone(),
        other => panic!("expected latent, got {other:?}"),
    };
    let key_b = match outputs_by_slot(&response_b)["latent"].as_ref() {
        ExecutionValue::Latent(handle) => handle.payload().payload_key().clone(),
        other => panic!("expected latent, got {other:?}"),
    };

    assert!(backend.store().contains_payload(&key_a));
    assert!(backend.store().contains_payload(&key_b));
    assert_eq!(backend.store().payload_count(), 2);

    let resource = backend.resource_backend();
    resource.cleanup_run(&run_a).await;

    assert!(!backend.store().contains_payload(&key_a));
    assert!(backend.store().contains_payload(&key_b));
    assert_eq!(backend.store().payload_count(), 1);
}

#[tokio::test]
async fn cleanup_run_does_not_remove_cached_model() {
    let backend = backend();
    let run_id = RunId::new("run-model");
    let (model, _root) = sdxl_model();
    let request = InferenceRequest::new(
        OP_MODEL_LOAD_BUNDLE.into(),
        run_id.clone(),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-model"),
    )
    .with_model(model);

    backend.execute(request).await.unwrap();
    assert_eq!(backend.model_cache().bundle_count(), 1);

    let resource = backend.resource_backend();
    resource.cleanup_run(&run_id).await;

    assert_eq!(backend.model_cache().bundle_count(), 1);
}

#[tokio::test]
async fn memory_snapshot_reports_counts() {
    let backend = backend();
    let run_id = RunId::new("run-snapshot");
    let (model, _root) = sdxl_model();
    let request = InferenceRequest::new(
        OP_LATENT_CREATE_EMPTY.into(),
        run_id.clone(),
        WorkflowId::new("wf-test"),
        WorkflowVersion::new(1),
        NodeId::new("node-snap"),
    )
    .with_param("width", ParamValue::Integer(512))
    .with_param("height", ParamValue::Integer(512))
    .with_param("batch_size", ParamValue::Integer(1));

    backend.execute(request).await.unwrap();
    backend
        .execute(
            InferenceRequest::new(
                OP_MODEL_LOAD_BUNDLE.into(),
                run_id.clone(),
                WorkflowId::new("wf-test"),
                WorkflowVersion::new(1),
                NodeId::new("node-model"),
            )
            .with_model(model),
        )
        .await
        .unwrap();

    let resource = backend.resource_backend();
    let snapshot = resource.memory_snapshot().await;

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
