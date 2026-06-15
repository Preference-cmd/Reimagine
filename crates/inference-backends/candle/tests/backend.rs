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
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::DType;
use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId, WorkflowId,
    WorkflowVersion,
};
use reimagine_inference::{
    InferenceBackend, InferenceError, InferenceRequest, InferenceResponse, ModelFormat,
    ResolvedInferenceModel, operation::*,
};
use reimagine_inference_candle::{CandleBackend, CandleBackendConfig};
use reimagine_runtime::{RunResourceBackend, RuntimeValue};

fn backend() -> CandleBackend {
    CandleBackend::new(CandleBackendConfig::new("/tmp/reimagine-candle-tests")).unwrap()
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
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-candle-it-{nonce}"))
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

fn outputs_by_slot(response: &InferenceResponse) -> HashMap<&str, &Arc<RuntimeValue>> {
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
    assert_eq!(backend.backend_kind(), "candle");
}

#[test]
fn capabilities_cover_all_v1_operations() {
    let backend = backend();
    let capabilities = backend.capabilities();
    assert_eq!(capabilities.backend_kind(), "candle");
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

    assert!(matches!(model.as_ref(), RuntimeValue::Model(handle) if
        handle.model_id().as_str() == "sdxl-base-1.0"
        && handle.role() == ModelRole::CheckpointBundle
        && handle.backend().as_str() == "candle"
        && handle.device_label() == Some("cpu")
        && !handle.payload_key().as_str().is_empty()
    ));
    assert!(matches!(clip.as_ref(), RuntimeValue::Clip(handle) if
        handle.model_id().as_str() == "sdxl-base-1.0"
        && handle.backend().as_str() == "candle"
        && handle.device_label() == Some("cpu")
    ));
    assert!(matches!(vae.as_ref(), RuntimeValue::Vae(handle) if
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
        RuntimeValue::Model(handle) => handle.payload_key().as_str().to_string(),
        other => panic!("expected model handle, got {other:?}"),
    };
    let key2 = match outputs_by_slot(&response2)["model"].as_ref() {
        RuntimeValue::Model(handle) => handle.payload_key().as_str().to_string(),
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

    assert!(matches!(latent.as_ref(), RuntimeValue::Latent(handle) if
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
        RuntimeValue::Latent(handle) => handle.payload().payload_key().clone(),
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
async fn text_encode_returns_not_implemented() {
    let backend = backend();
    let request = base_request(OP_TEXT_ENCODE)
        .with_input("clip", Arc::new(RuntimeValue::Null))
        .with_input("text", Arc::new(RuntimeValue::Null));
    let err = backend.execute(request).await.unwrap_err();
    assert_backend_not_implemented(err, OP_TEXT_ENCODE);
}

#[tokio::test]
async fn diffusion_sample_returns_not_implemented() {
    let backend = backend();
    let request = base_request(OP_DIFFUSION_SAMPLE)
        .with_input("model", Arc::new(RuntimeValue::Null))
        .with_input("positive", Arc::new(RuntimeValue::Null))
        .with_input("negative", Arc::new(RuntimeValue::Null))
        .with_input("latent", Arc::new(RuntimeValue::Null));
    let err = backend.execute(request).await.unwrap_err();
    assert_backend_not_implemented(err, OP_DIFFUSION_SAMPLE);
}

#[tokio::test]
async fn latent_decode_returns_not_implemented() {
    let backend = backend();
    let request = base_request(OP_LATENT_DECODE)
        .with_input("vae", Arc::new(RuntimeValue::Null))
        .with_input("latent", Arc::new(RuntimeValue::Null));
    let err = backend.execute(request).await.unwrap_err();
    assert_backend_not_implemented(err, OP_LATENT_DECODE);
}

#[tokio::test]
async fn image_save_returns_not_implemented() {
    let backend = backend();
    let request = base_request(OP_IMAGE_SAVE).with_input("image", Arc::new(RuntimeValue::Null));
    let err = backend.execute(request).await.unwrap_err();
    assert_backend_not_implemented(err, OP_IMAGE_SAVE);
}

#[tokio::test]
async fn image_preview_returns_not_implemented() {
    let backend = backend();
    let request = base_request(OP_IMAGE_PREVIEW).with_input("image", Arc::new(RuntimeValue::Null));
    let err = backend.execute(request).await.unwrap_err();
    assert_backend_not_implemented(err, OP_IMAGE_PREVIEW);
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
        RuntimeValue::Latent(handle) => handle.payload().payload_key().clone(),
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
        RuntimeValue::Latent(handle) => handle.payload().payload_key().clone(),
        other => panic!("expected latent, got {other:?}"),
    };
    let key_b = match outputs_by_slot(&response_b)["latent"].as_ref() {
        RuntimeValue::Latent(handle) => handle.payload().payload_key().clone(),
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
