//! Integration tests for the Candle inference backend adapter.
//!
//! These tests exercise the backend directly without the runtime or
//! executors. They prove that `model.load_bundle` and
//! `latent.create_empty` return correctly shaped lightweight handles,
//! and that the unimplemented heavy kernels return precise
//! `BackendNotImplemented` diagnostics.

use std::collections::HashMap;
use std::sync::Arc;

use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId, WorkflowId,
    WorkflowVersion,
};
use reimagine_inference::{
    InferenceBackend, InferenceError, InferenceRequest, InferenceResponse, ModelFormat,
    ResolvedInferenceModel, operation::*,
};
use reimagine_inference_candle::{CandleBackend, CandleBackendConfig};
use reimagine_runtime::RuntimeValue;

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

fn sdxl_model() -> ResolvedInferenceModel {
    ResolvedInferenceModel::new(
        ModelId::new("sdxl-base-1.0"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        "/models/sdxl-base-1.0.safetensors",
        ModelFormat::SafeTensors,
    )
}

fn outputs_by_slot(response: &InferenceResponse) -> HashMap<&str, &Arc<RuntimeValue>> {
    response
        .outputs()
        .iter()
        .map(|output| (output.slot_id().as_str(), output.value()))
        .collect()
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
    let request = base_request(OP_MODEL_LOAD_BUNDLE).with_model(sdxl_model());

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
    let request1 = base_request(OP_MODEL_LOAD_BUNDLE).with_model(sdxl_model());
    let response1 = backend.execute(request1).await.unwrap();
    let request2 = base_request(OP_MODEL_LOAD_BUNDLE).with_model(sdxl_model());
    let response2 = backend.execute(request2).await.unwrap();

    let key1 = match outputs_by_slot(&response1)["model"].as_ref() {
        RuntimeValue::Model(handle) => handle.payload_key().as_str().to_string(),
        other => panic!("expected model handle, got {other:?}"),
    };
    let key2 = match outputs_by_slot(&response2)["model"].as_ref() {
        RuntimeValue::Model(handle) => handle.payload_key().as_str().to_string(),
        other => panic!("expected model handle, got {other:?}"),
    };
    assert_eq!(key1, key2, "same model id should reuse cached payload key");
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

fn assert_backend_not_implemented(err: InferenceError, operation_id: &str) {
    assert!(
        matches!(err, InferenceError::BackendNotImplemented { operation_id: ref op, backend_kind: ref kind, .. } if op == operation_id && kind == "candle"),
        "expected candle backend-not-implemented for {operation_id}, got {err:?}"
    );
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
