//! Integration-style tests for the burn backend's `text.encode` adapter.
//!
//! These tests live in `tests/` and link against the burn crate
//! as an external consumer. The bundle cache is seeded through
//! the production `load_bundle` path with real safetensors
//! fixtures, so the adapter contract is exercised end-to-end
//! without depending on crate-private test seams.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use reimagine_core::model::{
    ModelId, ModelRole, NodeId, ParamValue, RunId, TensorDType, TensorShape, WorkflowId,
    WorkflowVersion,
};
use reimagine_inference::{
    Backend, BackendInstance, BackendPayloadKey, BackendTensorHandle, ExecutionValue,
    InferenceBackend, InferenceCapability, InferenceError, LoadBundleRequest, ModelFormat,
    ModelSourceKind, ResolvedInferenceModel, ResolvedInferenceModelSource,
    ResolvedInferenceModelSourceSet, RuntimeClipHandle, RuntimeLatent, TextEncodeRequest,
};
use reimagine_inference_burn::models::stable_diffusion::sdxl::{
    BurnSdxlComponentRole, metadata_keys,
};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};
use safetensors::tensor::{Dtype, View, serialize_to_file};

const BACKEND_LABEL: &str = "burn";

// ---------------------------------------------------------------------------
// Mini safetensors fixture (mirrors the shape used in tests/load_bundle.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ZeroTensorView {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for ZeroTensorView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn backend() -> BurnBackend {
    BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("burn backend")
}

fn tensor_view(shape: Vec<usize>) -> ZeroTensorView {
    let byte_len = shape.iter().product::<usize>() * Dtype::F32.bitsize() / 8;
    ZeroTensorView {
        dtype: Dtype::F32,
        shape,
        data: vec![0; byte_len],
    }
}

fn component_metadata(role: BurnSdxlComponentRole) -> HashMap<String, String> {
    HashMap::from([
        (
            metadata_keys::CONTRACT.to_owned(),
            "burn.component".to_owned(),
        ),
        (metadata_keys::CONTRACT_VERSION.to_owned(), "1".to_owned()),
        (metadata_keys::BACKEND.to_owned(), BACKEND_LABEL.to_owned()),
        (
            metadata_keys::MODEL_SERIES.to_owned(),
            "stable_diffusion".to_owned(),
        ),
        (metadata_keys::VARIANT.to_owned(), "sdxl".to_owned()),
        (
            metadata_keys::COMPONENT_ROLE.to_owned(),
            role.as_str().to_owned(),
        ),
        (
            metadata_keys::TENSOR_LAYOUT.to_owned(),
            "burn-module-snapshot".to_owned(),
        ),
        (metadata_keys::DTYPE_POLICY.to_owned(), "mixed".to_owned()),
    ])
}

fn write_component(path: &Path, role: BurnSdxlComponentRole) {
    std::fs::create_dir_all(path.parent().expect("component parent")).expect("component dir");
    let specs: Vec<(String, Vec<usize>)> = match role {
        BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => role
            .contract()
            .all_expected_tensor_specs()
            .into_iter()
            .filter(|s| s.required)
            .map(|s| (s.key, vec![1; s.shape.rank()]))
            .collect(),
        _ => role
            .contract()
            .expected_tensor_specs()
            .iter()
            .filter(|spec| spec.required)
            .map(|spec| (spec.key.to_owned(), vec![1; spec.shape.rank()]))
            .collect(),
    };
    let tensors = specs
        .into_iter()
        .map(|(key, shape)| (key, tensor_view(shape)))
        .collect::<Vec<_>>();

    serialize_to_file(tensors, Some(component_metadata(role)), path).expect("component file");
}

fn role_model_role(role: BurnSdxlComponentRole) -> ModelRole {
    match role {
        BurnSdxlComponentRole::Diffusion => ModelRole::DiffusionModel,
        BurnSdxlComponentRole::Vae => ModelRole::Vae,
        BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => {
            ModelRole::TextEncoder
        }
    }
}

fn resolver_metadata(role: BurnSdxlComponentRole) -> String {
    format!(
        "component={};backend=burn;converted_layout=burn_native_component_package;contract=burn.component;contract_version=1",
        role.as_str()
    )
}

fn split_component_source(
    root: &Path,
    role: BurnSdxlComponentRole,
) -> ResolvedInferenceModelSource {
    let path = root.join(role.as_str()).join("model.safetensors");
    write_component(&path, role);
    ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        role_model_role(role),
        path,
        ModelFormat::SafeTensors,
    )
    .with_metadata(resolver_metadata(role))
}

fn resolved_model_from_sources(
    root: &Path,
    sources: Vec<ResolvedInferenceModelSource>,
) -> ResolvedInferenceModel {
    ResolvedInferenceModel::new(
        ModelId::new("sdxl-base-burn"),
        reimagine_core::model::ModelSeries::new("stable_diffusion"),
        reimagine_core::model::ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        PathBuf::from(root),
        ModelFormat::SafeTensors,
    )
    .with_source_set(ResolvedInferenceModelSourceSet::from_sources(sources))
}

fn valid_split_sources(root: &Path) -> Vec<ResolvedInferenceModelSource> {
    BurnSdxlComponentRole::all()
        .into_iter()
        .map(|role| split_component_source(root, role))
        .collect::<Vec<_>>()
}

fn resolved_model(root: &Path) -> ResolvedInferenceModel {
    resolved_model_from_sources(root, valid_split_sources(root))
}

fn load_request(root: &Path) -> LoadBundleRequest {
    LoadBundleRequest::new(
        resolved_model(root),
        RunId::new("run-burn-text"),
        WorkflowId::new("wf-burn-text"),
        WorkflowVersion::new(1),
        NodeId::new("checkpoint-loader"),
    )
}

async fn seed_bundle_via_load(backend: &BurnBackend) {
    let temp = tempfile::tempdir().expect("tempdir");
    backend
        .load_bundle(load_request(temp.path()))
        .await
        .expect("load bundle");
}

// ---------------------------------------------------------------------------
// Adapter test fixtures
// ---------------------------------------------------------------------------

fn burn_clip(backend: &BurnBackend, instance: BackendInstance) -> RuntimeClipHandle {
    RuntimeClipHandle::with_instance(
        ModelId::new("sdxl-base-burn"),
        Backend::new(BACKEND_LABEL),
        instance,
        BackendPayloadKey::new("burn:model:sdxl-base-burn:clip"),
    )
    .with_device(backend.device_label())
}

fn foreign_burn_clip() -> RuntimeClipHandle {
    RuntimeClipHandle::with_instance(
        ModelId::new("sdxl-base-burn"),
        Backend::new(BACKEND_LABEL),
        BackendInstance::new("burn:wgpu"),
        BackendPayloadKey::new("burn:model:sdxl-base-burn:clip"),
    )
}

fn candle_clip() -> RuntimeClipHandle {
    RuntimeClipHandle::with_instance(
        ModelId::new("sdxl-base-burn"),
        Backend::new("candle"),
        BackendInstance::new("candle:cpu"),
        BackendPayloadKey::new("candle:model:sdxl-base-burn:clip"),
    )
}

fn build_text_request(clip: RuntimeClipHandle, text: ExecutionValue) -> TextEncodeRequest {
    TextEncodeRequest::new(
        clip,
        std::sync::Arc::new(text),
        RunId::new("run-text-it"),
        WorkflowId::new("wf-text-it"),
        WorkflowVersion::new(1),
        NodeId::new("node-text"),
    )
}

fn string_prompt(text: &str) -> ExecutionValue {
    ExecutionValue::Param(ParamValue::String(text.to_owned()))
}

fn latent_value() -> ExecutionValue {
    let handle = BackendTensorHandle::new(
        Backend::new(BACKEND_LABEL),
        BackendPayloadKey::new("latent:run-text-it:node-text"),
        TensorDType::F32,
        TensorShape::new(vec![1, 4, 8, 8]),
        "cpu",
    );
    ExecutionValue::Latent(RuntimeLatent::with_sdxl_base(handle, 64, 64, 1, 4))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_encode_is_advertised_as_a_burn_capability() {
    let backend = backend();
    let capabilities = backend.capabilities();
    assert!(
        capabilities.supports_capability(InferenceCapability::TextEncode),
        "burn/08f adds TextEncode capability"
    );
}

#[tokio::test]
async fn text_encode_validation_failures_surface_as_backend_execution_failed() {
    let backend = backend();
    let clip = candle_clip();
    let request = build_text_request(clip, string_prompt("hello"));

    let err = backend.text_encode(request).await.expect_err("rejected");
    match err {
        InferenceError::BackendExecutionFailed { message } => {
            assert!(message.contains("burn clip handle"), "message: {message}");
        }
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn text_encode_different_burn_instance_is_backend_execution_failed() {
    let backend = backend();
    let clip = foreign_burn_clip();
    let request = build_text_request(clip, string_prompt("hello"));

    let err = backend.text_encode(request).await.expect_err("rejected");
    match err {
        InferenceError::BackendExecutionFailed { message } => {
            assert!(message.contains("backend instance"), "message: {message}");
            assert!(message.contains("burn:wgpu"), "message: {message}");
        }
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn text_encode_missing_bundle_is_backend_execution_failed() {
    let backend = backend();
    let clip = burn_clip(&backend, backend.backend_instance());
    let request = build_text_request(clip, string_prompt("hello"));

    let err = backend.text_encode(request).await.expect_err("rejected");
    match err {
        InferenceError::BackendExecutionFailed { message } => {
            assert!(message.contains("loaded"), "message: {message}");
            assert!(message.contains("sdxl-base-burn"), "message: {message}");
        }
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn text_encode_non_string_prompt_is_backend_execution_failed() {
    let backend = backend();
    seed_bundle_via_load(&backend).await;
    let clip = burn_clip(&backend, backend.backend_instance());
    let request = build_text_request(clip, latent_value());

    let err = backend.text_encode(request).await.expect_err("rejected");
    match err {
        InferenceError::BackendExecutionFailed { message } => {
            assert!(message.contains("Param(String)"), "message: {message}");
        }
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn text_encode_succeeds_and_stores_conditioning() {
    let backend = backend();
    seed_bundle_via_load(&backend).await;
    let clip = burn_clip(&backend, backend.backend_instance());
    let request = build_text_request(clip, string_prompt("hello"));

    let response = backend
        .text_encode(request)
        .await
        .expect("burn/08f implements text.encode");
    let conditioning = response.conditioning();
    assert_eq!(conditioning.text_embedding().backend().as_str(), "burn");
    assert_eq!(
        conditioning.text_embedding().shape().dims(),
        &[1_usize, 77, 2048]
    );
    let pooled = conditioning
        .pooled_embedding()
        .expect("pooled handle present");
    assert_eq!(pooled.shape().dims(), &[1_usize, 1280]);
    // Production text.encode must insert a conditioning payload
    // into the shared store.
    let store = backend.store();
    assert!(store.payload_count() > 0, "conditioning payload stored");
}
