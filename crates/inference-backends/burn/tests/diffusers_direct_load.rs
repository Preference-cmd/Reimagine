//! Integration tests for diffusers-format direct loading (BE-05).
//!
//! Validates that Burn can load diffusers-format split safetensors directly
//! without offline conversion. This exercises the validation gate in
//! `loaded.rs` that accepts files lacking the `reimagine.contract` metadata
//! key, and the `infer_role_from_path` logic that maps directory names to
//! component roles.
//!
//! All tests use synthetic in-memory safetensors data and require no GPU
//! or model downloads.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, RunId, WorkflowId, WorkflowVersion,
};
use reimagine_inference::{
    InferenceBackend, InferenceError, LoadBundleRequest, ModelFormat, ModelSourceKind,
    ResolvedInferenceModel, ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};
use reimagine_inference_burn::models::stable_diffusion::sdxl::{BurnSdxlComponentRole, metadata_keys};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};
use safetensors::tensor::{Dtype, View, serialize_to_file};

// ---------------------------------------------------------------------------
// Tiny tensor view (zero-filled, sufficient for validation gate tests)
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

fn zeros(count: usize) -> Vec<u8> {
    vec![0; count]
}

fn tensor_view_f32(shape: Vec<usize>) -> ZeroTensorView {
    let byte_len = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
    ZeroTensorView {
        dtype: Dtype::F32,
        shape,
        data: zeros(byte_len),
    }
}

// ---------------------------------------------------------------------------
// Backend helpers
// ---------------------------------------------------------------------------

fn backend() -> BurnBackend {
    BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("burn backend")
}

fn run_id() -> RunId {
    RunId::new("run-diffusers-direct-load")
}

fn workflow_id() -> WorkflowId {
    WorkflowId::new("wf-diffusers-direct-load")
}

fn workflow_version() -> WorkflowVersion {
    WorkflowVersion::new(1)
}

// ---------------------------------------------------------------------------
// Synthetic diffusers-format safetensors writers
//
// These produce valid safetensors files that deliberately omit the
// `reimagine.contract` metadata key, mimicking raw HuggingFace diffusers
// split checkpoints.
// ---------------------------------------------------------------------------

/// Write a minimal diffusers-format unet safetensors file.
///
/// The tensor keys use diffusers naming (not Burn naming) and satisfy the
/// rank-4 contract check for the Diffusion role.
fn write_diffusers_unet(path: &Path) {
    std::fs::create_dir_all(path.parent().expect("unet parent")).expect("unet dir");
    let tensors: Vec<(String, ZeroTensorView)> = vec![
        ("conv_in.weight".into(), tensor_view_f32(vec![320, 4, 3, 3])),
        ("conv_in.bias".into(), tensor_view_f32(vec![320])),
        ("conv_out.weight".into(), tensor_view_f32(vec![4, 320, 3, 3])),
        ("conv_out.bias".into(), tensor_view_f32(vec![4])),
    ];
    // No metadata — this is a raw diffusers file.
    serialize_to_file(tensors, None, path).expect("write unet safetensors");
}

/// Write a minimal diffusers-format vae safetensors file.
fn write_diffusers_vae(path: &Path) {
    std::fs::create_dir_all(path.parent().expect("vae parent")).expect("vae dir");
    let tensors: Vec<(String, ZeroTensorView)> = vec![
        ("conv_in.weight".into(), tensor_view_f32(vec![128, 4, 3, 3])),
        ("conv_in.bias".into(), tensor_view_f32(vec![128])),
        ("conv_norm_out.weight".into(), tensor_view_f32(vec![128])),
        ("conv_norm_out.bias".into(), tensor_view_f32(vec![128])),
        ("conv_out.weight".into(), tensor_view_f32(vec![3, 128, 3, 3])),
        ("conv_out.bias".into(), tensor_view_f32(vec![3])),
    ];
    serialize_to_file(tensors, None, path).expect("write vae safetensors");
}

/// Write a minimal diffusers-format text_encoder safetensors file.
fn write_diffusers_text_encoder(path: &Path) {
    std::fs::create_dir_all(path.parent().expect("text_encoder parent")).expect("text_encoder dir");
    let tensors: Vec<(String, ZeroTensorView)> = vec![
        (
            "token_embedding.weight".into(),
            tensor_view_f32(vec![49408, 768]),
        ),
        (
            "position_embedding.weight".into(),
            tensor_view_f32(vec![77, 768]),
        ),
    ];
    serialize_to_file(tensors, None, path).expect("write text_encoder safetensors");
}

/// Write a minimal diffusers-format text_encoder_2 safetensors file.
fn write_diffusers_text_encoder_2(path: &Path) {
    std::fs::create_dir_all(path
        .parent()
        .expect("text_encoder_2 parent"))
    .expect("text_encoder_2 dir");
    let tensors: Vec<(String, ZeroTensorView)> = vec![
        (
            "token_embedding.weight".into(),
            tensor_view_f32(vec![49408, 1280]),
        ),
        (
            "position_embedding.weight".into(),
            tensor_view_f32(vec![77, 1280]),
        ),
    ];
    serialize_to_file(tensors, None, path).expect("write text_encoder_2 safetensors");
}

// ---------------------------------------------------------------------------
// Source construction helpers
// ---------------------------------------------------------------------------

fn role_model_role(role: BurnSdxlComponentRole) -> ModelRole {
    match role {
        BurnSdxlComponentRole::Diffusion => ModelRole::DiffusionModel,
        BurnSdxlComponentRole::Vae => ModelRole::Vae,
        BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => {
            ModelRole::TextEncoder
        }
    }
}

/// Build a `ResolvedInferenceModelSource` pointing at a diffusers-format
/// split file in the standard directory layout.
fn diffusers_source(root: &Path, role: BurnSdxlComponentRole) -> ResolvedInferenceModelSource {
    let dir_name = match role {
        BurnSdxlComponentRole::Diffusion => "unet",
        BurnSdxlComponentRole::Vae => "vae",
        BurnSdxlComponentRole::TextEncoder => "text_encoder",
        BurnSdxlComponentRole::TextEncoder2 => "text_encoder_2",
    };
    let path = root.join(dir_name).join("model.safetensors");

    match role {
        BurnSdxlComponentRole::Diffusion => write_diffusers_unet(&path),
        BurnSdxlComponentRole::Vae => write_diffusers_vae(&path),
        BurnSdxlComponentRole::TextEncoder => write_diffusers_text_encoder(&path),
        BurnSdxlComponentRole::TextEncoder2 => write_diffusers_text_encoder_2(&path),
    }

    // Diffusers-format: no projection metadata, no burn.contract key.
    ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        role_model_role(role),
        path,
        ModelFormat::SafeTensors,
    )
}

fn resolved_model(root: &Path, sources: Vec<ResolvedInferenceModelSource>) -> ResolvedInferenceModel {
    ResolvedInferenceModel::new(
        ModelId::new("diffusers-direct-load-test"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        PathBuf::from(root),
        ModelFormat::SafeTensors,
    )
    .with_source_set(ResolvedInferenceModelSourceSet::from_sources(sources))
}

fn load_request(root: &Path, sources: Vec<ResolvedInferenceModelSource>) -> LoadBundleRequest {
    LoadBundleRequest::new(
        resolved_model(root, sources),
        run_id(),
        workflow_id(),
        workflow_version(),
        NodeId::new("diffusers-checkpoint-loader"),
    )
}

/// Helper to assert the error message from a failed load_bundle call
/// contains the expected substring.
fn assert_load_error_contains(err: InferenceError, expected: &str) {
    match err {
        InferenceError::BackendExecutionFailed { message } => {
            assert!(
                message.contains(expected),
                "expected error to contain `{expected}`, got: {message}"
            );
        }
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 4-component diffusers layout: unet + vae + text_encoder + text_encoder_2.
///
/// Verifies the validation gate accepts all four split components when
/// the `reimagine.contract` metadata key is absent (raw diffusers format).
#[tokio::test]
async fn diffusers_direct_load_accepts_4_component_layout() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let sources: Vec<_> = BurnSdxlComponentRole::all()
        .into_iter()
        .map(|role| diffusers_source(temp.path(), role))
        .collect();

    assert_eq!(sources.len(), 4, "must have 4 sources");

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("4-component diffusers layout should be accepted");

    assert_eq!(response.model().backend().as_str(), "burn");
    assert_eq!(response.model().model_id().as_str(), "diffusers-direct-load-test");
    assert_eq!(response.model().role(), ModelRole::CheckpointBundle);
}

/// 3-component diffusers layout: unet + vae + text_encoder only.
///
/// Verifies the validation gate accepts the minimal set of required
/// components (Diffusion + Vae) with one optional text encoder.
#[tokio::test]
async fn diffusers_direct_load_accepts_3_component_layout() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let sources: Vec<_> = [
        BurnSdxlComponentRole::Diffusion,
        BurnSdxlComponentRole::Vae,
        BurnSdxlComponentRole::TextEncoder,
    ]
    .into_iter()
    .map(|role| diffusers_source(temp.path(), role))
    .collect();

    assert_eq!(sources.len(), 3, "must have 3 sources");

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("3-component diffusers layout should be accepted");

    assert_eq!(response.model().backend().as_str(), "burn");
    assert_eq!(
        response.model().model_id().as_str(),
        "diffusers-direct-load-test"
    );
    assert_eq!(response.model().role(), ModelRole::CheckpointBundle);
}

/// 2-component diffusers layout: unet + vae only (no text encoders).
///
/// Verifies the validation gate accepts the absolute minimum: Diffusion
/// and Vae are required; text encoders are optional.
#[tokio::test]
async fn diffusers_direct_load_accepts_2_component_minimal() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let sources: Vec<_> = [
        BurnSdxlComponentRole::Diffusion,
        BurnSdxlComponentRole::Vae,
    ]
    .into_iter()
    .map(|role| diffusers_source(temp.path(), role))
    .collect();

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("2-component minimal diffusers layout should be accepted");

    assert_eq!(response.model().backend().as_str(), "burn");
    assert_eq!(response.model().role(), ModelRole::CheckpointBundle);
}

/// Verify `infer_role_from_path` mapping for the Diffusion role.
///
/// The unet/ directory should be mapped to the Diffusion component role.
/// We verify this indirectly: a source with `ModelRole::DiffusionModel`
/// pointing at `unet/model.safetensors` should succeed. If the role
/// inference is wrong, the role-pair validation will fail.
#[tokio::test]
async fn diffusers_direct_load_infers_diffusion_from_unet_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let role = BurnSdxlComponentRole::Diffusion;
    let sources = vec![diffusers_source(temp.path(), role)];

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("unet/ directory should infer Diffusion role");

    assert_eq!(response.model().backend().as_str(), "burn");
}

/// Verify `infer_role_from_path` mapping for the Vae role.
///
/// The vae/ directory should be mapped to the Vae component role.
#[tokio::test]
async fn diffusers_direct_load_infers_vae_from_vae_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let role = BurnSdxlComponentRole::Vae;
    let sources = vec![diffusers_source(temp.path(), role)];

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("vae/ directory should infer Vae role");

    assert_eq!(response.model().backend().as_str(), "burn");
}

/// Verify `infer_role_from_path` mapping for the TextEncoder role.
///
/// The text_encoder/ directory should be mapped to the TextEncoder
/// component role.
#[tokio::test]
async fn diffusers_direct_load_infers_text_encoder_from_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let role = BurnSdxlComponentRole::TextEncoder;
    let sources = vec![diffusers_source(temp.path(), role)];

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("text_encoder/ directory should infer TextEncoder role");

    assert_eq!(response.model().backend().as_str(), "burn");
}

/// Verify `infer_role_from_path` mapping for the TextEncoder2 role.
///
/// The text_encoder_2/ directory should be mapped to the TextEncoder2
/// component role.
#[tokio::test]
async fn diffusers_direct_load_infers_text_encoder_2_from_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let role = BurnSdxlComponentRole::TextEncoder2;
    let sources = vec![diffusers_source(temp.path(), role)];

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("text_encoder_2/ directory should infer TextEncoder2 role");

    assert_eq!(response.model().backend().as_str(), "burn");
}

/// A diffusers file in an unrecognized directory should fail.
///
/// `infer_role_from_path` should return `None` for directories that
/// don't match the known patterns (unet/, vae/, text_encoder/,
/// text_encoder_2/), and the validation gate should reject the source.
#[tokio::test]
async fn diffusers_direct_load_rejects_unrecognised_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    // Write a safetensors file in a bogus directory name.
    let path = temp
        .path()
        .join("bogus_component")
        .join("model.safetensors");
    std::fs::create_dir_all(path.parent().expect("bogus parent")).expect("bogus dir");
    let tensors: Vec<(String, ZeroTensorView)> = vec![
        ("weight".into(), tensor_view_f32(vec![4, 4, 3, 3])),
    ];
    serialize_to_file(tensors, None, &path).expect("write bogus safetensors");

    let source = ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        ModelRole::DiffusionModel,
        path,
        ModelFormat::SafeTensors,
    );
    let model = resolved_model(temp.path(), vec![source]);

    let err = backend
        .load_bundle(LoadBundleRequest::new(
            model,
            run_id(),
            workflow_id(),
            workflow_version(),
            NodeId::new("loader"),
        ))
        .await
        .expect_err("unrecognised directory should be rejected");

    assert_load_error_contains(err, "does not match a known component directory pattern");
}

/// Diffusers detection: files without the `reimagine.contract` metadata
/// key must be treated as diffusers format.
///
/// This is the fundamental gate test. We write safetensors with empty
/// metadata (None), which means `has_burn_component_metadata` returns
/// false. The validation path must fall through to `infer_role_from_path`.
#[tokio::test]
async fn diffusers_direct_load_detected_by_absent_contract_metadata() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    // Create 3 diffusers-format files with explicit None metadata.
    let unet_path = temp.path().join("unet").join("model.safetensors");
    let vae_path = temp.path().join("vae").join("model.safetensors");
    let te_path = temp
        .path()
        .join("text_encoder")
        .join("model.safetensors");

    write_diffusers_unet(&unet_path);
    write_diffusers_vae(&vae_path);
    write_diffusers_text_encoder(&te_path);

    // Explicitly verify no burn.contract metadata exists.
    let bytes = std::fs::read(&unet_path).expect("read unet file");
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).expect("read metadata");
    let meta_map = header.metadata().clone().unwrap_or_default();
    assert!(
        !meta_map.contains_key(metadata_keys::CONTRACT),
        "unet file must not contain burn.contract metadata for diffusers detection"
    );

    let sources = vec![
        ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::DiffusionModel,
            unet_path,
            ModelFormat::SafeTensors,
        ),
        ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::Vae,
            vae_path,
            ModelFormat::SafeTensors,
        ),
        ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::TextEncoder,
            te_path,
            ModelFormat::SafeTensors,
        ),
    ];

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("diffusers files without contract metadata should be accepted");

    assert_eq!(response.model().backend().as_str(), "burn");
}

/// Verify that a file with `reimagine.contract` metadata (Burn native
/// format) is accepted through the converted-format path, and that
/// diffusers files go through the inference path.
///
/// This tests both sides of the `has_burn_component_metadata` gate.
#[tokio::test]
async fn diffusers_direct_load_coexists_with_burn_native_format() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    // Create a Burn-native unet file WITH contract metadata.
    let burn_unet_path = temp.path().join("unet").join("model.safetensors");
    std::fs::create_dir_all(burn_unet_path.parent().expect("unet parent")).expect("unet dir");
    let burn_tensors: Vec<(String, ZeroTensorView)> = vec![
        ("conv_in.weight".into(), tensor_view_f32(vec![320, 4, 3, 3])),
        ("conv_in.bias".into(), tensor_view_f32(vec![320])),
        ("conv_out.weight".into(), tensor_view_f32(vec![4, 320, 3, 3])),
        ("conv_out.bias".into(), tensor_view_f32(vec![4])),
    ];
    let burn_meta = std::collections::HashMap::from([
        (metadata_keys::CONTRACT.to_owned(), "burn.component".to_owned()),
        (metadata_keys::CONTRACT_VERSION.to_owned(), "1".to_owned()),
        (metadata_keys::BACKEND.to_owned(), "burn".to_owned()),
        (metadata_keys::MODEL_SERIES.to_owned(), "stable_diffusion".to_owned()),
        (metadata_keys::VARIANT.to_owned(), "sdxl".to_owned()),
        (
            metadata_keys::COMPONENT_ROLE.to_owned(),
            BurnSdxlComponentRole::Diffusion.as_str().to_owned(),
        ),
        (
            metadata_keys::TENSOR_LAYOUT.to_owned(),
            "burn-module-snapshot".to_owned(),
        ),
        (metadata_keys::DTYPE_POLICY.to_owned(), "mixed".to_owned()),
    ]);
    serialize_to_file(burn_tensors, Some(burn_meta), &burn_unet_path).expect("write burn unet");

    // Create diffusers-format vae and text_encoder files (no contract metadata).
    let vae_path = temp.path().join("vae").join("model.safetensors");
    let te_path = temp
        .path()
        .join("text_encoder")
        .join("model.safetensors");
    write_diffusers_vae(&vae_path);
    write_diffusers_text_encoder(&te_path);

    let sources = vec![
        ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::DiffusionModel,
            burn_unet_path,
            ModelFormat::SafeTensors,
        ),
        ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::Vae,
            vae_path,
            ModelFormat::SafeTensors,
        ),
        ResolvedInferenceModelSource::new(
            ModelSourceKind::SplitComponent,
            ModelRole::TextEncoder,
            te_path,
            ModelFormat::SafeTensors,
        ),
    ];

    let response = backend
        .load_bundle(load_request(temp.path(), sources))
        .await
        .expect("mixed Burn-native and diffusers sources should be accepted");

    assert_eq!(response.model().backend().as_str(), "burn");
    assert_eq!(response.model().role(), ModelRole::CheckpointBundle);
}

/// Verify that the validation gate rejects a 5-component layout
/// (too many sources).
#[tokio::test]
async fn diffusers_direct_load_rejects_too_many_components() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    // Create 4 valid diffusers sources plus a duplicate Diffusion source
    // in a different directory name (which will fail infer_role_from_path
    // before we even hit the count check, so we test with 5 real sources).
    let mut sources: Vec<_> = BurnSdxlComponentRole::all()
        .into_iter()
        .map(|role| diffusers_source(temp.path(), role))
        .collect();

    // Add a 5th source with a different directory that maps to Diffusion.
    let extra_unet_path = temp
        .path()
        .join("extra_unet")
        .join("model.safetensors");
    std::fs::create_dir_all(extra_unet_path.parent().expect("extra parent")).expect("extra dir");
    let extra_tensors: Vec<(String, ZeroTensorView)> = vec![
        ("conv_in.weight".into(), tensor_view_f32(vec![4, 4, 3, 3])),
    ];
    serialize_to_file(extra_tensors, None, &extra_unet_path).expect("write extra unet");
    sources.push(ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        ModelRole::DiffusionModel,
        extra_unet_path,
        ModelFormat::SafeTensors,
    ));

    let model = resolved_model(temp.path(), sources);

    let err = backend
        .load_bundle(LoadBundleRequest::new(
            model,
            run_id(),
            workflow_id(),
            workflow_version(),
            NodeId::new("loader"),
        ))
        .await
        .expect_err("5-component layout should be rejected");

    assert_load_error_contains(err, "requires 3 or 4 converted SplitComponent sources");
}

/// Verify that the validation gate rejects a 1-component layout
/// (too few sources).
#[tokio::test]
async fn diffusers_direct_load_rejects_too_few_components() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let sources = vec![diffusers_source(
        temp.path(),
        BurnSdxlComponentRole::Diffusion,
    )];

    let model = resolved_model(temp.path(), sources);

    let err = backend
        .load_bundle(LoadBundleRequest::new(
            model,
            run_id(),
            workflow_id(),
            workflow_version(),
            NodeId::new("loader"),
        ))
        .await
        .expect_err("1-component layout should be rejected");

    assert_load_error_contains(err, "requires 3 or 4 converted SplitComponent sources");
}
