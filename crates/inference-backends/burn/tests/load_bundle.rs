use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, RunId, WorkflowId, WorkflowVersion,
};
use reimagine_inference::{
    BackendInstanceObservation, CreateEmptyLatentRequest, InferenceBackend, InferenceCapability,
    InferenceError, LoadBundleRequest, ModelFormat, ModelSourceKind, ResolvedInferenceModel,
    ResolvedInferenceModelSource, ResolvedInferenceModelSourceSet,
};
use reimagine_inference_burn::models::stable_diffusion::sdxl::{
    BurnSdxlComponentRole, metadata_keys,
};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};
use safetensors::tensor::{Dtype, View, serialize_to_file};

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

/// Default `burn:<label>` backend instance label expected for the
/// test backend under the active feature.
///
/// - `burn:wgpu:default` under `wgpu`.
/// - `burn:flex:cpu` under `flex`.
fn expected_default_instance() -> &'static str {
    #[cfg(feature = "wgpu")]
    {
        "burn:wgpu:default"
    }
    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    {
        "burn:flex:cpu"
    }
}

/// Default device short label.
fn expected_default_device_label() -> &'static str {
    #[cfg(feature = "wgpu")]
    {
        "wgpu:default"
    }
    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    {
        "flex:cpu"
    }
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
    component_metadata_with(role, "1")
}

fn component_metadata_with(
    role: BurnSdxlComponentRole,
    contract_version: &str,
) -> HashMap<String, String> {
    HashMap::from([
        (
            metadata_keys::CONTRACT.to_owned(),
            "burn.component".to_owned(),
        ),
        (
            metadata_keys::CONTRACT_VERSION.to_owned(),
            contract_version.to_owned(),
        ),
        (metadata_keys::BACKEND.to_owned(), "burn".to_owned()),
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
    write_component_with_metadata(path, role, component_metadata(role));
}

fn write_component_with_metadata(
    path: &Path,
    role: BurnSdxlComponentRole,
    metadata: HashMap<String, String>,
) {
    std::fs::create_dir_all(path.parent().expect("component parent")).expect("component dir");
    // Use the full executable spec set for text-encoder components
    // so the runtime validation (validate_component_inventory_full)
    // passes with all transformer-block keys present.
    let specs: Vec<(String, Vec<usize>)> = match role {
        BurnSdxlComponentRole::TextEncoder | BurnSdxlComponentRole::TextEncoder2 => role
            .contract()
            .all_expected_tensor_specs()
            .into_iter()
            .filter(|s| s.required)
            .map(|s| {
                // The test fixture writes a single-element zero
                // buffer for each text-encoder tensor. Real CLIP
                // forward (burn/14c) treats zero-data weights as a
                // no-op so the surrounding plumbing can be exercised
                // without materializing the full vocab × width
                // embedding tensors. The validator checks tensor
                // rank (not specific dim values), so use
                // `vec![1; rank]` as the shape and a 1-element
                // zero buffer.
                let rank = s.shape.rank();
                (s.key.clone(), vec![1usize; rank.max(1)])
            })
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

    serialize_to_file(tensors, Some(metadata), path).expect("component file");
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

fn split_component_source_with_metadata(
    root: &Path,
    role: BurnSdxlComponentRole,
    metadata: String,
) -> ResolvedInferenceModelSource {
    let path = root.join(role.as_str()).join("model.safetensors");
    write_component(&path, role);

    ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        role_model_role(role),
        path,
        ModelFormat::SafeTensors,
    )
    .with_metadata(metadata)
}

fn resolved_model_from_sources(
    root: &Path,
    sources: Vec<ResolvedInferenceModelSource>,
) -> ResolvedInferenceModel {
    ResolvedInferenceModel::new(
        ModelId::new("sdxl-base-burn"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
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
        RunId::new("run-burn-load"),
        WorkflowId::new("workflow-burn-load"),
        WorkflowVersion::new(1),
        NodeId::new("checkpoint-loader"),
    )
}

fn request_from_model(model: ResolvedInferenceModel) -> LoadBundleRequest {
    LoadBundleRequest::new(
        model,
        RunId::new("run-burn-load"),
        WorkflowId::new("workflow-burn-load"),
        WorkflowVersion::new(1),
        NodeId::new("checkpoint-loader"),
    )
}

fn assert_backend_execution_failed_contains(err: InferenceError, expected: &str) {
    match err {
        InferenceError::BackendExecutionFailed { message } => {
            assert!(
                message.contains(expected),
                "expected `{message}` to contain `{expected}`"
            );
        }
        other => panic!("expected BackendExecutionFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn load_bundle_accepts_converted_burn_native_sdxl_components() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();

    let response = backend
        .load_bundle(load_request(temp.path()))
        .await
        .expect("load converted components");

    let capabilities = backend.capabilities();
    assert!(capabilities.supports_capability(InferenceCapability::LoadBundle));

    assert_eq!(response.model().backend().as_str(), "burn");
    assert_eq!(
        response.model().backend_instance().as_str(),
        expected_default_instance()
    );
    assert_eq!(
        response.model().device_label(),
        Some(expected_default_device_label())
    );
    assert_eq!(response.model().model_id().as_str(), "sdxl-base-burn");
    assert_eq!(response.model().role(), ModelRole::CheckpointBundle);
    assert_eq!(
        response.model().payload_key().as_str(),
        "burn:model:sdxl-base-burn:diffusion"
    );

    assert_eq!(response.clip().backend().as_str(), "burn");
    assert_eq!(
        response.clip().backend_instance().as_str(),
        expected_default_instance()
    );
    assert_eq!(
        response.clip().device_label(),
        Some(expected_default_device_label())
    );
    assert_eq!(
        response.clip().payload_key().as_str(),
        "burn:model:sdxl-base-burn:clip"
    );

    assert_eq!(response.vae().backend().as_str(), "burn");
    assert_eq!(
        response.vae().backend_instance().as_str(),
        expected_default_instance()
    );
    assert_eq!(
        response.vae().device_label(),
        Some(expected_default_device_label())
    );
    assert_eq!(
        response.vae().payload_key().as_str(),
        "burn:model:sdxl-base-burn:vae"
    );
}

#[tokio::test]
async fn load_bundle_rejects_checkpoint_bundle_sources() {
    let temp = tempfile::tempdir().expect("tempdir");
    let checkpoint = temp.path().join("checkpoint.safetensors");
    write_component(&checkpoint, BurnSdxlComponentRole::Diffusion);
    let source = ResolvedInferenceModelSource::new(
        ModelSourceKind::CheckpointBundle,
        ModelRole::CheckpointBundle,
        checkpoint,
        ModelFormat::SafeTensors,
    );
    let model = resolved_model_from_sources(temp.path(), vec![source]);

    let err = backend()
        .load_bundle(request_from_model(model))
        .await
        .expect_err("checkpoint bundle source is not a Burn runtime input");

    assert_backend_execution_failed_contains(err, "checkpoint bundle sources must be converted");
}

#[tokio::test]
async fn load_bundle_rejects_missing_component_without_cache_mutation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();
    let sources = BurnSdxlComponentRole::all()
        .into_iter()
        .filter(|role| *role != BurnSdxlComponentRole::TextEncoder2)
        .map(|role| split_component_source(temp.path(), role))
        .collect::<Vec<_>>();
    let model = resolved_model_from_sources(temp.path(), sources);

    let err = backend
        .load_bundle(request_from_model(model))
        .await
        .expect_err("missing component should fail");

    assert_backend_execution_failed_contains(err, "requires exactly 4");
    assert_eq!(backend.model_cache().bundle_count(), 0);
}

#[tokio::test]
async fn load_bundle_rejects_duplicate_component_labels_without_cache_mutation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();
    let mut sources = valid_split_sources(temp.path());
    let duplicate_path = temp
        .path()
        .join("duplicate-text-encoder")
        .join("model.safetensors");
    write_component(&duplicate_path, BurnSdxlComponentRole::TextEncoder);
    sources[3] = ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        ModelRole::TextEncoder,
        duplicate_path,
        ModelFormat::SafeTensors,
    )
    .with_metadata(resolver_metadata(BurnSdxlComponentRole::TextEncoder));
    let model = resolved_model_from_sources(temp.path(), sources);

    let err = backend
        .load_bundle(request_from_model(model))
        .await
        .expect_err("duplicate component should fail");

    assert_backend_execution_failed_contains(err, "duplicate Burn SDXL component `text_encoder`");
    assert_eq!(backend.model_cache().bundle_count(), 0);
}

#[tokio::test]
async fn load_bundle_rejects_resolver_metadata_conflict_with_file_metadata() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut sources = valid_split_sources(temp.path());
    sources[0] = split_component_source_with_metadata(
        temp.path(),
        BurnSdxlComponentRole::Diffusion,
        resolver_metadata(BurnSdxlComponentRole::Vae),
    );
    let model = resolved_model_from_sources(temp.path(), sources);

    let err = backend()
        .load_bundle(request_from_model(model))
        .await
        .expect_err("resolver/file component conflict should fail");

    assert_backend_execution_failed_contains(err, "metadata mismatch");
}

#[tokio::test]
async fn load_bundle_rejects_file_contract_version_mismatch_without_replacing_cached_bundle() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();
    backend
        .load_bundle(load_request(temp.path()))
        .await
        .expect("initial load");
    assert_eq!(backend.model_cache().bundle_count(), 1);

    let replacement = tempfile::tempdir().expect("replacement tempdir");
    let mut sources = valid_split_sources(replacement.path());
    let bad_path = replacement
        .path()
        .join(BurnSdxlComponentRole::Vae.as_str())
        .join("model.safetensors");
    write_component_with_metadata(
        &bad_path,
        BurnSdxlComponentRole::Vae,
        component_metadata_with(BurnSdxlComponentRole::Vae, "2"),
    );
    sources[1] = ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        ModelRole::Vae,
        bad_path,
        ModelFormat::SafeTensors,
    )
    .with_metadata(resolver_metadata(BurnSdxlComponentRole::Vae));
    let model = resolved_model_from_sources(replacement.path(), sources);

    let err = backend
        .load_bundle(request_from_model(model))
        .await
        .expect_err("replacement with invalid contract should fail");

    assert_backend_execution_failed_contains(
        err,
        "unsupported Burn SDXL component contract version",
    );
    assert_eq!(backend.model_cache().bundle_count(), 1);
}

#[tokio::test]
async fn runtime_hooks_snapshot_reports_cached_model_count_after_load() {
    let temp = tempfile::tempdir().expect("tempdir");
    let backend = backend();
    let hooks = backend.runtime_hooks(None, None, None);

    backend
        .load_bundle(load_request(temp.path()))
        .await
        .expect("load bundle");
    let snapshot = hooks.snapshot().await;

    assert_eq!(
        snapshot.observations.get("cached_models"),
        Some(&"1".to_owned())
    );
    assert_eq!(
        snapshot.observations.get("run_payloads"),
        Some(&"0".to_owned())
    );
}

#[tokio::test]
async fn downstream_capabilities_remain_not_implemented_except_create_empty_latent() {
    // burn/05 originally asserted every downstream capability
    // (including CreateEmptyLatent) was BackendNotImplemented.
    // burn/09 ships a real V1 `latent.create_empty` implementation
    // for the burn backend, so CreateEmptyLatent is now expected
    // to succeed. Other downstream capabilities (text_encode,
    // diffusion_sample, latent_decode/encode, image_*)
    // remain BackendNotImplemented until their dedicated issues
    // land.
    let backend = backend();
    let request = CreateEmptyLatentRequest::new(
        512,
        512,
        1,
        RunId::new("run-burn-downstream"),
        WorkflowId::new("workflow-burn-downstream"),
        WorkflowVersion::new(1),
        NodeId::new("latent"),
    );

    let response = backend
        .create_empty_latent(request)
        .await
        .expect("burn/09 implements latent.create_empty");

    let latent = response.into_latent();
    assert_eq!(
        latent.latent_space().id().as_str(),
        "stable_diffusion/sdxl/base"
    );
    assert_eq!(latent.payload().backend().as_str(), "burn");
    assert_eq!(
        latent.payload().backend_instance().as_str(),
        expected_default_instance()
    );
    assert_eq!(latent.payload().shape().dims(), &[1_usize, 4, 64, 64]);
}
