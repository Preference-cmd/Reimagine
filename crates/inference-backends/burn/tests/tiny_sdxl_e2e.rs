//! End-to-end fixture for a tiny Burn-native SDXL component package.
//!
//! This test intentionally exercises the public capability chain instead of
//! crate-private helpers:
//!
//! load_bundle -> text.encode -> latent.create_empty -> diffusion.sample ->
//! latent.decode -> image.preview/image.save

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use reimagine_core::model::{
    ModelId, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId, WorkflowId,
    WorkflowVersion,
};
use reimagine_inference::{
    CreateEmptyLatentRequest, DiffusionSampleRequest, ExecutionValue, ImagePreviewRequest,
    ImageSaveRequest, InferenceBackend, LatentDecodeRequest, LoadBundleRequest, ModelFormat,
    ModelSourceKind, ResolvedInferenceModel, ResolvedInferenceModelSource,
    ResolvedInferenceModelSourceSet, SamplerName, SchedulerName, TextEncodeRequest,
};
use reimagine_inference_burn::models::stable_diffusion::sdxl::{
    BurnSdxlComponentRole, metadata_keys,
};
use reimagine_inference_burn::{BurnBackend, BurnBackendConfig};
use safetensors::tensor::{Dtype, View, serialize_to_file};

const MODEL_ID: &str = "tiny-sdxl-burn";
const RUN_ID: &str = "run-tiny-sdxl";
const WORKFLOW_ID: &str = "wf-tiny-sdxl";
const FIXTURE_PROFILE_KEY: &str = "reimagine.fixture_profile";
const FIXTURE_PROFILE_VALUE: &str = "tiny_sdxl_e2e";
const TINY_TEXT_WIDTH: usize = 8;
const TINY_TEXT_INNER_WIDTH: usize = 32;
const TINY_TEXT_VOCAB: usize = 49_408;
const TINY_TEXT_SEQUENCE: usize = 77;

#[derive(Debug, Clone)]
struct F32TensorView {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for F32TensorView {
    fn dtype(&self) -> Dtype {
        Dtype::F32
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

#[tokio::test]
async fn tiny_sdxl_component_package_runs_public_burn_capability_chain() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let backend =
        BurnBackend::new(BurnBackendConfig::new(root.path(), output.path())).expect("burn backend");

    let loaded = backend
        .load_bundle(load_request(root.path()))
        .await
        .expect("load tiny fixture bundle");

    let positive = backend
        .text_encode(text_request(
            loaded.clip().clone(),
            "small bright city at sunrise",
            "text-positive",
        ))
        .await
        .expect("positive text.encode")
        .into_conditioning();
    assert_eq!(
        positive.text_embedding().shape().dims(),
        &[1_usize, TINY_TEXT_SEQUENCE, TINY_TEXT_WIDTH * 2],
        "tiny fixture must use component-backed CLIP modules, not synthetic full SDXL placeholders"
    );
    assert_eq!(
        positive
            .pooled_embedding()
            .expect("pooled embedding")
            .shape()
            .dims(),
        &[1_usize, TINY_TEXT_WIDTH]
    );
    let negative = backend
        .text_encode(text_request(
            loaded.clip().clone(),
            "low quality blur",
            "text-negative",
        ))
        .await
        .expect("negative text.encode")
        .into_conditioning();

    let latent = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            run_id(),
            workflow_id(),
            workflow_version(),
            NodeId::new("latent-empty"),
        ))
        .await
        .expect("latent.create_empty")
        .into_latent();

    let sampled = backend
        .diffusion_sample(DiffusionSampleRequest::new(
            loaded.model().clone(),
            positive,
            negative,
            latent,
            1234,
            1,
            1.0,
            SamplerName::Euler,
            SchedulerName::Normal,
            1.0,
            run_id(),
            workflow_id(),
            workflow_version(),
            NodeId::new("diffusion"),
        ))
        .await
        .expect("diffusion.sample")
        .into_latent();

    let image = backend
        .latent_decode(LatentDecodeRequest::new(
            loaded.vae().clone(),
            sampled,
            run_id(),
            workflow_id(),
            workflow_version(),
            NodeId::new("vae-decode"),
        ))
        .await
        .expect("latent.decode")
        .into_image();

    let preview = backend
        .image_preview(ImagePreviewRequest::new(
            image.clone(),
            run_id(),
            workflow_id(),
            workflow_version(),
            NodeId::new("preview"),
        ))
        .await
        .expect("image.preview")
        .into_artifact();
    let saved = backend
        .image_save(
            ImageSaveRequest::new(
                image,
                run_id(),
                workflow_id(),
                workflow_version(),
                NodeId::new("save"),
            )
            .with_filename_prefix("tiny-sdxl"),
        )
        .await
        .expect("image.save")
        .into_artifact();

    assert_png_artifact(output.path(), preview.as_str());
    assert_png_artifact(output.path(), saved.as_str());
}

fn load_request(root: &Path) -> LoadBundleRequest {
    LoadBundleRequest::new(
        resolved_model(root),
        run_id(),
        workflow_id(),
        workflow_version(),
        NodeId::new("load-bundle"),
    )
}

fn resolved_model(root: &Path) -> ResolvedInferenceModel {
    let sources = BurnSdxlComponentRole::all()
        .into_iter()
        .map(|role| split_component_source(root, role))
        .collect();

    ResolvedInferenceModel::new(
        ModelId::new(MODEL_ID),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
        PathBuf::from(root),
        ModelFormat::SafeTensors,
    )
    .with_source_set(ResolvedInferenceModelSourceSet::from_sources(sources))
}

fn split_component_source(
    root: &Path,
    role: BurnSdxlComponentRole,
) -> ResolvedInferenceModelSource {
    let path = root.join(role.as_str()).join("model.safetensors");
    match role {
        BurnSdxlComponentRole::TextEncoder => {
            write_text_component(&path, role, "model.text_encoder")
        }
        BurnSdxlComponentRole::TextEncoder2 => {
            write_text_component(&path, role, "model.text_encoder_2")
        }
        BurnSdxlComponentRole::Diffusion => write_diffusion_component(&path),
        BurnSdxlComponentRole::Vae => write_vae_component(&path),
    }

    ResolvedInferenceModelSource::new(
        ModelSourceKind::SplitComponent,
        role_model_role(role),
        path,
        ModelFormat::SafeTensors,
    )
    .with_metadata(resolver_metadata(role))
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

fn component_metadata(role: BurnSdxlComponentRole) -> HashMap<String, String> {
    HashMap::from([
        (
            metadata_keys::CONTRACT.to_owned(),
            "burn.component".to_owned(),
        ),
        (metadata_keys::CONTRACT_VERSION.to_owned(), "1".to_owned()),
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
        (
            FIXTURE_PROFILE_KEY.to_owned(),
            FIXTURE_PROFILE_VALUE.to_owned(),
        ),
    ])
}

fn write_text_component(path: &Path, role: BurnSdxlComponentRole, prefix: &str) {
    let mut tensors = vec![
        tensor(
            &format!("{prefix}.token_embedding.weight"),
            vec![TINY_TEXT_VOCAB, TINY_TEXT_WIDTH],
            repeating_values(TINY_TEXT_VOCAB * TINY_TEXT_WIDTH, 0.001),
        ),
        tensor(
            &format!("{prefix}.position_embedding.weight"),
            vec![TINY_TEXT_SEQUENCE, TINY_TEXT_WIDTH],
            repeating_values(TINY_TEXT_SEQUENCE * TINY_TEXT_WIDTH, 0.002),
        ),
        tensor(
            &format!("{prefix}.final_layer_norm.gamma"),
            vec![TINY_TEXT_WIDTH],
            vec![1.0; TINY_TEXT_WIDTH],
        ),
        tensor(
            &format!("{prefix}.final_layer_norm.beta"),
            vec![TINY_TEXT_WIDTH],
            vec![0.0; TINY_TEXT_WIDTH],
        ),
    ];

    if role == BurnSdxlComponentRole::TextEncoder2 {
        tensors.push(tensor(
            &format!("{prefix}.text_projection.weight"),
            vec![TINY_TEXT_WIDTH, TINY_TEXT_WIDTH],
            identity(TINY_TEXT_WIDTH),
        ));
        tensors.push(tensor(
            &format!("{prefix}.text_projection.bias"),
            vec![TINY_TEXT_WIDTH],
            vec![0.0; TINY_TEXT_WIDTH],
        ));
    }

    tensors.extend(transformer_block_tensors(prefix));
    write_tensors(path, role, tensors);
}

fn transformer_block_tensors(prefix: &str) -> Vec<(String, F32TensorView)> {
    let block = format!("{prefix}.transformer.resblocks.0");
    vec![
        tensor(
            &format!("{block}.ln_1.weight"),
            vec![TINY_TEXT_WIDTH],
            vec![1.0; TINY_TEXT_WIDTH],
        ),
        tensor(
            &format!("{block}.ln_1.bias"),
            vec![TINY_TEXT_WIDTH],
            vec![0.0; TINY_TEXT_WIDTH],
        ),
        tensor(
            &format!("{block}.ln_2.weight"),
            vec![TINY_TEXT_WIDTH],
            vec![1.0; TINY_TEXT_WIDTH],
        ),
        tensor(
            &format!("{block}.ln_2.bias"),
            vec![TINY_TEXT_WIDTH],
            vec![0.0; TINY_TEXT_WIDTH],
        ),
        tensor(
            &format!("{block}.attn.in_proj_weight"),
            vec![TINY_TEXT_WIDTH * 3, TINY_TEXT_WIDTH],
            repeating_values(TINY_TEXT_WIDTH * TINY_TEXT_WIDTH * 3, 0.003),
        ),
        tensor(
            &format!("{block}.attn.in_proj_bias"),
            vec![TINY_TEXT_WIDTH * 3],
            vec![0.0; TINY_TEXT_WIDTH * 3],
        ),
        tensor(
            &format!("{block}.attn.out_proj.weight"),
            vec![TINY_TEXT_WIDTH, TINY_TEXT_WIDTH],
            identity(TINY_TEXT_WIDTH),
        ),
        tensor(
            &format!("{block}.attn.out_proj.bias"),
            vec![TINY_TEXT_WIDTH],
            vec![0.0; TINY_TEXT_WIDTH],
        ),
        tensor(
            &format!("{block}.mlp.fc1.weight"),
            vec![TINY_TEXT_INNER_WIDTH, TINY_TEXT_WIDTH],
            repeating_values(TINY_TEXT_INNER_WIDTH * TINY_TEXT_WIDTH, 0.004),
        ),
        tensor(
            &format!("{block}.mlp.fc1.bias"),
            vec![TINY_TEXT_INNER_WIDTH],
            vec![0.0; TINY_TEXT_INNER_WIDTH],
        ),
        tensor(
            &format!("{block}.mlp.fc2.weight"),
            vec![TINY_TEXT_WIDTH, TINY_TEXT_INNER_WIDTH],
            repeating_values(TINY_TEXT_WIDTH * TINY_TEXT_INNER_WIDTH, 0.005),
        ),
        tensor(
            &format!("{block}.mlp.fc2.bias"),
            vec![TINY_TEXT_WIDTH],
            vec![0.0; TINY_TEXT_WIDTH],
        ),
    ]
}

fn write_diffusion_component(path: &Path) {
    write_tensors(
        path,
        BurnSdxlComponentRole::Diffusion,
        vec![
            tensor(
                "model.diffusion.input_blocks.0.0.weight",
                vec![4, 4, 3, 3],
                zeros(4 * 4 * 3 * 3),
            ),
            tensor(
                "model.diffusion.time_embed.0.weight",
                vec![8, 4],
                zeros(8 * 4),
            ),
            tensor(
                "model.diffusion.conv_in.weight",
                vec![4, 4, 3, 3],
                zeros(4 * 4 * 3 * 3),
            ),
            tensor("model.diffusion.conv_in.bias", vec![4], vec![0.0; 4]),
            tensor(
                "model.diffusion.out.0.weight",
                vec![4, 4, 3, 3],
                zeros(4 * 4 * 3 * 3),
            ),
            tensor("model.diffusion.out.0.bias", vec![4], vec![0.0; 4]),
        ],
    );
}

fn write_vae_component(path: &Path) {
    write_tensors(
        path,
        BurnSdxlComponentRole::Vae,
        vec![
            tensor(
                "model.vae.encoder.conv_in.weight",
                vec![4, 3, 3, 3],
                zeros(4 * 3 * 3 * 3),
            ),
            tensor(
                "model.vae.decoder.conv_out.weight",
                vec![3, 4, 3, 3],
                zeros(3 * 4 * 3 * 3),
            ),
            tensor("model.vae.decoder.conv_out.bias", vec![3], vec![0.0; 3]),
        ],
    );
}

fn write_tensors(path: &Path, role: BurnSdxlComponentRole, tensors: Vec<(String, F32TensorView)>) {
    std::fs::create_dir_all(path.parent().expect("component parent")).expect("component dir");
    serialize_to_file(tensors, Some(component_metadata(role)), path).expect("component file");
}

fn tensor(name: &str, shape: Vec<usize>, values: Vec<f32>) -> (String, F32TensorView) {
    assert_eq!(
        shape.iter().product::<usize>(),
        values.len(),
        "tensor {name} value count matches shape"
    );
    let data = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    (name.to_owned(), F32TensorView { shape, data })
}

fn zeros(count: usize) -> Vec<f32> {
    vec![0.0; count]
}

fn repeating_values(count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|idx| ((idx % 17) as f32 + 1.0) * scale)
        .collect()
}

fn identity(width: usize) -> Vec<f32> {
    let mut values = vec![0.0; width * width];
    for idx in 0..width {
        values[idx * width + idx] = 1.0;
    }
    values
}

fn text_request(
    clip: reimagine_inference::RuntimeClipHandle,
    prompt: &str,
    node: &str,
) -> TextEncodeRequest {
    TextEncodeRequest::new(
        clip,
        Arc::new(ExecutionValue::Param(ParamValue::String(prompt.to_owned()))),
        run_id(),
        workflow_id(),
        workflow_version(),
        NodeId::new(node),
    )
}

fn run_id() -> RunId {
    RunId::new(RUN_ID)
}

fn workflow_id() -> WorkflowId {
    WorkflowId::new(WORKFLOW_ID)
}

fn workflow_version() -> WorkflowVersion {
    WorkflowVersion::new(1)
}

fn assert_png_artifact(output_dir: &Path, artifact: &str) {
    let relative = artifact
        .strip_prefix("output/")
        .expect("artifact is output-relative");
    let path = output_dir.join(relative);
    let bytes = std::fs::read(&path).unwrap_or_else(|err| {
        panic!("read PNG artifact `{}`: {err}", path.display());
    });
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"), "PNG header");
    let image = image::load_from_memory(&bytes).expect("valid PNG");
    assert_eq!(image.width(), 64);
    assert_eq!(image.height(), 64);
}
