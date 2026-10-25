//! Full SDXL pipeline integration test through `reimagine-app-host`.
//!
//! Exercises the complete capability chain directly through `WorkspaceHost`
//! with a real `reimagine-inference-burn-worker` subprocess:
//!
//!   load model (builtin.checkpoint_loader) -> text encode (builtin.clip_text_encode x2)
//!   -> create latent (builtin.empty_latent_image) -> sample (builtin.ksampler)
//!   -> decode (builtin.vae_decode) -> save (builtin.save_image)
//!
//! The model weights are a synthetic tiny SDXL component package written at
//! test time (same fixture shape as `crates/inference-backends/burn/tests/
//! tiny_sdxl_e2e.rs`): no real weights, no network, runs on the default wgpu
//! device. The test asserts the run completes and the produced artifact is a
//! 64x64 PNG (signature + decoded dimensions).
//!
//! Worker binary resolution order:
//!   1. `REIMAGINE_BURN_WORKER` env var (explicit override)
//!   2. `CARGO_BIN_EXE_reimagine-inference-burn-worker` (set when this test is
//!      built by the burn-worker package)
//!   3. `<target>/<debug|release>/reimagine-inference-burn-worker` from the
//!      workspace target dir (honors `CARGO_TARGET_DIR`)
//!   4. On-demand `cargo build -p reimagine-inference-burn-worker` fallback
//!
//! If the binary cannot be resolved the test skips (logs why); it never
//! requires an existing converted SDXL workspace.

#![cfg(feature = "aigc")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reimagine_app_host::WorkspaceScope;
use reimagine_app_host::{
    ModelService, RunWorkflowRequest, RunWorkflowResult, StaticWorkerInventoryProvider,
    WorkerBackendCandidate, WorkerInventorySnapshot, WorkspaceHost,
};
use reimagine_backend_worker_host::{ExpectedWorkerIdentity, WorkerLaunchSpec, WorkerLimits};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, ProtocolRange, WorkerInstallationId, WorkerInstanceProfile,
};
use reimagine_config::{AppPaths, InferenceBackendConfig, InferenceBackendKind};
use reimagine_core::event::RunEventKind;
use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant, NodeId};
use reimagine_core::readiness::{RunTarget, RunTargetSelection};
use reimagine_model_manager::{
    ModelComponentSource, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelRootId,
    ModelSource, ModelSourceStatus,
};
use reimagine_runtime::TaskState;
mod common;
use common::TestRunEventCollector;
use reimagine_core::event::EventBus;
use serde_json::Value;

const WORKFLOW_ID: &str = "wf-burn-full-pipeline";
const MODEL_ID: &str = "tiny-sdxl-burn";
const INSTANCE_LABEL: &str = "burn:wgpu:default";
const WORKER_BINARY: &str = "reimagine-inference-burn-worker";
const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

/// Writes the synthetic tiny SDXL component package (4 safetensors files)
/// under `models_dir()/tiny-sdxl-burn/<role>/model.safetensors`.
mod tiny_fixture {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::path::Path;

    use safetensors::tensor::{Dtype, View, serialize_to_file};

    const ROLES: [&str; 4] = ["diffusion", "vae", "text_encoder", "text_encoder_2"];
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

    pub(crate) fn write_package(models_dir: &Path) {
        for role in ROLES {
            write_component(&models_dir.join("tiny-sdxl-burn").join(role), role);
        }
    }

    fn component_metadata(role: &str) -> HashMap<String, String> {
        HashMap::from([
            ("reimagine.contract".to_owned(), "burn.component".to_owned()),
            ("reimagine.contract_version".to_owned(), "1".to_owned()),
            ("reimagine.backend".to_owned(), "burn".to_owned()),
            (
                "reimagine.model_series".to_owned(),
                "stable_diffusion".to_owned(),
            ),
            ("reimagine.variant".to_owned(), "sdxl".to_owned()),
            ("reimagine.component_role".to_owned(), role.to_owned()),
            (
                "reimagine.tensor_layout".to_owned(),
                "burn-module-snapshot".to_owned(),
            ),
            ("reimagine.dtype_policy".to_owned(), "mixed".to_owned()),
            (
                "reimagine.fixture_profile".to_owned(),
                "tiny_sdxl_e2e".to_owned(),
            ),
        ])
    }

    fn write_component(dir: &Path, role: &str) {
        let path = dir.join("model.safetensors");
        std::fs::create_dir_all(dir).expect("component dir");
        let tensors = match role {
            "diffusion" => diffusion_tensors(),
            "vae" => vae_tensors(),
            "text_encoder" => text_tensors("model.text_encoder", false),
            "text_encoder_2" => text_tensors("model.text_encoder_2", true),
            other => panic!("unknown tiny fixture role: {other}"),
        };
        serialize_to_file(tensors, Some(component_metadata(role)), &path).expect("component file");
    }

    fn text_tensors(prefix: &str, has_projection: bool) -> Vec<(String, F32TensorView)> {
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

        if has_projection {
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

        let block = format!("{prefix}.transformer.resblocks.0");
        tensors.extend([
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
        ]);
        tensors
    }

    fn diffusion_tensors() -> Vec<(String, F32TensorView)> {
        vec![
            tensor("conv_in.weight", vec![4, 4, 3, 3], zeros(4 * 4 * 3 * 3)),
            tensor("conv_in.bias", vec![4], vec![0.0; 4]),
            tensor("conv_out.weight", vec![4, 4, 3, 3], zeros(4 * 4 * 3 * 3)),
            tensor("conv_out.bias", vec![4], vec![0.0; 4]),
        ]
    }

    fn vae_tensors() -> Vec<(String, F32TensorView)> {
        let mut tensors = Vec::new();
        // conv_in: [out=512, in=4, 3, 3]
        tensors.push(tensor(
            "conv_in.weight",
            vec![512, 4, 3, 3],
            zeros(512 * 4 * 3 * 3),
        ));
        tensors.push(tensor("conv_in.bias", vec![512], vec![0.0; 512]));
        // mid_block.resnets.0/1: 512->512
        for rn in 0..2 {
            for nm in ["norm1", "norm2"] {
                tensors.push(tensor(
                    &format!("mid_block.resnets.{rn}.{nm}.weight"),
                    vec![512],
                    vec![1.0; 512],
                ));
                tensors.push(tensor(
                    &format!("mid_block.resnets.{rn}.{nm}.bias"),
                    vec![512],
                    vec![0.0; 512],
                ));
            }
            for cv in ["conv1", "conv2"] {
                tensors.push(tensor(
                    &format!("mid_block.resnets.{rn}.{cv}.weight"),
                    vec![512, 512, 3, 3],
                    zeros(512 * 512 * 3 * 3),
                ));
                tensors.push(tensor(
                    &format!("mid_block.resnets.{rn}.{cv}.bias"),
                    vec![512],
                    vec![0.0; 512],
                ));
            }
        }
        // mid_block.attentions.0: 512->512 (diffusers dialect)
        for tk in ["to_q", "to_k", "to_v"] {
            tensors.push(tensor(
                &format!("mid_block.attentions.0.{tk}.weight"),
                vec![512, 512, 1, 1],
                zeros(512 * 512),
            ));
            tensors.push(tensor(
                &format!("mid_block.attentions.0.{tk}.bias"),
                vec![512],
                vec![0.0; 512],
            ));
        }
        tensors.push(tensor(
            "mid_block.attentions.0.to_out.0.weight",
            vec![512, 512, 1, 1],
            zeros(512 * 512),
        ));
        tensors.push(tensor(
            "mid_block.attentions.0.to_out.0.bias",
            vec![512],
            vec![0.0; 512],
        ));
        tensors.push(tensor(
            "mid_block.attentions.0.group_norm.weight",
            vec![512],
            vec![1.0; 512],
        ));
        tensors.push(tensor(
            "mid_block.attentions.0.group_norm.bias",
            vec![512],
            vec![0.0; 512],
        ));
        // up_blocks: 4 blocks, 3 resnets each; first two resnets of each block
        // have skip connections; blocks 0-2 upsample.
        let up_block_channels: [(usize, usize, bool); 4] = [
            (512, 512, true),
            (512, 512, true),
            (512, 256, true),
            (256, 128, false),
        ];
        for (block_idx, (in_ch, out_ch, has_up)) in up_block_channels.iter().enumerate() {
            for rn in 0..3 {
                let is_first_with_skip = rn == 0 && *in_ch != *out_ch;
                let res_in_ch = if is_first_with_skip { *in_ch } else { *out_ch };
                for nm in ["norm1", "norm2"] {
                    let ch = if nm == "norm1" { res_in_ch } else { *out_ch };
                    tensors.push(tensor(
                        &format!("up_blocks.{block_idx}.resnets.{rn}.{nm}.weight"),
                        vec![ch],
                        vec![1.0; ch],
                    ));
                    tensors.push(tensor(
                        &format!("up_blocks.{block_idx}.resnets.{rn}.{nm}.bias"),
                        vec![ch],
                        vec![0.0; ch],
                    ));
                }
                tensors.push(tensor(
                    &format!("up_blocks.{block_idx}.resnets.{rn}.conv1.weight"),
                    vec![*out_ch, res_in_ch, 3, 3],
                    zeros(out_ch * res_in_ch * 3 * 3),
                ));
                tensors.push(tensor(
                    &format!("up_blocks.{block_idx}.resnets.{rn}.conv1.bias"),
                    vec![*out_ch],
                    vec![0.0; *out_ch],
                ));
                tensors.push(tensor(
                    &format!("up_blocks.{block_idx}.resnets.{rn}.conv2.weight"),
                    vec![*out_ch, *out_ch, 3, 3],
                    zeros(out_ch * out_ch * 3 * 3),
                ));
                tensors.push(tensor(
                    &format!("up_blocks.{block_idx}.resnets.{rn}.conv2.bias"),
                    vec![*out_ch],
                    vec![0.0; *out_ch],
                ));
                if is_first_with_skip {
                    tensors.push(tensor(
                        &format!("up_blocks.{block_idx}.resnets.{rn}.conv_shortcut.weight"),
                        vec![*out_ch, *in_ch, 1, 1],
                        zeros(out_ch * in_ch),
                    ));
                    tensors.push(tensor(
                        &format!("up_blocks.{block_idx}.resnets.{rn}.conv_shortcut.bias"),
                        vec![*out_ch],
                        vec![0.0; *out_ch],
                    ));
                }
            }
            if *has_up {
                tensors.push(tensor(
                    &format!("up_blocks.{block_idx}.upsamplers.0.conv.weight"),
                    vec![*out_ch, *out_ch, 3, 3],
                    zeros(out_ch * out_ch * 3 * 3),
                ));
                tensors.push(tensor(
                    &format!("up_blocks.{block_idx}.upsamplers.0.conv.bias"),
                    vec![*out_ch],
                    vec![0.0; *out_ch],
                ));
            }
        }
        // conv_norm_out: GroupNorm(32, 128); conv_out: [out=3, in=128, 3, 3]
        tensors.push(tensor("conv_norm_out.weight", vec![128], vec![1.0; 128]));
        tensors.push(tensor("conv_norm_out.bias", vec![128], vec![0.0; 128]));
        tensors.push(tensor(
            "conv_out.weight",
            vec![3, 128, 3, 3],
            zeros(3 * 128 * 3 * 3),
        ));
        tensors.push(tensor("conv_out.bias", vec![3], vec![0.0; 3]));
        tensors
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
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-burn-pipeline-{prefix}-{nonce}"))
}

/// Resolve the burn worker binary; returns `None` (test skips) when it can
/// neither be located nor built.
fn resolve_worker_executable() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("REIMAGINE_BURN_WORKER") {
        let path = PathBuf::from(value);
        if path.is_file() {
            return Some(path);
        }
        eprintln!(
            "burn_full_pipeline: REIMAGINE_BURN_WORKER is set but not a file: {}",
            path.display()
        );
    }

    if let Some(value) = std::env::var_os(format!("CARGO_BIN_EXE_{WORKER_BINARY}")) {
        let path = PathBuf::from(value);
        if path.is_file() {
            return Some(path);
        }
    }

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate is two levels below the workspace root");
    let target_root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target"));
    for profile in ["debug", "release"] {
        let candidate = target_root.join(profile).join(WORKER_BINARY);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    eprintln!(
        "burn_full_pipeline: worker binary not found; building `cargo build -p reimagine-inference-burn-worker` ..."
    );
    let built = Command::new("cargo")
        .arg("build")
        .arg("-p")
        .arg("reimagine-inference-burn-worker")
        .arg("--quiet")
        .current_dir(workspace_root)
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    let debug_binary = target_root.join("debug").join(WORKER_BINARY);
    if built && debug_binary.is_file() {
        return Some(debug_binary);
    }
    eprintln!(
        "burn_full_pipeline: skipping; could not resolve/build {WORKER_BINARY} (CARGO_TARGET_DIR={})",
        target_root.display()
    );
    None
}

/// Helper to build a `ModelService` for tests with a mock `ModelAcquisitionService`.
fn test_model_service(paths: AppPaths) -> ModelService {
    let config = reimagine_config::AppConfig::new(paths.clone());
    let acquisition_service = Arc::new(reimagine_app_host::ModelAcquisitionService::new(
        paths.clone(),
        &config,
    ));
    ModelService::new(paths, acquisition_service)
}

fn worker_launch_spec(paths: &AppPaths, executable: PathBuf) -> WorkerLaunchSpec {
    let instance = BackendInstanceId::from(INSTANCE_LABEL);
    WorkerLaunchSpec {
        executable,
        expected: ExpectedWorkerIdentity {
            backend_instance_id: instance,
            installation_id: WorkerInstallationId::from("dev"),
            backend_kind: "burn".to_owned(),
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            manifest_digest: "dev".to_owned(),
        },
        supported_protocols: ProtocolRange::new(2, 2),
        limits: WorkerLimits {
            request_timeout: Duration::from_secs(120),
            ..WorkerLimits::default()
        },
        environment: vec![
            (
                "REIMAGINE_MODELS_DIR".to_owned(),
                paths.models_dir().display().to_string(),
            ),
            (
                "REIMAGINE_OUTPUT_DIR".to_owned(),
                paths.output_dir().display().to_string(),
            ),
            (
                "REIMAGINE_ALLOWED_MODEL_ROOTS".to_owned(),
                paths.models_dir().display().to_string(),
            ),
            (
                "REIMAGINE_ALLOWED_OUTPUT_ROOTS".to_owned(),
                paths.output_dir().display().to_string(),
            ),
        ],
    }
}

/// Writes the model manifest describing the tiny SDXL package with
/// per-component sources (burn native component package contract).
async fn write_tiny_manifest(paths: &AppPaths) {
    let component = |role: ModelRole, name: &str| {
        ModelComponentSource::new(
            role,
            ModelSource::relative(
                ModelRootId::new("base"),
                format!("tiny-sdxl-burn/{name}/model.safetensors"),
            ),
            ModelFormat::SafeTensors,
        )
        .with_metadata("component", name)
        .with_metadata("backend", "burn")
        .with_metadata("converted_layout", "burn_native_component_package")
        .with_metadata("contract", "burn.component")
        .with_metadata("contract_version", "1")
    };
    let descriptor = ModelDescriptor::new(
        ModelId::new(MODEL_ID),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![
            ModelRole::CheckpointBundle,
            ModelRole::DiffusionModel,
            ModelRole::TextEncoder,
            ModelRole::Vae,
        ],
        ModelSource::relative(
            ModelRootId::new("base"),
            "tiny-sdxl-burn/diffusion/model.safetensors",
        ),
        ModelFormat::SafeTensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_components(vec![
        component(ModelRole::DiffusionModel, "diffusion"),
        component(ModelRole::Vae, "vae"),
        component(ModelRole::TextEncoder, "text_encoder"),
        component(ModelRole::TextEncoder, "text_encoder_2"),
    ]);
    test_model_service(paths.clone())
        .save_manifest(
            &ModelManifest::new()
                .with_root(ModelRoot::base_models())
                .with_model(descriptor),
        )
        .await
        .expect("save tiny manifest");
}

/// The smoke SDXL workflow re-parameterized for the tiny 64x64 fixture.
fn tiny_workflow() -> reimagine_core::workflow::Workflow {
    let workflow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sdxl-base-burn-smoke-workflow.json");
    let workflow_raw = std::fs::read_to_string(&workflow_path)
        .expect("sdxl-base-burn-smoke-workflow.json must be readable");
    let mut workflow: Value =
        serde_json::from_str(&workflow_raw).expect("smoke workflow must be valid JSON");
    workflow["id"] = WORKFLOW_ID.into();
    workflow["nodes"][0]["params"]["checkpoint"]["value"]["id"] = MODEL_ID.into();
    workflow["nodes"][5]["params"]["width"]["value"] = 64.into();
    workflow["nodes"][5]["params"]["height"]["value"] = 64.into();
    serde_json::from_value(workflow).expect("workflow JSON to Workflow domain model")
}

#[tokio::test]
// The wgpu worker needs a GPU-backed adapter; CI's headless Ubuntu
// runners have none (macOS job runs this for real). Runs locally and
// on macos; ignored on Linux CI. Override with `cargo test -- --ignored`.
#[cfg_attr(target_os = "linux", ignore = "requires GPU-backed wgpu worker")]
async fn burn_full_pipeline_load_encode_latent_sample_decode_saves_png() {
    let executable = match resolve_worker_executable() {
        Some(path) => path,
        None => return,
    };
    let started_at = Instant::now();

    // Fresh workspace with synthetic tiny SDXL weights + manifest.
    let workspace_root = unique_temp_dir("burn-full-pipeline");
    let paths = AppPaths::new(&workspace_root);
    paths.ensure_all().await.expect("workspace dirs");
    tiny_fixture::write_package(paths.models_dir());
    write_tiny_manifest(&paths).await;

    let backend_config = InferenceBackendConfig {
        schema_version: "1".to_owned(),
        backend: InferenceBackendKind::Burn,
        selected_instance: Some(INSTANCE_LABEL.to_owned()),
        ..InferenceBackendConfig::default()
    };

    // Boot the process-backed workspace host (real burn worker subprocess).
    let event_bus = EventBus::new();
    let collector = TestRunEventCollector::new(&event_bus);
    let host = Arc::new(
        WorkspaceHost::try_with_backend_config_and_worker_inventory(
            WorkspaceScope::new("ws-burn-full-pipeline"),
            &workspace_root,
            backend_config,
            event_bus,
            Arc::new(StaticWorkerInventoryProvider::new(
                WorkerInventorySnapshot::new(vec![
                    WorkerBackendCandidate::try_new(
                        worker_launch_spec(&paths, executable),
                        WorkerInstanceProfile {
                            backend_instance_id: BackendInstanceId::from(INSTANCE_LABEL),
                            device_label: "wgpu:default".to_owned(),
                            capabilities: vec![
                                "model.load_bundle",
                                "latent.create_empty",
                                "text.encode",
                                "diffusion.sample",
                                "latent.decode",
                                "latent.encode",
                                "image.import",
                                "image.save",
                                "image.preview",
                            ]
                            .into_iter()
                            .map(str::to_owned)
                            .collect(),
                            operation_options: serde_json::json!({}),
                        },
                    )
                    .expect("worker candidate"),
                ]),
            )),
        )
        .await
        .expect("process-backed workspace host"),
    );

    // 1. Open / register the tiny workflow.
    let workflow = tiny_workflow();
    let workflow_id = host.workflow_service().register_workflow(workflow);
    assert_eq!(workflow_id.as_str(), WORKFLOW_ID);

    // 2. Run the full pipeline up to save_image directly via app-host.
    let request = RunWorkflowRequest::new(
        workflow_id,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::Node {
            node_id: NodeId::new("node_save_image"),
        }]),
    )
    .with_correlation_id(reimagine_core::diagnostic::CorrelationId::new(
        "burn-full-pipeline",
    ));

    let result = host
        .run_workflow(request)
        .await
        .expect("start workflow run");
    let RunWorkflowResult::Started { handle, .. } = result else {
        panic!("expected Started outcome, got {result:?}");
    };
    let run_id = handle.run_id().clone();

    // 3. Poll for a terminal state.
    let deadline = Instant::now() + Duration::from_secs(300);
    let summary = loop {
        if let Some(summary) = host.run_summary(&run_id)
            && summary.state.is_terminal()
        {
            break summary;
        }
        assert!(
            Instant::now() < deadline,
            "run {run_id} did not reach a terminal state within 300s"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    };

    // 4. The full pipeline must complete with a save_image PNG artifact.
    assert_eq!(
        summary.state,
        TaskState::Completed,
        "full pipeline run must complete; summary diagnostics: {:?}",
        summary.diagnostics
    );

    assert!(
        !summary.artifacts.is_empty(),
        "completed summary must include artifacts"
    );
    let artifact_meta = &summary.artifacts[0];
    assert_eq!(
        artifact_meta.node_id.as_str(),
        "node_save_image",
        "final artifact must be the node_save_image output"
    );

    // 5. Events must show run completion and artifact creation.
    let recorded_events = collector.events();
    assert!(
        recorded_events
            .iter()
            .any(|e| e.kind() == RunEventKind::RunCompleted),
        "events must include RunCompleted evidence"
    );
    assert!(
        recorded_events
            .iter()
            .any(|e| e.kind() == RunEventKind::ArtifactCreated),
        "events must include ArtifactCreated evidence"
    );

    // 6. Access the artifact through WorkspaceHost and verify PNG format + dimensions.
    let access = host
        .resolve_artifact(&artifact_meta.id)
        .expect("artifact must resolve");
    assert_eq!(
        access.media_type, "image/png",
        "artifact media type must be image/png"
    );
    let artifact_bytes = tokio::fs::read(&access.path)
        .await
        .expect("read artifact bytes");
    assert!(
        artifact_bytes.starts_with(PNG_SIGNATURE),
        "artifact must be a valid PNG (signature mismatch)"
    );
    let image = image::load_from_memory(&artifact_bytes).expect("PNG must decode");
    assert_eq!(
        (image.width(), image.height()),
        (64, 64),
        "artifact must be 64x64"
    );

    eprintln!(
        "burn_full_pipeline completed: model={MODEL_ID}, instance={INSTANCE_LABEL}, run_id={run_id}, artifact_id={}, duration={}s",
        artifact_meta.id,
        started_at.elapsed().as_secs(),
    );
}
