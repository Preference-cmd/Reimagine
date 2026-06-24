//! App-host smoke test for the Candle-backed workspace.
//!
//! This test builds an SDXL-shaped workflow, registers a matching
//! `sdxl-base-1.0` manifest entry, and runs the workflow through a
//! real `WorkspaceHost` constructed with the Candle backend.
//!
//! This placeholder-model smoke path should fail precisely at real SDXL
//! text encoder loading; committed tests do not carry CLIP weights.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use reimagine_agent::WorkspaceScope;
use reimagine_app_host::{
    BackendSelection, ModelService, RunWorkflowRequest, RunWorkflowResult, WorkspaceHost,
};
use reimagine_config::AppPaths;
use reimagine_core::model::{
    ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, WorkflowVersion,
};
use reimagine_core::readiness::{RunTarget, RunTargetSelection};
use reimagine_core::workflow::{Endpoint, Workflow, WorkflowEdge, WorkflowNode};
use reimagine_model_manager::{
    ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelSource, ModelSourceStatus,
};
use reimagine_runtime::{RunState, VecRunEventSink};

const MODEL_ID: &str = "sdxl-base-1.0";
const CHECKPOINT_FILENAME: &str = "sdxl-base-1.0.safetensors";

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-candle-smoke-{prefix}-{nonce}"))
}

fn manifest_with_model() -> ModelManifest {
    let descriptor = ModelDescriptor::new(
        ModelId::new(MODEL_ID),
        reimagine_core::model::ModelSeries::new("stable_diffusion"),
        reimagine_core::model::ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(ModelRoot::base_models().id().clone(), CHECKPOINT_FILENAME),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available);
    ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor)
}

fn model_ref() -> ModelRef {
    ModelRef::new(
        ModelId::new(MODEL_ID),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    )
}

fn edge(id: &str, from_node: &str, from_slot: &str, to_node: &str, to_slot: &str) -> WorkflowEdge {
    WorkflowEdge::new(
        id,
        Endpoint::node_slot(NodeId::new(from_node), from_slot.into()),
        Endpoint::node_slot(NodeId::new(to_node), to_slot.into()),
    )
}

fn build_sdxl_workflow() -> Workflow {
    Workflow::new("workflow_sdxl_base_demo", WorkflowVersion::new(1))
        .with_node(
            WorkflowNode::new("node_checkpoint", "builtin.checkpoint_loader")
                .with_label("Checkpoint")
                .with_param("checkpoint", ParamValue::ModelRef(model_ref())),
        )
        .with_node(
            WorkflowNode::new("node_positive_prompt", "builtin.string")
                .with_label("Positive Prompt")
                .with_param(
                    "value",
                    ParamValue::String("cinematic lake at sunrise, detailed, soft light".into()),
                ),
        )
        .with_node(
            WorkflowNode::new("node_negative_prompt", "builtin.string")
                .with_label("Negative Prompt")
                .with_param(
                    "value",
                    ParamValue::String("low quality, blurry, distorted".into()),
                ),
        )
        .with_node(
            WorkflowNode::new("node_positive_encode", "builtin.clip_text_encode")
                .with_label("Positive CLIP Encode"),
        )
        .with_node(
            WorkflowNode::new("node_negative_encode", "builtin.clip_text_encode")
                .with_label("Negative CLIP Encode"),
        )
        .with_node(
            WorkflowNode::new("node_latent", "builtin.empty_latent_image")
                .with_label("Empty Latent")
                .with_param("width", ParamValue::Integer(1024))
                .with_param("height", ParamValue::Integer(1024))
                .with_param("batch_size", ParamValue::Integer(1)),
        )
        .with_node(
            WorkflowNode::new("node_sampler", "builtin.ksampler")
                .with_label("KSampler")
                .with_param("seed", ParamValue::Seed(123456789))
                .with_param("steps", ParamValue::Integer(30))
                .with_param("cfg", ParamValue::Float(7.0))
                .with_param("sampler", ParamValue::Select("euler".into()))
                .with_param("scheduler", ParamValue::Select("normal".into()))
                .with_param("denoise", ParamValue::Float(1.0)),
        )
        .with_node(
            WorkflowNode::new("node_vae_decode", "builtin.vae_decode").with_label("VAE Decode"),
        )
        .with_node(
            WorkflowNode::new("node_save_image", "builtin.save_image")
                .with_label("Save Image")
                .with_param("filename_prefix", ParamValue::String("sdxl_demo".into())),
        )
        .with_edge(edge(
            "edge_checkpoint_model_sampler",
            "node_checkpoint",
            "model",
            "node_sampler",
            "model",
        ))
        .with_edge(edge(
            "edge_checkpoint_clip_positive",
            "node_checkpoint",
            "clip",
            "node_positive_encode",
            "clip",
        ))
        .with_edge(edge(
            "edge_checkpoint_clip_negative",
            "node_checkpoint",
            "clip",
            "node_negative_encode",
            "clip",
        ))
        .with_edge(edge(
            "edge_positive_prompt_encode",
            "node_positive_prompt",
            "value",
            "node_positive_encode",
            "text",
        ))
        .with_edge(edge(
            "edge_negative_prompt_encode",
            "node_negative_prompt",
            "value",
            "node_negative_encode",
            "text",
        ))
        .with_edge(edge(
            "edge_positive_conditioning_sampler",
            "node_positive_encode",
            "conditioning",
            "node_sampler",
            "positive",
        ))
        .with_edge(edge(
            "edge_negative_conditioning_sampler",
            "node_negative_encode",
            "conditioning",
            "node_sampler",
            "negative",
        ))
        .with_edge(edge(
            "edge_latent_sampler",
            "node_latent",
            "latent",
            "node_sampler",
            "latent",
        ))
        .with_edge(edge(
            "edge_sampler_vae_decode",
            "node_sampler",
            "latent",
            "node_vae_decode",
            "latent",
        ))
        .with_edge(edge(
            "edge_checkpoint_vae_decode",
            "node_checkpoint",
            "vae",
            "node_vae_decode",
            "vae",
        ))
        .with_edge(edge(
            "edge_vae_decode_save",
            "node_vae_decode",
            "image",
            "node_save_image",
            "image",
        ))
}

async fn run_to_completion(host: &WorkspaceHost, run_id: &reimagine_core::model::RunId) {
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(summary) = host.runtime_service().summary(run_id) {
            assert!(
                summary.state.is_terminal(),
                "run {run_id} should be terminal"
            );
            return;
        }
        if std::time::Instant::now() > deadline {
            panic!("run {run_id} did not finish in time");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn candle_backend_sdxl_placeholder_workflow_reports_missing_text_encoder_weights() {
    let base = unique_temp_dir("app-host");
    let paths = AppPaths::new(&base);
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::write(paths.models_dir().join(CHECKPOINT_FILENAME), b"placeholder")
        .await
        .unwrap();

    let model_service = ModelService::new(paths.clone());
    model_service
        .save_manifest(&manifest_with_model())
        .await
        .expect("save manifest");

    let event_sink: reimagine_runtime::BoxedRunEventSink = Arc::new(VecRunEventSink::new());
    let host = WorkspaceHost::with_defaults_and_backend(
        WorkspaceScope::new("ws-candle-smoke"),
        &base,
        BackendSelection::Candle,
        event_sink,
    );

    let workflow: Workflow = build_sdxl_workflow();
    let workflow_id = host.workflow_service().register_workflow(workflow);

    let request = RunWorkflowRequest::new(
        workflow_id.clone(),
        RunTargetSelection::ExplicitTargets(vec![RunTarget::Node {
            node_id: NodeId::new("node_save_image"),
        }]),
    );

    let result = host
        .run_workflow(request)
        .await
        .expect("run_workflow should hand off to runtime");

    let RunWorkflowResult::Started { handle, .. } = result else {
        panic!("expected run to start, got {result:?}");
    };

    run_to_completion(&host, handle.run_id()).await;

    let summary = host
        .runtime_service()
        .summary(handle.run_id())
        .expect("summary should exist after completion");

    assert_eq!(summary.state, RunState::Failed);
    let diagnostics = summary
        .diagnostics
        .iter()
        .map(|diagnostic| diagnostic.message().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        diagnostics.contains("text encoder weights"),
        "expected missing text encoder weights diagnostic, got: {diagnostics}"
    );
    assert!(
        summary.artifacts.is_empty(),
        "placeholder model should not produce image artifacts"
    );
}
