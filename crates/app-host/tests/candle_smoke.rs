//! App-host smoke test for the Candle-backed workspace.
//!
//! This test loads the canonical SDXL base workflow from
//! `docs/architecture/examples/sdxl-base-workflow.json`, registers a
//! matching `sdxl-base-1.0` manifest entry, and runs the workflow
//! through a real `WorkspaceHost` constructed with the Candle backend.
//!
//! The full pipeline (model.load_bundle → text.encode × 2 →
//! empty_latent_image → diffusion.sample → latent.decode → image.save)
//! completes end-to-end with an image artifact written to the output dir.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use reimagine_agent::WorkspaceScope;
use reimagine_app_host::{
    BackendSelection, ModelService, RunWorkflowRequest, RunWorkflowResult, WorkspaceHost,
};
use reimagine_config::AppPaths;
use reimagine_core::model::{ModelId, ModelRole, NodeId};
use reimagine_core::readiness::{RunTarget, RunTargetSelection};
use reimagine_core::workflow::Workflow;
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

fn load_sdxl_workflow() -> Workflow {
    let path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/architecture/examples/sdxl-base-workflow.json"
    ));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("failed to parse SDXL workflow: {e}"))
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
async fn candle_backend_sdxl_workflow_completes_with_image_artifact() {
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

    let workflow: Workflow = load_sdxl_workflow();
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

    assert_eq!(
        summary.state,
        RunState::Completed,
        "expected run to complete successfully"
    );
    assert!(
        !summary.artifacts.is_empty(),
        "expected at least one artifact in run summary"
    );

    let output_dir = paths.output_dir();
    let mut entries = tokio::fs::read_dir(output_dir)
        .await
        .expect("output dir should exist");
    let png_path = loop {
        let entry = entries
            .next_entry()
            .await
            .expect("output dir entry read")
            .expect("output dir should contain a PNG file");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("png") {
            break path;
        }
    };

    let metadata = tokio::fs::metadata(&png_path)
        .await
        .expect("png file metadata");
    assert!(metadata.len() > 0, "PNG file should be non-empty");

    let bytes = tokio::fs::read(&png_path).await.expect("png file read");
    assert!(
        bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
        "PNG file should have PNG signature"
    );
}
