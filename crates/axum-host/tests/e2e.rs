//! End-to-end HTTP tests for the V1 Axum host surface.
//!
//! These tests build a real `WorkspaceHost` with a mock node executor
//! and a real runtime service, then drive the HTTP router via
//! `tower::ServiceExt::oneshot`. The test runtime is wired with a
//! `RunEventRecorder` so `GET /runs/:id/events` returns real events
//! emitted during the run.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::header;
use axum::http::{Request, StatusCode};
use reimagine_agent::WorkspaceScope;
use reimagine_app_host::{BackendSelection, ModelService, WorkspaceHost};
use reimagine_config::{AppConfig, AppPaths};
use reimagine_core::model::{
    ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, ParamValue, WorkflowId,
    WorkflowVersion,
};
use reimagine_core::workflow::{Workflow, WorkflowNode};
use reimagine_model_manager::{
    ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelSource, ModelSourceStatus,
};
use reimagine_nodes::{BUILTIN_CHECKPOINT_LOADER, BUILTIN_KSAMPLER, BuiltinNodeCatalog};
use reimagine_runtime::RunEventSink;
use reimagine_runtime::{
    BoxedNodeExecutor, ExecutionOutput, NodeExecutionContext, NodeExecutor, NodeExecutorRegistry,
    RuntimeService,
};
use serde_json::Value;
use tower::ServiceExt;

use reimagine_axum_host::{AxumHostState, RunEventRecorder, build_router};

const WORKFLOW_ID: &str = "wf-axum-test";
const MODEL_ID: &str = "sdxl-base-1.0";
const CHECKPOINT_FILENAME: &str = "sdxl-base-1.0.safetensors";

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-axum-e2e-{prefix}-{nonce}"))
}

fn builtin_catalog() -> Arc<BuiltinNodeCatalog> {
    Arc::new(BuiltinNodeCatalog::v1())
}

fn mock_loader_executor() -> BoxedNodeExecutor {
    Arc::new(MockLoaderExecutor)
}

struct MockLoaderExecutor;

#[async_trait]
impl NodeExecutor for MockLoaderExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, reimagine_runtime::NodeExecutorError> {
        Ok(Vec::new())
    }
}

fn model_ref_for(model_id: &ModelId) -> ModelRef {
    ModelRef::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    )
}

fn build_workflow(model_id: &ModelId) -> Workflow {
    Workflow::new(WORKFLOW_ID, WorkflowVersion::new(1)).with_node(
        WorkflowNode::new("loader", BUILTIN_CHECKPOINT_LOADER)
            .with_param("checkpoint", ParamValue::ModelRef(model_ref_for(model_id))),
    )
}

fn manifest_with_model(model_id: &ModelId, filename: &str) -> ModelManifest {
    let descriptor = ModelDescriptor::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(ModelRoot::base_models().id().clone(), filename),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available);
    ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor)
}

fn manifest_with_missing_model() -> ModelManifest {
    ModelManifest::new().with_root(ModelRoot::base_models())
}

/// Build a host with a registered mock executor, a populated model
/// manifest, and a fresh event recorder. Tests that need a different
/// manifest or runtime can call the lower-level helpers below.
async fn build_ready_host(
    manifest: ModelManifest,
    base: &str,
) -> (
    Arc<WorkspaceHost>,
    Arc<RuntimeService>,
    Arc<RunEventRecorder>,
) {
    let paths = AppPaths::new(unique_temp_dir(base));
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::write(paths.models_dir().join(CHECKPOINT_FILENAME), b"placeholder")
        .await
        .unwrap();
    let model_service = ModelService::new(paths.clone());
    model_service
        .save_manifest(&manifest)
        .await
        .expect("save manifest");

    let mut registry = NodeExecutorRegistry::default();
    registry
        .register(BUILTIN_CHECKPOINT_LOADER, mock_loader_executor())
        .expect("register mock loader");

    let recorder = Arc::new(RunEventRecorder::new());
    let runtime = Arc::new(RuntimeService::new(
        registry,
        Arc::new(reimagine_runtime::NoopBackendInstanceRuntimeHooks::default()),
        recorder.clone() as Arc<dyn RunEventSink>,
        Arc::new(reimagine_runtime::SystemClock),
    ));

    let host = Arc::new(WorkspaceHost::new(
        WorkspaceScope::new(format!("ws-{base}")),
        AppConfig::new(paths),
        reimagine_config::InferenceBackendConfig::default(),
        runtime.clone(),
        builtin_catalog(),
    ));
    (host, runtime, recorder)
}

/// Build a host that uses the real Candle inference backend and the
/// built-in executor registry, with a populated model manifest.
async fn build_candle_ready_host(
    manifest: ModelManifest,
    base: &str,
) -> (
    Arc<WorkspaceHost>,
    Arc<RunEventRecorder>,
    std::path::PathBuf,
) {
    let base_path = unique_temp_dir(base);
    let paths = AppPaths::new(&base_path);
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::write(paths.models_dir().join(CHECKPOINT_FILENAME), b"placeholder")
        .await
        .unwrap();
    let model_service = ModelService::new(paths.clone());
    model_service
        .save_manifest(&manifest)
        .await
        .expect("save manifest");

    let recorder = Arc::new(RunEventRecorder::new());
    let host = WorkspaceHost::with_defaults_and_backend(
        WorkspaceScope::new(format!("ws-{base}")),
        &base_path,
        BackendSelection::Candle,
        recorder.clone() as Arc<dyn RunEventSink>,
    );
    (Arc::new(host), recorder, base_path)
}

fn load_sdxl_workflow_json() -> serde_json::Value {
    let path = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/architecture/examples/sdxl-base-workflow.json"
    ));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("failed to parse SDXL workflow: {e}"))
}

fn build_state(host: Arc<WorkspaceHost>, recorder: Arc<RunEventRecorder>) -> AxumHostState {
    AxumHostState::new(host, recorder)
}

async fn body_bytes(body: Body) -> Vec<u8> {
    use http_body_util::BodyExt;
    body.collect().await.unwrap().to_bytes().to_vec()
}

async fn run_to_completion(runtime: &RuntimeService, run_id: &reimagine_core::model::RunId) {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if let Some(summary) = runtime.summary(run_id) {
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

fn json_request(method: &str, uri: &str, body: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(json) => {
            builder = builder.header(header::CONTENT_TYPE, "application/json");
            Body::from(json.to_string())
        }
        None => Body::empty(),
    };
    builder.body(body).expect("build request")
}

#[tokio::test]
async fn health_returns_workspace_scope() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "health").await;
    let app = build_router().with_state(build_state(host, recorder));

    let response = app
        .oneshot(json_request("GET", "/health", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["status"], "ok");
    assert!(json["workspace"].as_str().unwrap().starts_with("ws-"));
}

#[tokio::test]
async fn nodes_route_returns_workspace_catalog_projection() {
    let (host, _runtime, recorder) = build_ready_host(manifest_with_missing_model(), "nodes").await;
    let expected_len = host.list_node_defs().len();
    let app = build_router().with_state(build_state(host, recorder));

    let response = app
        .oneshot(json_request("GET", "/nodes", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    let nodes = json["nodes"].as_array().expect("nodes array");
    assert_eq!(nodes.len(), expected_len);

    let ksampler = nodes
        .iter()
        .find(|node| node["type"] == BUILTIN_KSAMPLER)
        .expect("builtin.ksampler is projected from app-host catalog");
    assert_eq!(ksampler["displayName"], "KSampler");
    assert!(
        ksampler["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|slot| slot["id"] == "model")
    );
    assert!(
        ksampler["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .any(|param| param["id"] == "seed")
    );
}

#[tokio::test]
async fn open_workflow_with_inline_body_registers_workflow() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "open-inline").await;
    let app = build_router().with_state(build_state(host.clone(), recorder));

    let workflow_json = serde_json::to_value(build_workflow(&ModelId::new(MODEL_ID))).unwrap();
    let body = serde_json::to_string(&serde_json::json!({ "workflow": workflow_json })).unwrap();
    let response = app
        .oneshot(json_request("POST", "/workflows/open", Some(&body)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["workflow_id"], WORKFLOW_ID);
    assert_eq!(json["source"], "inline");
    assert!(
        host.workflow_service()
            .contains(&WorkflowId::new(WORKFLOW_ID))
    );
}

#[tokio::test]
async fn open_workflow_twice_returns_existing() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "open-existing").await;
    host.workflow_service()
        .register_workflow(build_workflow(&ModelId::new(MODEL_ID)));
    let app = build_router().with_state(build_state(host, recorder));

    let body = serde_json::to_string(&serde_json::json!({ "id": WORKFLOW_ID })).unwrap();
    let response = app
        .oneshot(json_request("POST", "/workflows/open", Some(&body)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["source"], "existing");
}

#[tokio::test]
async fn open_workflow_with_neither_id_nor_workflow_returns_400() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "open-bad").await;
    let app = build_router().with_state(build_state(host, recorder));

    let response = app
        .oneshot(json_request("POST", "/workflows/open", Some("{}")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "bad_request");
}

#[tokio::test]
async fn open_workflow_with_both_id_and_workflow_returns_400() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "open-both").await;
    let app = build_router().with_state(build_state(host, recorder));
    let workflow_json = serde_json::to_value(build_workflow(&ModelId::new(MODEL_ID))).unwrap();
    let body = serde_json::to_string(&serde_json::json!({
        "id": WORKFLOW_ID,
        "workflow": workflow_json,
    }))
    .unwrap();
    let response = app
        .oneshot(json_request("POST", "/workflows/open", Some(&body)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "bad_request");
}

#[tokio::test]
async fn open_workflow_with_id_round_trips_through_disk() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "open-disk").await;
    let workflow = build_workflow(&ModelId::new(MODEL_ID));
    let workflow_json = serde_json::to_value(&workflow).unwrap();

    // Drop the file directly into the host's `workflows_dir` so the
    // open route must read it from disk. We intentionally avoid
    // registering the workflow in memory first.
    let workflows_dir = host.workflow_service().workflows_dir().to_path_buf();
    tokio::fs::create_dir_all(&workflows_dir).await.unwrap();
    let target = workflows_dir.join(format!("{WORKFLOW_ID}.json"));
    tokio::fs::write(&target, serde_json::to_vec_pretty(&workflow_json).unwrap())
        .await
        .unwrap();

    let app = build_router().with_state(build_state(host, recorder));
    let body = serde_json::to_string(&serde_json::json!({ "id": WORKFLOW_ID })).unwrap();
    let response = app
        .oneshot(json_request("POST", "/workflows/open", Some(&body)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["workflow_id"], WORKFLOW_ID);
    assert_eq!(json["source"], "disk");
}

#[tokio::test]
async fn run_unknown_workflow_returns_404() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "run-unknown").await;
    let app = build_router().with_state(build_state(host, recorder));

    let response = app
        .oneshot(json_request("POST", "/workflows/missing-wf/run", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "unknown_workflow");
}

#[tokio::test]
async fn run_blocked_when_model_missing() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "run-blocked").await;
    let workflow_id = host
        .workflow_service()
        .register_workflow(build_workflow(&ModelId::new(MODEL_ID)));
    let app = build_router().with_state(build_state(host, recorder));

    let body = serde_json::to_string(&serde_json::json!({
        "target_selection": {
            "kind": "explicit",
            "targets": [
                { "kind": "node_output", "node_id": "loader", "slot_id": "model" },
                { "kind": "node_output", "node_id": "loader", "slot_id": "clip" },
                { "kind": "node_output", "node_id": "loader", "slot_id": "vae" }
            ]
        }
    }))
    .unwrap();
    let response = app
        .oneshot(json_request(
            "POST",
            &format!("/workflows/{workflow_id}/run"),
            Some(&body),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["outcome"], "blocked");
    let diagnostics = json["diagnostics"].as_array().expect("diagnostics array");
    assert!(
        diagnostics
            .iter()
            .any(|d| d["code"].as_str().unwrap().contains("MODEL_REF_NOT_FOUND"))
    );
}

#[tokio::test]
async fn run_happy_path_completes_and_serves_snapshot_and_events() {
    let model_id = ModelId::new(MODEL_ID);
    let (host, runtime, recorder) = build_ready_host(
        manifest_with_model(&model_id, CHECKPOINT_FILENAME),
        "run-happy",
    )
    .await;
    let workflow_id = host
        .workflow_service()
        .register_workflow(build_workflow(&model_id));
    let app = build_router().with_state(build_state(host.clone(), recorder.clone()));

    let body = serde_json::to_string(&serde_json::json!({
        "target_selection": {
            "kind": "explicit",
            "targets": [
                { "kind": "node_output", "node_id": "loader", "slot_id": "model" },
                { "kind": "node_output", "node_id": "loader", "slot_id": "clip" },
                { "kind": "node_output", "node_id": "loader", "slot_id": "vae" }
            ]
        },
        "correlation_id": "corr-axum-1"
    }))
    .unwrap();
    let response = app
        .clone()
        .oneshot(json_request(
            "POST",
            &format!("/workflows/{workflow_id}/run"),
            Some(&body),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["outcome"], "started");
    let run_id = json["run_id"].as_str().expect("run_id").to_string();
    let initial_state = json["initial_snapshot"]["state"].as_str().unwrap();
    assert!(
        initial_state == "Queued" || initial_state == "Running",
        "initial state should be Queued or Running, got {initial_state}"
    );

    let parsed_run_id = reimagine_core::model::RunId::new(run_id.clone());
    run_to_completion(&runtime, &parsed_run_id).await;

    // GET /runs/:id — should return a summary now that the run is terminal.
    let response = app
        .clone()
        .oneshot(json_request("GET", &format!("/runs/{run_id}"), None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["kind"], "summary");
    assert_eq!(json["state"], "Completed");
    assert_eq!(json["workflow_id"].as_str().unwrap(), workflow_id.as_str());

    // GET /runs/:id/events — should record the lifecycle events.
    let response = app
        .clone()
        .oneshot(json_request("GET", &format!("/runs/{run_id}/events"), None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    let events = json["events"].as_array().expect("events array");
    let kinds: Vec<&str> = events
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or(""))
        .collect();
    assert!(kinds.iter().any(|k| *k == "RunQueued"), "kinds = {kinds:?}");
    assert!(
        kinds.iter().any(|k| *k == "RunStarted"),
        "kinds = {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| *k == "RunCompleted"),
        "kinds = {kinds:?}"
    );
    assert!(
        events.iter().any(|e| e["correlation_id"]
            .as_str()
            .unwrap_or("")
            .contains("corr-axum-1")),
        "expected at least one event to carry the correlation id, got {events:?}"
    );
}

#[tokio::test]
async fn get_run_unknown_returns_404() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "runs-unknown").await;
    let app = build_router().with_state(build_state(host, recorder));

    let response = app
        .oneshot(json_request("GET", "/runs/does-not-exist", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "unknown_run");
}

#[tokio::test]
async fn get_run_events_for_unknown_run_returns_empty_list() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "runs-events-empty").await;
    let app = build_router().with_state(build_state(host, recorder));

    let response = app
        .oneshot(json_request("GET", "/runs/never-started/events", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["run_id"], "never-started");
    assert_eq!(json["events"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn run_handoff_uses_explicit_target_selection() {
    let model_id = ModelId::new(MODEL_ID);
    let (host, runtime, recorder) = build_ready_host(
        manifest_with_model(&model_id, CHECKPOINT_FILENAME),
        "run-targets",
    )
    .await;
    let workflow_id = host
        .workflow_service()
        .register_workflow(build_workflow(&model_id));
    let app = build_router().with_state(build_state(host, recorder));

    let body = serde_json::json!({
        "target_selection": {
            "kind": "explicit",
            "targets": [
                { "kind": "node_output", "node_id": "loader", "slot_id": "model" },
                { "kind": "node_output", "node_id": "loader", "slot_id": "clip" },
                { "kind": "node_output", "node_id": "loader", "slot_id": "vae" }
            ]
        }
    });
    let response = app
        .oneshot(json_request(
            "POST",
            &format!("/workflows/{workflow_id}/run"),
            Some(&body.to_string()),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["outcome"], "started");
    let run_id = reimagine_core::model::RunId::new(json["run_id"].as_str().unwrap().to_string());
    run_to_completion(&runtime, &run_id).await;
}

#[tokio::test]
async fn candle_sdxl_workflow_run_completes_with_image_artifact() {
    let model_id = ModelId::new(MODEL_ID);
    let (host, recorder, base_path) = build_candle_ready_host(
        manifest_with_model(&model_id, CHECKPOINT_FILENAME),
        "candle-sdxl",
    )
    .await;
    let app = build_router().with_state(build_state(host.clone(), recorder.clone()));

    // Open the canonical SDXL workflow inline.
    let workflow_json = load_sdxl_workflow_json();
    let workflow_id = workflow_json["id"].as_str().expect("workflow id");
    let open_body = serde_json::json!({ "workflow": workflow_json }).to_string();
    let response = app
        .clone()
        .oneshot(json_request("POST", "/workflows/open", Some(&open_body)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Run with an explicit target that forces the full pipeline up to
    // save_image. After diffusion.sample lands, the next heavy
    // unimplemented kernel is latent.decode.
    let run_body = serde_json::json!({
        "target_selection": {
            "kind": "explicit",
            "targets": [
                { "kind": "node", "node_id": "node_save_image" }
            ]
        },
        "correlation_id": "corr-candle-sdxl"
    })
    .to_string();
    let response = app
        .clone()
        .oneshot(json_request(
            "POST",
            &format!("/workflows/{workflow_id}/run"),
            Some(&run_body),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["outcome"], "started");
    let run_id_str = json["run_id"].as_str().expect("run_id");
    let run_id = reimagine_core::model::RunId::new(run_id_str.to_string());

    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(summary) = host.runtime_service().summary(&run_id) {
            assert!(
                summary.state.is_terminal(),
                "run {run_id} should be terminal"
            );
            assert_eq!(
                summary.state,
                reimagine_runtime::RunState::Completed,
                "expected run to complete successfully"
            );
            break;
        }
        if std::time::Instant::now() > deadline {
            panic!("run {run_id} did not finish in time");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // GET /runs/:id should return a summary with Completed state.
    let response = app
        .clone()
        .oneshot(json_request("GET", &format!("/runs/{run_id_str}"), None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["kind"], "summary");
    assert_eq!(json["state"], "Completed");
    let artifacts = json["artifacts"]
        .as_array()
        .expect("summary artifacts should be an array");
    assert_eq!(artifacts.len(), 1, "summary should expose one artifact");
    let reference = artifacts[0]["reference"]
        .as_str()
        .expect("artifact should expose host-neutral reference");
    assert!(
        reference.ends_with(".png"),
        "artifact reference should point at the generated PNG, got {reference}"
    );

    // GET /runs/:id/events should include lifecycle events and no RunFailed.
    let response = app
        .oneshot(json_request(
            "GET",
            &format!("/runs/{run_id_str}/events"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    let events = json["events"].as_array().expect("events array");
    let kinds: Vec<&str> = events
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or(""))
        .collect();
    assert!(kinds.iter().any(|k| *k == "RunQueued"), "kinds = {kinds:?}");
    assert!(
        kinds.iter().any(|k| *k == "RunStarted"),
        "kinds = {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| *k == "RunCompleted"),
        "kinds = {kinds:?}"
    );
    assert!(
        !kinds.iter().any(|k| *k == "RunFailed"),
        "expected no RunFailed event, got {kinds:?}"
    );
    assert!(
        events.iter().any(|e| e["correlation_id"]
            .as_str()
            .unwrap_or("")
            .contains("corr-candle-sdxl")),
        "expected an event to carry the correlation id, got {events:?}"
    );

    // Verify a PNG file was written to the workspace output dir.
    let paths = AppPaths::new(&base_path);
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

#[tokio::test]
async fn artifact_route_serves_png_bytes() {
    let model_id = ModelId::new(MODEL_ID);
    let (host, recorder, base_path) = build_candle_ready_host(
        manifest_with_model(&model_id, CHECKPOINT_FILENAME),
        "candle-sdxl-artifact",
    )
    .await;
    let app = build_router().with_state(build_state(host.clone(), recorder.clone()));

    // Open the canonical SDXL workflow inline.
    let workflow_json = load_sdxl_workflow_json();
    let workflow_id = workflow_json["id"].as_str().expect("workflow id");
    let open_body = serde_json::json!({ "workflow": workflow_json }).to_string();
    let response = app
        .clone()
        .oneshot(json_request("POST", "/workflows/open", Some(&open_body)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Run with an explicit target that forces the full pipeline up to save_image.
    let run_body = serde_json::json!({
        "target_selection": {
            "kind": "explicit",
            "targets": [
                { "kind": "node", "node_id": "node_save_image" }
            ]
        },
        "correlation_id": "corr-candle-sdxl-artifact"
    })
    .to_string();
    let response = app
        .clone()
        .oneshot(json_request(
            "POST",
            &format!("/workflows/{workflow_id}/run"),
            Some(&run_body),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["outcome"], "started");
    let run_id_str = json["run_id"].as_str().expect("run_id");
    let run_id = reimagine_core::model::RunId::new(run_id_str.to_string());

    // Wait for run to complete
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(summary) = host.runtime_service().summary(&run_id) {
            assert!(
                summary.state.is_terminal(),
                "run {run_id} should be terminal"
            );
            assert_eq!(
                summary.state,
                reimagine_runtime::RunState::Completed,
                "expected run to complete successfully"
            );
            break;
        }
        if std::time::Instant::now() > deadline {
            panic!("run {run_id} did not finish in time");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // GET /runs/:id to get the artifact id
    let response = app
        .clone()
        .oneshot(json_request("GET", &format!("/runs/{run_id_str}"), None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    let artifacts = json["artifacts"]
        .as_array()
        .expect("summary artifacts should be an array");
    assert_eq!(artifacts.len(), 1, "summary should expose one artifact");
    let artifact_id = artifacts[0]["id"]
        .as_str()
        .expect("artifact should have an id");

    // GET /artifacts/:artifact_id should return the PNG bytes
    let response = app
        .clone()
        .oneshot(json_request(
            "GET",
            &format!("/artifacts/{artifact_id}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("content-type").unwrap(), "image/png");

    let bytes = body_bytes(response.into_body()).await;
    assert!(
        bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
        "PNG response should have PNG signature"
    );
    assert!(bytes.len() > 100, "PNG response should be non-trivial size");
}

#[tokio::test]
async fn artifact_route_returns_404_for_unknown_id() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "artifact-404").await;
    let app = build_router().with_state(build_state(host, recorder));

    let response = app
        .oneshot(json_request(
            "GET",
            "/artifacts/nonexistent-artifact-id",
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "unknown_artifact");
}

#[tokio::test]
async fn artifact_route_returns_404_for_path_traversal_id() {
    let (host, _runtime, recorder) =
        build_ready_host(manifest_with_missing_model(), "artifact-traversal").await;
    let app = build_router().with_state(build_state(host, recorder));

    let response = app
        .oneshot(json_request(
            "GET",
            "/artifacts/..%2F..%2F..%2Fetc%2Fpasswd",
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = body_bytes(response.into_body()).await;
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    // Should be either unknown_artifact (not found in store) or
    // unsafe_artifact_reference (path safety rejected)
    let code = json["error"]["code"].as_str().unwrap();
    assert!(
        code == "unknown_artifact" || code == "unsafe_artifact_reference",
        "expected artifact error code, got {code}"
    );
}
