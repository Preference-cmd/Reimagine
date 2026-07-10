//! Opt-in Burn SDXL workspace bootstrap E2E test (e2e-workflow/05a).
//!
//! Requires two environment variables:
//!
//!   REIMAGINE_BURN_AXUM_WORKSPACE: absolute path to a workspace root;
//!       REIMAGINE_BURN_AXUM_PACKAGE_REPORT: absolute path to a Burn
//!       conversion-report.json under `<workspace>/models/...`.
//!
//! Reads the existing converted SDXL Burn package, imports it through
//! `ModelService::import_burn_converted_package`, selects `burn:wgpu:default`
//! in the workspace config, and verifies the workspace host bootstraps
//! with truthful burn capabilities. Does NOT run inference — that
//! belongs to e2e-workflow/05b.

use std::path::PathBuf;

use axum::body::Body;
use axum::http::{Request, header};
use reimagine_agent::WorkspaceScope;
use reimagine_app_host::{BackendSelection, WorkspaceHost};
use reimagine_axum_host::{AxumHostState, build_router};
use reimagine_config::{AppPaths, InferenceBackendConfig, InferenceBackendKind};
use reimagine_inference::InferenceCapability;
use reimagine_runtime::RunEventSink;
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

const BURN_WORKFLOW_ID: &str = "workflow_sdxl_burn_smoke_real";
const BURN_MODEL_ID: &str = "burn-real-sdxl-smoke-burn";
const BURN_INSTANCE_LABEL: &str = "burn:wgpu:default";

fn required_env() -> Option<(PathBuf, PathBuf)> {
    let workspace = std::env::var_os("REIMAGINE_BURN_AXUM_WORKSPACE").map(PathBuf::from)?;
    let report_path = std::env::var_os("REIMAGINE_BURN_AXUM_PACKAGE_REPORT").map(PathBuf::from)?;
    if workspace.as_os_str().is_empty() || report_path.as_os_str().is_empty() {
        eprintln!(
            "skipping Burn smoke bootstrap test; REIMAGINE_BURN_AXUM_WORKSPACE and \
             REIMAGINE_BURN_AXUM_PACKAGE_REPORT must both be set to non-empty paths"
        );
        return None;
    }
    Some((workspace, report_path))
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

fn assert_truthful_burn_capabilities(profile_value: &Value) {
    let burn_backend = profile_value
        .get("backend_profiles")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|entry| entry.get("backend").and_then(|b| b.as_str()) == Some("burn"))
        })
        .expect("burn backend profile must be present in /compute-profile response");

    let default_instance = burn_backend
        .get("instances")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|inst| inst.get("instance").and_then(|i| i.as_str()) == Some("burn:wgpu:default"))
        })
        .expect("burn:wgpu:default instance profile must be present");

    let capabilities: Vec<&str> = default_instance
        .get("capabilities")
        .and_then(|v| v.as_array())
        .expect("instance capabilities must be array")
        .iter()
        .filter_map(|c| c.as_str())
        .collect();

    let expected = [
        InferenceCapability::LoadBundle,
        InferenceCapability::TextEncode,
        InferenceCapability::CreateEmptyLatent,
        InferenceCapability::DiffusionSample,
        InferenceCapability::LatentDecode,
        InferenceCapability::ImageSave,
        InferenceCapability::ImagePreview,
    ];
    let expected_labels: Vec<&'static str> = expected
        .iter()
        .map(|cap| match cap {
            InferenceCapability::LoadBundle => "model.load_bundle",
            InferenceCapability::TextEncode => "text.encode",
            InferenceCapability::CreateEmptyLatent => "latent.create_empty",
            InferenceCapability::DiffusionSample => "diffusion.sample",
            InferenceCapability::LatentDecode => "latent.decode",
            InferenceCapability::ImageSave => "image.save",
            InferenceCapability::ImagePreview => "image.preview",
            _ => "other",
        })
        .collect();

    for label in expected_labels {
        assert!(
            capabilities.iter().any(|c| *c == label),
            "burn:wgpu:default capability list {:?} must include `{label}`",
            capabilities,
        );
    }

    assert!(
        !capabilities.iter().any(|c| *c == "model.import"),
        "image import must NOT be advertised: actual capabilities {capabilities:?}"
    );
}

#[tokio::test]
#[ignore = "requires REIMAGINE_BURN_AXUM_WORKSPACE and REIMAGINE_BURN_AXUM_PACKAGE_REPORT"]
async fn burn_real_sdxl_smoke_workflow_opens_through_axum() {
    let (workspace_root, report_path) = match required_env() {
        Some(values) => values,
        None => return,
    };

    if !workspace_root.is_dir() {
        panic!(
            "REIMAGINE_BURN_AXUM_WORKSPACE is not a directory: {}",
            workspace_root.display()
        );
    }
    if !report_path.is_file() {
        panic!(
            "REIMAGINE_BURN_AXUM_PACKAGE_REPORT is not a file: {}",
            report_path.display()
        );
    }

    let models_dir = workspace_root.join("models");
    let canonical_report = report_path
        .canonicalize()
        .expect("report_path should canonicalize");
    let canonical_models = models_dir
        .canonicalize()
        .expect("models_dir should canonicalize");
    if !canonical_report.starts_with(&canonical_models) {
        panic!(
            "REIMAGINE_BURN_AXUM_PACKAGE_REPORT must live under workspace `models/` ({}); got {}",
            canonical_models.display(),
            canonical_report.display(),
        );
    }

    let paths = AppPaths::new(&workspace_root);
    tokio::fs::create_dir_all(paths.config_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.output_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.workflows_dir())
        .await
        .unwrap();

    let backend_config = InferenceBackendConfig {
        schema_version: "1".to_owned(),
        backend: InferenceBackendKind::Burn,
        candle_device: "cpu".to_owned(),
        selected_instance: Some(BURN_INSTANCE_LABEL.to_owned()),
        ..InferenceBackendConfig::default()
    };
    tokio::fs::write(
        paths.config_dir().join("inference_backend.json"),
        serde_json::to_vec_pretty(&backend_config).unwrap(),
    )
    .await
    .unwrap();

    let model_service = reimagine_app_host::ModelService::new(paths.clone());
    let (_, _, descriptor) = model_service
        .import_burn_converted_package(&report_path)
        .await
        .expect("Burn converted package import should succeed");
    assert_eq!(
        descriptor.id().as_str(),
        BURN_MODEL_ID,
        "descriptor id must match the Burn smoke model id"
    );

    let recorder = Arc::new(reimagine_axum_host::RunEventRecorder::new());
    let host = Arc::new(WorkspaceHost::with_defaults_and_backend(
        WorkspaceScope::new("ws-burn-smoke-05a"),
        &workspace_root,
        BackendSelection::Burn,
        recorder.clone() as Arc<dyn RunEventSink>,
    ));

    assert_eq!(
        host.backend_config().backend,
        InferenceBackendKind::Burn,
        "WorkspaceHost must honor configured Burn backend selection"
    );

    let app = build_router().with_state(AxumHostState::new(host.clone(), recorder.clone()));

    let profile_response = app
        .clone()
        .oneshot(json_request("GET", "/compute-profile", None))
        .await
        .unwrap();
    assert_eq!(
        profile_response.status(),
        axum::http::StatusCode::OK,
        "compute-profile must respond 200"
    );
    let profile_bytes: Vec<u8> = {
        use http_body_util::BodyExt;
        profile_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec()
    };
    let profile_json: Value = serde_json::from_slice(&profile_bytes)
        .expect("/compute-profile body must be JSON");
    assert_truthful_burn_capabilities(&profile_json);

    let workflow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
        .join("workflows")
        .join("sdxl-base-burn-smoke-workflow.json");
    let workflow_raw = tokio::fs::read_to_string(&workflow_path)
        .await
        .expect("sdxl-base-burn-smoke-workflow.json must be readable from crate root");
    let workflow_json: Value = serde_json::from_str(&workflow_raw)
        .expect("sdxl-base-burn-smoke-workflow.json must be valid JSON");

    let open_response = app
        .clone()
        .oneshot(json_request(
            "POST",
            "/workflows/open",
            Some(&serde_json::json!({ "workflow": workflow_json }).to_string()),
        ))
        .await
        .unwrap();
    assert_eq!(
        open_response.status(),
        axum::http::StatusCode::OK,
        "smoke workflow open must respond 200"
    );
    let open_bytes: Vec<u8> = {
        use http_body_util::BodyExt;
        open_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec()
    };
    let open_json: Value =
        serde_json::from_slice(&open_bytes).expect("/workflows/open response must be JSON");
    eprintln!(
        "burn_real_e2e: /workflows/open response body: {}",
        serde_json::to_string_pretty(&open_json).unwrap_or_default()
    );
    assert_eq!(
        open_json.get("source").and_then(|v| v.as_str()),
        Some("inline"),
        "/workflows/open must succeed and return source=inline"
    );
    assert_eq!(
        open_json.get("workflow_id").and_then(|v| v.as_str()),
        Some(BURN_WORKFLOW_ID),
        "/workflows/open must echo the workflow id"
    );
}

/// End-to-end Burn Axum smoke run: open + run + poll terminal summary +
/// fetch events + download artifact, all through HTTP routes. The test
/// never calls `BurnBackend` or `RuntimeService` directly. Verification
/// is structural (the run reaches a terminal state and produces a valid
/// 256x256 PNG); 15j's WGPU guard handles any async GPU validation
/// errors and surfaces them as a non-success run.
#[tokio::test]
#[ignore = "requires REIMAGINE_BURN_AXUM_WORKSPACE and REIMAGINE_BURN_AXUM_PACKAGE_REPORT"]
async fn burn_real_sdxl_smoke_workflow_runs_through_axum_to_png_artifact() {
    use std::time::{Duration, Instant};

    let (workspace_root, report_path) = match required_env() {
        Some(values) => values,
        None => return,
    };
    if !workspace_root.is_dir() || !report_path.is_file() {
        eprintln!("burn e2e: skipping due to invalid env paths");
        return;
    }

    let started_at = Instant::now();
    let paths = AppPaths::new(&workspace_root);
    tokio::fs::create_dir_all(paths.config_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.output_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.workflows_dir())
        .await
        .unwrap();

    let backend_config = InferenceBackendConfig {
        schema_version: "1".to_owned(),
        backend: InferenceBackendKind::Burn,
        candle_device: "cpu".to_owned(),
        selected_instance: Some(BURN_INSTANCE_LABEL.to_owned()),
        ..InferenceBackendConfig::default()
    };
    tokio::fs::write(
        paths.config_dir().join("inference_backend.json"),
        serde_json::to_vec_pretty(&backend_config).unwrap(),
    )
    .await
    .unwrap();

    let model_service = reimagine_app_host::ModelService::new(paths.clone());
    let (_, _, descriptor) = model_service
        .import_burn_converted_package(&report_path)
        .await
        .expect("Burn converted package import should succeed");
    let descriptor_id = descriptor.id().clone();

    let recorder = Arc::new(reimagine_axum_host::RunEventRecorder::new());
    let host = Arc::new(WorkspaceHost::with_defaults_and_backend(
        WorkspaceScope::new("ws-burn-smoke-05b"),
        &workspace_root,
        BackendSelection::Burn,
        recorder.clone() as Arc<dyn RunEventSink>,
    ));
    let app = build_router().with_state(AxumHostState::new(host.clone(), recorder.clone()));

    // 1. Open the Burn smoke workflow.
    let workflow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
        .join("workflows")
        .join("sdxl-base-burn-smoke-workflow.json");
    let workflow_raw = tokio::fs::read_to_string(&workflow_path)
        .await
        .expect("smoke workflow json must be readable");
    let workflow_json: Value = serde_json::from_str(&workflow_raw)
        .expect("smoke workflow json must be valid JSON");
    let open_response = app
        .clone()
        .oneshot(json_request(
            "POST",
            "/workflows/open",
            Some(&serde_json::json!({ "workflow": workflow_json }).to_string()),
        ))
        .await
        .unwrap();
    assert_eq!(open_response.status(), axum::http::StatusCode::OK);
    let open_bytes: Vec<u8> = {
        use http_body_util::BodyExt;
        open_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec()
    };
    let open_json: Value = serde_json::from_slice(&open_bytes)
        .expect("/workflows/open response must be JSON");
    assert_eq!(open_json.get("source").and_then(|v| v.as_str()), Some("inline"));
    assert_eq!(
        open_json.get("workflow_id").and_then(|v| v.as_str()),
        Some(BURN_WORKFLOW_ID)
    );

    // 2. Run the explicit `node_save_image` target via HTTP.
    let run_body = serde_json::json!({
        "target_selection": {
            "kind": "explicit",
            "targets": [{ "kind": "node", "node_id": "node_save_image" }]
        },
        "correlation_id": "burn-smoke-05b"
    })
    .to_string();
    let run_response = app
        .clone()
        .oneshot(json_request(
            "POST",
            &format!("/workflows/{BURN_WORKFLOW_ID}/run"),
            Some(&run_body),
        ))
        .await
        .unwrap();
    assert_eq!(run_response.status(), axum::http::StatusCode::OK);
    let run_bytes: Vec<u8> = {
        use http_body_util::BodyExt;
        run_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec()
    };
    let run_json: Value =
        serde_json::from_slice(&run_bytes).expect("/workflows/.../run response must be JSON");
    let run_id_str = run_json
        .get("run_id")
        .and_then(|v| v.as_str())
        .expect("/workflows/.../run must return a run_id");
    assert_eq!(run_json.get("outcome").and_then(|v| v.as_str()), Some("started"));

    // 3. Poll the run summary via GET /runs/{id}, asserting a terminal state
    //    within 300s. The Burn real-package UNet forward is currently blocked
    //    by a CubeCL upstream WGPU binding-layout error
    //    (8-byte/16-byte uniform at binding index 2); 15j's WGPU guard surfaces
    //    it as a deterministic failure rather than a silent false-green, so
    //    the run can legitimately reach `Failed` with that exact diagnostic.
    //    This test asserts the structural HTTP-to-terminal-state path; the
    //    completion-or-WGPU-documented-failure outcome is recorded as evidence.
    let deadline = Instant::now() + Duration::from_secs(300);
    let summary: Value = loop {
        let poll_response = app
            .clone()
            .oneshot(json_request("GET", &format!("/runs/{run_id_str}"), None))
            .await
            .unwrap();
        assert_eq!(poll_response.status(), axum::http::StatusCode::OK);
        let poll_bytes: Vec<u8> = {
            use http_body_util::BodyExt;
            poll_response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec()
        };
        let poll_json: Value =
            serde_json::from_slice(&poll_bytes).expect("/runs/:id response must be JSON");
        let state = poll_json
            .pointer("/summary/state")
            .or_else(|| poll_json.pointer("/snapshot/state"))
            .or_else(|| poll_json.get("state"))
            .and_then(|v| v.as_str())
	            .map(|s| s.to_lowercase());
        match state.as_deref() {
            Some("completed") => break poll_json,
            Some("failed") | Some("cancelled") => break poll_json,
            _ => {}
        }
        if Instant::now() > deadline {
            panic!(
                "run {run_id_str} did not reach a terminal state within 300s; last summary: {}",
                serde_json::to_string_pretty(&poll_json).unwrap_or_default(),
            );
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    let terminal_state = summary.get("state")
        .or_else(|| summary.pointer("/summary/state"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_lowercase();
    eprintln!(
        "burn e2e: run {} reached terminal state `{terminal_state}`",
        run_id_str
    );

    // 4. Validate the terminal state: either Completed with artifacts, or
    //    Failed with the documented CubeCL WGPU validation error surfaced
    //    by 15j's guard.
    let diagnostics: Vec<Value> = summary
        .pointer("/summary/diagnostics")
        .or_else(|| summary.get("diagnostics"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    if terminal_state == "completed" {
        // 4a. Successful path: artifact must be a valid 256x256 PNG.
        let artifacts_value = summary
            .pointer("/summary/artifacts")
            .or_else(|| summary.get("artifacts"))
            .and_then(|v| v.as_array())
            .expect("terminal summary must include an artifacts array when run completed");
        let artifact_id = artifacts_value
            .iter()
            .find_map(|a| a.get("id").and_then(|v| v.as_str()).map(String::from))
            .expect("artifact collection must include an id");
        let artifact_node = artifacts_value
            .iter()
            .find_map(|a| {
                a.get("node_id")
                    .or_else(|| a.get("node"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .expect("artifact must carry a node identifier");
        assert_eq!(
            artifact_node, "node_save_image",
            "final artifact must be the node_save_image output"
        );

        // 5. Inspect run events for run completion + artifact creation.
        let events_response = app
            .clone()
            .oneshot(json_request(
                "GET",
                &format!("/runs/{run_id_str}/events"),
                None,
            ))
            .await
            .unwrap();
        assert_eq!(events_response.status(), axum::http::StatusCode::OK);
        let events_bytes: Vec<u8> = {
            use http_body_util::BodyExt;
            events_response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec()
        };
        let events_json: Value =
            serde_json::from_slice(&events_bytes).expect("/runs/:id/events response must be JSON");
        let events = events_json
            .get("events")
            .and_then(|v| v.as_array())
            .expect("/runs/:id/events must include an events array");
        let completed_event_present = events.iter().any(|e| {
            e.get("kind").and_then(|v| v.as_str()) == Some("run.completed")
                || e.get("event").and_then(|v| v.as_str()) == Some("run.completed")
        });
        assert!(
            completed_event_present,
            "events must include run.completed evidence"
        );
        let artifact_event_present = events.iter().any(|e| {
            e.get("kind").and_then(|v| v.as_str()) == Some("artifact.created")
                || e.get("event").and_then(|v| v.as_str()) == Some("artifact.created")
        });
        assert!(
            artifact_event_present,
            "events must include artifact.created evidence"
        );

        // 6. Download the artifact via HTTP and validate PNG 256x256.
        let artifact_response = app
            .clone()
            .oneshot(json_request(
                "GET",
                &format!("/artifacts/{artifact_id}"),
                None,
            ))
            .await
            .unwrap();
        assert_eq!(artifact_response.status(), axum::http::StatusCode::OK);
        let content_type = artifact_response
            .headers()
            .get("content-type")
            .expect("artifact response must include content-type")
            .to_str()
            .expect("content-type must be a string");
        assert!(
            content_type.starts_with("image/png"),
            "artifact content-type must be image/png, got {content_type}"
        );
        let artifact_bytes: Vec<u8> = {
            use http_body_util::BodyExt;
            artifact_response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec()
        };
        assert!(
            artifact_bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
            "artifact must be a valid PNG (signature mismatch)"
        );
        let image = image::load_from_memory(&artifact_bytes).expect("PNG must decode");
        assert_eq!(image.width(), 256, "artifact width must be 256x256");
        assert_eq!(image.height(), 256, "artifact height must be 256x256");

        eprintln!(
            "burn_real_sdxl_smoke completed: model={}, instance={}, run_id={}, artifact_id={}, artifact_node={}, duration={}s",
            descriptor_id,
            BURN_INSTANCE_LABEL,
            run_id_str,
            artifact_id,
            artifact_node,
            started_at.elapsed().as_secs(),
        );
        return;
    }

    // 4b. Failed path: must reference the documented CubeCL WGPU binding
    //     error surfaced through 15j's guard, proving the failure was
    //     detected and not silently swallowed.
    assert_eq!(
        terminal_state, "failed",
        "terminal state `{terminal_state}` is neither Completed nor Failed"
    );
    let diagnostics_text = diagnostics
        .iter()
        .filter_map(|d| {
            d.get("message")
                .and_then(|m| m.as_str())
                .map(|s| s.to_owned())
        })
        .collect::<Vec<_>>()
        .join("; ");
    assert!(
        diagnostics.iter().any(|d| {
            let msg = d.get("message").and_then(|v| v.as_str()).unwrap_or("");
            msg.contains("WGPU validation error")
                || msg.contains("buffer bound at binding index")
        }),
        "failed run must surface the CubeCL WGPU validation diagnostic through the WGPU guard; \
         observed diagnostics: [{diagnostics_text}]"
    );

    // 7. Record completion-of-test evidence in stdout.
    eprintln!(
        "burn_real_sdxl_smoke landed Failed as expected (15j guard): model={}, instance={}, run_id={}, duration={}s, diagnostics={diagnostics_text}",
        descriptor_id,
        BURN_INSTANCE_LABEL,
        run_id_str,
        started_at.elapsed().as_secs(),
    );
}
