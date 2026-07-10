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

fn workspace_status(result: &Value) -> &str {
    result.get("status").and_then(|v| v.as_str()).unwrap_or("")
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
            InferenceCapability::LoadBundle => "load_bundle",
            InferenceCapability::TextEncode => "text_encode",
            InferenceCapability::CreateEmptyLatent => "create_empty_latent",
            InferenceCapability::DiffusionSample => "diffusion_sample",
            InferenceCapability::LatentDecode => "latent_decode",
            InferenceCapability::ImageSave => "image_save",
            InferenceCapability::ImagePreview => "image_preview",
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
        !capabilities.iter().any(|c| *c == "image_import"),
        "image_import must NOT be advertised: actual capabilities {capabilities:?}"
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
    assert_eq!(workspace_status(&open_json), "ok");

    let resolved = open_json
        .get("workflow_id")
        .or_else(|| open_json.get("id"))
        .or_else(|| open_json.get("workflow").and_then(|w| w.get("id")))
        .and_then(|v| v.as_str());
    if let Some(opened_id) = resolved {
        assert_eq!(
            opened_id, BURN_WORKFLOW_ID,
            "smoke workflow id must match the burn smoke id constant"
        );
    }
}
