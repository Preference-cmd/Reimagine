//! Opt-in real-weight E2E test for the SDXL base pipeline.
//!
//! Requires `REIMAGINE_SDXL_REAL_SPLIT_DIR` to point at a directory
//! containing Candle-example-compatible split SDXL weights:
//!
//!   <dir>/unet/model.safetensors
//!   <dir>/text_encoder/model.safetensors
//!   <dir>/text_encoder_2/model.safetensors
//!   <dir>/vae/model.safetensors
//!
//! Set `REIMAGINE_SDXL_REAL_DEVICE` to override the default `cpu`
//! Candle device (e.g., `metal`, `cuda:0`).

mod sdxl_workflows;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use reimagine_agent::WorkspaceScope;
use reimagine_app_host::{BackendSelection, WorkspaceHost};
use reimagine_axum_host::{AxumHostState, build_router};
use reimagine_config::{AppPaths, InferenceBackendConfig, InferenceBackendKind};
use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
use reimagine_model_manager::{
    ModelComponentSource, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelSource,
    ModelSourceStatus,
};
use reimagine_runtime::RunEventSink;
use sdxl_workflows::{SdxlWorkflowOptions, image_to_image, text_to_image};
use serde_json::Value;
use tower::ServiceExt;

const WORKFLOW_ID: &str = "workflow_sdxl_base_demo";
const MODEL_ID: &str = "sdxl-base-1.0";

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-axum-real-e2e-{prefix}-{nonce}"))
}

fn required_split_paths(split_dir: &std::path::Path) -> [PathBuf; 4] {
    [
        split_dir.join("unet/model.safetensors"),
        split_dir.join("text_encoder/model.safetensors"),
        split_dir.join("text_encoder_2/model.safetensors"),
        split_dir.join("vae/model.safetensors"),
    ]
}

fn build_manifest(split_dir: &std::path::Path) -> ModelManifest {
    let [unet, clip_l, clip_g, vae] = required_split_paths(split_dir);

    // The primary source is arbitrary for a split descriptor; using the
    // UNet file keeps manifest validation happy and is never executed as
    // a checkpoint bundle because components are present.
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
        ModelSource::absolute(unet.to_string_lossy()),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_component(
        ModelComponentSource::new(
            ModelRole::DiffusionModel,
            ModelSource::absolute(unet.to_string_lossy()),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "unet"),
    )
    .with_component(
        ModelComponentSource::new(
            ModelRole::TextEncoder,
            ModelSource::absolute(clip_l.to_string_lossy()),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "clip_l"),
    )
    .with_component(
        ModelComponentSource::new(
            ModelRole::TextEncoder,
            ModelSource::absolute(clip_g.to_string_lossy()),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "clip_g"),
    )
    .with_component(
        ModelComponentSource::new(
            ModelRole::Vae,
            ModelSource::absolute(vae.to_string_lossy()),
            ModelFormat::Safetensors,
        )
        .with_metadata("component", "vae"),
    );

    ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor)
}

fn sdxl_workflow_json() -> Value {
    text_to_image(SdxlWorkflowOptions {
        workflow_id: WORKFLOW_ID,
        model_id: MODEL_ID,
        name: "SDXL Base Text to Image (Real E2E)",
        description: "Minimal SDXL base workflow for real-weight E2E validation.",
        filename_prefix: "sdxl_real_e2e",
        denoise: 1.0,
    })
}

fn sdxl_img2img_workflow_json() -> Value {
    image_to_image(
        SdxlWorkflowOptions {
            workflow_id: WORKFLOW_ID,
            model_id: MODEL_ID,
            name: "SDXL Base Image to Image (Real E2E)",
            description: "Minimal SDXL img2img workflow for real-weight E2E validation.",
            filename_prefix: "sdxl_img2img_real_e2e",
            denoise: 0.45,
        },
        "img2img-source.png",
    )
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

async fn body_bytes(body: Body) -> Vec<u8> {
    use http_body_util::BodyExt;
    body.collect().await.unwrap().to_bytes().to_vec()
}

async fn write_img2img_input_png(path: &std::path::Path) {
    tokio::task::spawn_blocking({
        let path = path.to_path_buf();
        move || {
            let width = 512;
            let height = 512;
            let mut image = image::RgbImage::new(width, height);
            for (x, y, pixel) in image.enumerate_pixels_mut() {
                let r = ((x as f32 / width as f32) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
                *pixel = image::Rgb([r, g, b]);
            }
            image
                .save_with_format(&path, image::ImageFormat::Png)
                .expect("write img2img PNG fixture");
        }
    })
    .await
    .expect("join PNG fixture writer");
}

async fn run_inline_workflow_to_png_artifact(
    app: axum::Router,
    host: Arc<WorkspaceHost>,
    base_path: &std::path::Path,
    workflow_json: Value,
    correlation_id: &str,
) {
    let workflow_id = workflow_json["id"]
        .as_str()
        .expect("workflow id")
        .to_string();
    let open_body = serde_json::json!({ "workflow": workflow_json }).to_string();
    let response = app
        .clone()
        .oneshot(json_request("POST", "/workflows/open", Some(&open_body)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let run_body = serde_json::json!({
        "target_selection": {
            "kind": "explicit",
            "targets": [{ "kind": "node", "node_id": "node_save_image" }]
        },
        "correlation_id": correlation_id
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

    let deadline = std::time::Instant::now() + Duration::from_secs(600);
    let artifact_id = loop {
        if let Some(summary) = host.runtime_service().summary(&run_id) {
            assert!(
                summary.state == reimagine_runtime::RunState::Completed,
                "run should complete successfully; state={:?}, diagnostics={:?}",
                summary.state,
                summary.diagnostics
            );
            assert!(
                !summary.artifacts.is_empty(),
                "run should produce at least one artifact"
            );
            let artifact = &summary.artifacts[0];
            assert_eq!(artifact.node_id.as_str(), "node_save_image");
            break artifact.id.clone();
        }
        if std::time::Instant::now() > deadline {
            panic!("run {run_id} did not finish within deadline");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };

    let summary = host
        .runtime_service()
        .summary(&run_id)
        .expect("summary should still exist");
    let artifact_ref = summary
        .artifacts
        .iter()
        .find(|a| a.id == artifact_id)
        .map(|a| a.reference.as_str())
        .expect("artifact reference should exist");
    let artifact_path = base_path.join(artifact_ref);
    assert!(
        artifact_path.starts_with(base_path.join("output")),
        "artifact path should stay under workspace output dir: {}",
        artifact_path.display()
    );

    let metadata = tokio::fs::metadata(&artifact_path)
        .await
        .expect("artifact file should exist");
    assert!(metadata.len() > 0, "artifact file should be non-empty");
    let header = tokio::fs::read(&artifact_path)
        .await
        .expect("read artifact file");
    assert!(
        header.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
        "artifact should be a valid PNG file"
    );

    let response = app
        .clone()
        .oneshot(json_request(
            "GET",
            &format!("/artifacts/{}", artifact_id.as_str()),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("content-type").unwrap(), "image/png");
    let served_bytes = body_bytes(response.into_body()).await;
    assert_eq!(
        served_bytes.len(),
        metadata.len() as usize,
        "served PNG should match file size"
    );
    assert_eq!(served_bytes, header, "served PNG bytes should match file");
}

#[tokio::test]
#[ignore = "requires real SDXL split weights at REIMAGINE_SDXL_REAL_SPLIT_DIR"]
async fn real_sdxl_split_weights_produces_png_artifact() {
    let Some(split_dir) = std::env::var_os("REIMAGINE_SDXL_REAL_SPLIT_DIR").map(PathBuf::from)
    else {
        eprintln!(
            "skipping real E2E test; set REIMAGINE_SDXL_REAL_SPLIT_DIR to a directory with Candle-compatible split SDXL weights"
        );
        return;
    };

    for path in required_split_paths(&split_dir) {
        assert!(
            path.exists(),
            "missing required split weight: {}",
            path.display()
        );
    }

    let base_path = unique_temp_dir("real-e2e");
    let paths = AppPaths::new(&base_path);
    tokio::fs::create_dir_all(paths.config_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.output_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.workflows_dir())
        .await
        .unwrap();

    // inference_backend.json
    let device = std::env::var("REIMAGINE_SDXL_REAL_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
    let backend_config = InferenceBackendConfig {
        schema_version: "1".to_owned(),
        backend: InferenceBackendKind::Candle,
        candle_device: device,
    };
    let backend_config_path = paths.config_dir().join("inference_backend.json");
    tokio::fs::write(
        &backend_config_path,
        serde_json::to_vec_pretty(&backend_config).unwrap(),
    )
    .await
    .unwrap();

    // manifest.json
    let manifest = build_manifest(&split_dir);
    let model_service = reimagine_app_host::ModelService::new(paths.clone());
    model_service
        .save_manifest(&manifest)
        .await
        .expect("save manifest");

    let recorder = Arc::new(reimagine_axum_host::RunEventRecorder::new());
    let host = Arc::new(WorkspaceHost::with_defaults_and_backend(
        WorkspaceScope::new("ws-real-e2e"),
        &base_path,
        BackendSelection::Candle,
        recorder.clone() as Arc<dyn RunEventSink>,
    ));

    let app = build_router().with_state(AxumHostState::new(host.clone(), recorder.clone()));

    run_inline_workflow_to_png_artifact(
        app,
        host.clone(),
        &base_path,
        sdxl_workflow_json(),
        "corr-real-e2e",
    )
    .await;

    // Cleanup the temporary workspace.
    let _ = tokio::fs::remove_dir_all(&base_path).await;
}

#[tokio::test]
#[ignore = "requires real SDXL split weights at REIMAGINE_SDXL_REAL_SPLIT_DIR"]
async fn real_sdxl_img2img_split_weights_produces_png_artifact() {
    let Some(split_dir) = std::env::var_os("REIMAGINE_SDXL_REAL_SPLIT_DIR").map(PathBuf::from)
    else {
        eprintln!(
            "skipping real img2img E2E test; set REIMAGINE_SDXL_REAL_SPLIT_DIR to a directory with Candle-compatible split SDXL weights"
        );
        return;
    };

    for path in required_split_paths(&split_dir) {
        assert!(
            path.exists(),
            "missing required split weight: {}",
            path.display()
        );
    }

    let base_path = unique_temp_dir("real-img2img-e2e");
    let paths = AppPaths::new(&base_path);
    tokio::fs::create_dir_all(paths.config_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.input_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.output_dir()).await.unwrap();
    tokio::fs::create_dir_all(paths.workflows_dir())
        .await
        .unwrap();

    let input_image_path = paths.input_dir().join("img2img-source.png");
    write_img2img_input_png(&input_image_path).await;

    let device = std::env::var("REIMAGINE_SDXL_REAL_DEVICE").unwrap_or_else(|_| "cpu".to_owned());
    let backend_config = InferenceBackendConfig {
        schema_version: "1".to_owned(),
        backend: InferenceBackendKind::Candle,
        candle_device: device,
    };
    let backend_config_path = paths.config_dir().join("inference_backend.json");
    tokio::fs::write(
        &backend_config_path,
        serde_json::to_vec_pretty(&backend_config).unwrap(),
    )
    .await
    .unwrap();

    let manifest = build_manifest(&split_dir);
    let model_service = reimagine_app_host::ModelService::new(paths.clone());
    model_service
        .save_manifest(&manifest)
        .await
        .expect("save manifest");

    let recorder = Arc::new(reimagine_axum_host::RunEventRecorder::new());
    let host = Arc::new(WorkspaceHost::with_defaults_and_backend(
        WorkspaceScope::new("ws-real-img2img-e2e"),
        &base_path,
        BackendSelection::Candle,
        recorder.clone() as Arc<dyn RunEventSink>,
    ));

    let app = build_router().with_state(AxumHostState::new(host.clone(), recorder.clone()));

    run_inline_workflow_to_png_artifact(
        app,
        host.clone(),
        &base_path,
        sdxl_img2img_workflow_json(),
        "corr-real-img2img-e2e",
    )
    .await;

    let _ = tokio::fs::remove_dir_all(&base_path).await;
}
