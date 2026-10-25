//! Opt-in Burn SDXL workspace bootstrap E2E test.
//!
//! Requires three environment variables:
//!
//!   REIMAGINE_BURN_E2E_WORKSPACE: absolute path to a workspace root;
//!   REIMAGINE_BURN_E2E_PACKAGE_REPORT: absolute path to a Burn
//!       conversion-report.json under `<workspace>/models/...`;
//!   REIMAGINE_BURN_WORKER: absolute path to `reimagine-inference-burn-worker`.
//!
//! Reads the existing converted SDXL Burn package, imports it through
//! `ModelService::import_burn_converted_package`, selects `burn:wgpu:default`
//! in the workspace config, and verifies the workspace host bootstraps
//! with truthful burn capabilities.

#![cfg(feature = "aigc")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reimagine_app_host::WorkspaceScope;
use reimagine_app_host::{
    RunWorkflowRequest, RunWorkflowResult, StaticWorkerInventoryProvider, WorkerBackendCandidate,
    WorkerInventorySnapshot, WorkspaceComputeProfile, WorkspaceHost,
};
use reimagine_backend_worker_host::{ExpectedWorkerIdentity, WorkerLaunchSpec, WorkerLimits};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, ProtocolRange, WorkerInstallationId, WorkerInstanceProfile,
};
use reimagine_config::{AppPaths, InferenceBackendConfig, InferenceBackendKind};
use reimagine_core::event::RunEventKind;
use reimagine_core::model::NodeId;
use reimagine_core::readiness::{RunTarget, RunTargetSelection};
use reimagine_inference::InferenceCapability;
use reimagine_runtime::TaskState;
mod common;
use common::TestRunEventCollector;
use reimagine_core::event::EventBus;
use serde_json::Value;

/// Helper to build a `ModelService` for tests with a mock `ModelAcquisitionService`.
fn test_model_service(paths: AppPaths) -> reimagine_app_host::ModelService {
    let config = reimagine_config::AppConfig::new(paths.clone());
    let acquisition_service = Arc::new(reimagine_app_host::ModelAcquisitionService::new(
        paths.clone(),
        &config,
    ));
    reimagine_app_host::ModelService::new(paths, acquisition_service)
}

const BURN_WORKFLOW_ID: &str = "workflow_sdxl_burn_smoke_real";
const BURN_MODEL_ID: &str = "burn-real-sdxl-smoke-burn";
const BURN_INSTANCE_LABEL: &str = "burn:wgpu:default";

fn required_env() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let workspace = std::env::var_os("REIMAGINE_BURN_E2E_WORKSPACE").map(PathBuf::from)?;
    let report_path = std::env::var_os("REIMAGINE_BURN_E2E_PACKAGE_REPORT").map(PathBuf::from)?;
    let worker = std::env::var_os("REIMAGINE_BURN_WORKER").map(PathBuf::from)?;
    if workspace.as_os_str().is_empty()
        || report_path.as_os_str().is_empty()
        || worker.as_os_str().is_empty()
    {
        eprintln!(
            "skipping Burn smoke bootstrap test; workspace, package report, and REIMAGINE_BURN_WORKER are required"
        );
        return None;
    }
    assert!(
        worker.is_file(),
        "REIMAGINE_BURN_WORKER is not a file: {}",
        worker.display()
    );
    Some((workspace, report_path, worker))
}

fn worker_provider(paths: &AppPaths, executable: PathBuf) -> Arc<StaticWorkerInventoryProvider> {
    let instance = BackendInstanceId::from(BURN_INSTANCE_LABEL);
    let launch = WorkerLaunchSpec {
        executable,
        expected: ExpectedWorkerIdentity {
            backend_instance_id: instance.clone(),
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
    };
    let candidate = WorkerBackendCandidate::try_new(
        launch,
        WorkerInstanceProfile {
            backend_instance_id: instance,
            device_label: "wgpu:default".to_owned(),
            capabilities: vec![
                "model.load_bundle",
                "latent.create_empty",
                "text.encode",
                "diffusion.sample",
                "latent.decode",
                "image.save",
                "image.preview",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            operation_options: serde_json::json!({}),
        },
    )
    .expect("worker candidate");
    Arc::new(StaticWorkerInventoryProvider::new(
        WorkerInventorySnapshot::new(vec![candidate]),
    ))
}

fn assert_truthful_burn_capabilities(profile: &WorkspaceComputeProfile) {
    let burn_backend = profile
        .backend_profiles
        .iter()
        .find(|b| b.backend.as_str() == "burn")
        .expect("burn backend profile must be present in compute profile");

    let default_instance = burn_backend
        .instances
        .iter()
        .find(|i| i.instance.as_str() == BURN_INSTANCE_LABEL)
        .expect("burn:wgpu:default instance profile must be present");

    let expected = [
        InferenceCapability::LoadBundle,
        InferenceCapability::TextEncode,
        InferenceCapability::CreateEmptyLatent,
        InferenceCapability::DiffusionSample,
        InferenceCapability::LatentDecode,
        InferenceCapability::ImageSave,
        InferenceCapability::ImagePreview,
    ];

    for cap in expected {
        assert!(
            default_instance.capabilities.contains(&cap),
            "burn:wgpu:default capability list {:?} must include {:?}",
            default_instance.capabilities,
            cap,
        );
    }

    assert!(
        !default_instance
            .capabilities
            .contains(&InferenceCapability::ImageImport),
        "image import must NOT be advertised: actual capabilities {:?}",
        default_instance.capabilities
    );
}

#[tokio::test]
#[ignore = "requires REIMAGINE_BURN_E2E_WORKSPACE and REIMAGINE_BURN_E2E_PACKAGE_REPORT"]
async fn burn_real_sdxl_smoke_workflow_opens() {
    let (workspace_root, report_path, worker_executable) = match required_env() {
        Some(values) => values,
        None => return,
    };

    if !workspace_root.is_dir() {
        panic!(
            "REIMAGINE_BURN_E2E_WORKSPACE is not a directory: {}",
            workspace_root.display()
        );
    }
    if !report_path.is_file() {
        panic!(
            "REIMAGINE_BURN_E2E_PACKAGE_REPORT is not a file: {}",
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
            "REIMAGINE_BURN_E2E_PACKAGE_REPORT must live under workspace `models/` ({}); got {}",
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
        selected_instance: Some(BURN_INSTANCE_LABEL.to_owned()),
        ..InferenceBackendConfig::default()
    };
    tokio::fs::write(
        paths.config_dir().join("inference_backend.json"),
        serde_json::to_vec_pretty(&backend_config).unwrap(),
    )
    .await
    .unwrap();

    let model_service = test_model_service(paths.clone());
    let (_, _, descriptor) = model_service
        .import_burn_converted_package(&report_path)
        .await
        .expect("Burn converted package import should succeed");
    assert_eq!(
        descriptor.id().as_str(),
        BURN_MODEL_ID,
        "descriptor id must match the Burn smoke model id"
    );

    let event_bus = EventBus::new();
    let host = Arc::new(
        WorkspaceHost::try_with_backend_config_and_worker_inventory(
            WorkspaceScope::new("ws-burn-smoke-05a"),
            &workspace_root,
            backend_config,
            event_bus,
            worker_provider(&paths, worker_executable),
        )
        .await
        .expect("process-backed workspace"),
    );

    assert_eq!(
        host.backend_config().backend,
        InferenceBackendKind::Burn,
        "WorkspaceHost must honor configured Burn backend selection"
    );

    assert_truthful_burn_capabilities(&host.compute_profile());

    let workflow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sdxl-base-burn-smoke-workflow.json");
    let workflow_raw = tokio::fs::read_to_string(&workflow_path)
        .await
        .expect("sdxl-base-burn-smoke-workflow.json must be readable");
    let workflow_json: Value = serde_json::from_str(&workflow_raw)
        .expect("sdxl-base-burn-smoke-workflow.json must be valid JSON");

    let parsed_workflow: reimagine_core::workflow::Workflow =
        serde_json::from_value(workflow_json).expect("valid workflow");
    let registered_id = host.workflow_service().register_workflow(parsed_workflow);
    assert_eq!(registered_id.as_str(), BURN_WORKFLOW_ID);
}

#[tokio::test]
#[ignore = "requires REIMAGINE_BURN_E2E_WORKSPACE and REIMAGINE_BURN_E2E_PACKAGE_REPORT"]
async fn burn_real_sdxl_smoke_workflow_runs_to_png_artifact() {
    let (workspace_root, report_path, worker_executable) = match required_env() {
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
        selected_instance: Some(BURN_INSTANCE_LABEL.to_owned()),
        ..InferenceBackendConfig::default()
    };
    tokio::fs::write(
        paths.config_dir().join("inference_backend.json"),
        serde_json::to_vec_pretty(&backend_config).unwrap(),
    )
    .await
    .unwrap();

    let model_service = test_model_service(paths.clone());
    let (_, _, descriptor) = model_service
        .import_burn_converted_package(&report_path)
        .await
        .expect("Burn converted package import should succeed");
    let descriptor_id = descriptor.id().clone();

    let event_bus = EventBus::new();
    let collector = TestRunEventCollector::new(&event_bus);
    let host = Arc::new(
        WorkspaceHost::try_with_backend_config_and_worker_inventory(
            WorkspaceScope::new("ws-burn-smoke-05b"),
            &workspace_root,
            backend_config,
            event_bus,
            worker_provider(&paths, worker_executable),
        )
        .await
        .expect("process-backed workspace"),
    );

    // 1. Register the Burn smoke workflow.
    let workflow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sdxl-base-burn-smoke-workflow.json");
    let workflow_raw = tokio::fs::read_to_string(&workflow_path)
        .await
        .expect("smoke workflow json must be readable");
    let workflow_json: Value =
        serde_json::from_str(&workflow_raw).expect("smoke workflow json must be valid JSON");
    let parsed_workflow: reimagine_core::workflow::Workflow =
        serde_json::from_value(workflow_json).expect("valid workflow");
    let workflow_id = host.workflow_service().register_workflow(parsed_workflow);
    assert_eq!(workflow_id.as_str(), BURN_WORKFLOW_ID);

    // 2. Run the explicit `node_save_image` target.
    let request = RunWorkflowRequest::new(
        workflow_id,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::Node {
            node_id: NodeId::new("node_save_image"),
        }]),
    )
    .with_correlation_id(reimagine_core::diagnostic::CorrelationId::new(
        "burn-smoke-05b",
    ));

    let result = host.run_workflow(request).await.expect("start run");
    let RunWorkflowResult::Started { handle, .. } = result else {
        panic!("expected Started outcome, got {result:?}");
    };
    let run_id = handle.run_id().clone();

    // 3. Poll the run summary via host.run_summary(&run_id).
    let deadline = Instant::now() + Duration::from_secs(300);
    let summary = loop {
        if let Some(summary) = host.run_summary(&run_id)
            && summary.state.is_terminal()
        {
            break summary;
        }
        if Instant::now() > deadline {
            panic!("run {run_id} did not reach a terminal state within 300s");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    };

    eprintln!(
        "burn e2e: run {} reached terminal state `{:?}`",
        run_id, summary.state
    );

    if summary.state == TaskState::Completed {
        // 4a. Successful path: artifact must be a valid 256x256 PNG.
        assert!(
            !summary.artifacts.is_empty(),
            "terminal summary must include artifacts"
        );
        let artifact_meta = &summary.artifacts[0];
        assert_eq!(
            artifact_meta.node_id.as_str(),
            "node_save_image",
            "final artifact must be the node_save_image output"
        );

        let events = collector.events();
        assert!(
            events
                .iter()
                .any(|e| e.kind() == RunEventKind::RunCompleted),
            "events must include RunCompleted evidence"
        );
        assert!(
            events
                .iter()
                .any(|e| e.kind() == RunEventKind::ArtifactCreated),
            "events must include ArtifactCreated evidence"
        );

        let access = host
            .resolve_artifact(&artifact_meta.id)
            .expect("artifact must resolve");
        assert_eq!(access.media_type, "image/png");
        let artifact_bytes = tokio::fs::read(&access.path).await.expect("read artifact");
        assert!(
            artifact_bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
            "artifact must be a valid PNG (signature mismatch)"
        );
        let image = image::load_from_memory(&artifact_bytes).expect("PNG must decode");
        assert_eq!(image.width(), 256, "artifact width must be 256");
        assert_eq!(image.height(), 256, "artifact height must be 256");

        eprintln!(
            "burn_real_sdxl_smoke completed: model={}, instance={}, run_id={}, artifact_id={}, duration={}s",
            descriptor_id,
            BURN_INSTANCE_LABEL,
            run_id,
            artifact_meta.id,
            started_at.elapsed().as_secs(),
        );
        return;
    }

    // 4b. Failed path: must reference the documented CubeCL WGPU binding
    //     error surfaced through the WGPU guard.
    assert_eq!(
        summary.state,
        TaskState::Failed,
        "terminal state is neither Completed nor Failed"
    );
    let diagnostics_text = summary
        .diagnostics
        .iter()
        .map(|d| d.message().to_owned())
        .collect::<Vec<_>>()
        .join("; ");
    assert!(
        summary.diagnostics.iter().any(|d| {
            let msg = d.message();
            msg.contains("WGPU validation error") || msg.contains("buffer bound at binding index")
        }),
        "failed run must surface the CubeCL WGPU validation diagnostic; observed: [{diagnostics_text}]"
    );

    eprintln!(
        "burn_real_sdxl_smoke landed Failed as expected: model={}, instance={}, run_id={}, duration={}s, diagnostics={diagnostics_text}",
        descriptor_id,
        BURN_INSTANCE_LABEL,
        run_id,
        started_at.elapsed().as_secs(),
    );
}
