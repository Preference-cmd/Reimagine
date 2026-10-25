//! End-to-end process tests for the Burn worker binary.
//!
//! The real `reimagine-inference-burn-worker` binary is launched by the
//! worker supervisor over a private Unix socket (the v2 gRPC contract):
//! handshake, execution, cancellation, health, cleanup, shutdown, the
//! stdin lifeline, and socket cleanup. Uses the
//! `CARGO_BIN_EXE_reimagine-inference-burn-worker` env var that Cargo
//! sets at test build time.

use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use reimagine_backend_worker_host::ProtocolRange;
use reimagine_backend_worker_host::{
    ExpectedWorkerIdentity, ProcessInferenceBackend, WorkerLaunchSpec, WorkerLimits,
    WorkerSupervisor,
};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, TerminalOutcome, WorkerInstallationId, WorkerInstanceProfile,
};
use reimagine_core::model::NodeId;
use reimagine_inference::{
    CreateEmptyLatentRequest, DiffusionSampleRequest, ImageImportRequest, ImagePreviewRequest,
    ImageSaveRequest, InferenceBackend, LatentDecodeRequest, LatentEncodeRequest,
    ResolvedImageSource, SamplerName, SchedulerName,
};

/// Helper to build a `ModelService` for tests with a mock `ModelAcquisitionService`.
fn test_model_service(paths: reimagine_config::AppPaths) -> reimagine_app_host::ModelService {
    use std::sync::Arc;
    let config = reimagine_config::AppConfig::new(paths.clone());
    let acquisition_service = Arc::new(reimagine_app_host::ModelAcquisitionService::new(
        paths.clone(),
        &config,
    ));
    reimagine_app_host::ModelService::new(paths, acquisition_service)
}

mod tiny_fixture {
    include!("../../burn/tests/tiny_sdxl_e2e.rs");
}

fn process_launch_spec(
    models_dir: &std::path::Path,
    output_dir: &std::path::Path,
) -> WorkerLaunchSpec {
    #[cfg(feature = "cuda")]
    let backend_instance_id = "burn:cuda:0";
    #[cfg(feature = "rocm")]
    let backend_instance_id = "burn:rocm:0";
    #[cfg(feature = "wgpu")]
    let backend_instance_id = "burn:wgpu:default";
    #[cfg(all(
        not(any(feature = "cuda", feature = "rocm", feature = "wgpu")),
        feature = "flex"
    ))]
    let backend_instance_id = "burn:flex:cpu";

    WorkerLaunchSpec {
        executable: env!("CARGO_BIN_EXE_reimagine-inference-burn-worker").into(),
        expected: ExpectedWorkerIdentity {
            backend_instance_id: BackendInstanceId::from(backend_instance_id),
            installation_id: WorkerInstallationId::from("dev"),
            backend_kind: "burn".to_owned(),
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            manifest_digest: "dev".to_owned(),
        },
        supported_protocols: ProtocolRange::new(2, 2),
        limits: WorkerLimits {
            startup_timeout: Duration::from_secs(15),
            request_timeout: Duration::from_secs(120),
            ..WorkerLimits::default()
        },
        environment: vec![
            (
                "REIMAGINE_MODELS_DIR".to_owned(),
                models_dir.display().to_string(),
            ),
            (
                "REIMAGINE_OUTPUT_DIR".to_owned(),
                output_dir.display().to_string(),
            ),
            (
                "REIMAGINE_ALLOWED_MODEL_ROOTS".to_owned(),
                models_dir.display().to_string(),
            ),
            (
                "REIMAGINE_ALLOWED_OUTPUT_ROOTS".to_owned(),
                output_dir.display().to_string(),
            ),
        ],
    }
}

#[tokio::test]
async fn app_host_activates_selected_worker_from_injected_inventory() {
    use reimagine_app_host::WorkspaceScope;
    use reimagine_app_host::{
        StaticWorkerInventoryProvider, WorkerBackendCandidate, WorkerInventorySnapshot,
        WorkspaceHost,
    };
    use reimagine_config::{AppPaths, InferenceBackendConfig};

    let workspace = tempfile::tempdir().expect("workspace");
    let paths = AppPaths::new(workspace.path());
    paths.ensure_all().await.expect("workspace dirs");
    let launch = process_launch_spec(paths.models_dir(), paths.output_dir());
    let instance = launch.expected.backend_instance_id.clone();
    let candidate = WorkerBackendCandidate::try_new(
        launch,
        WorkerInstanceProfile {
            backend_instance_id: instance.clone(),
            device_label: instance
                .0
                .strip_prefix("burn:")
                .unwrap_or("wgpu:default")
                .to_owned(),
            capabilities: vec![
                "model.load_bundle".to_owned(),
                "latent.create_empty".to_owned(),
                "text.encode".to_owned(),
                "diffusion.sample".to_owned(),
                "latent.decode".to_owned(),
                "latent.encode".to_owned(),
                "image.import".to_owned(),
                "image.save".to_owned(),
                "image.preview".to_owned(),
            ],
            operation_options: serde_json::json!({}),
        },
    )
    .expect("candidate");
    let host = WorkspaceHost::try_with_backend_config_and_worker_inventory(
        WorkspaceScope::new("mb04-real-worker"),
        workspace.path(),
        InferenceBackendConfig {
            selected_instance: Some(instance.0.clone()),
            ..InferenceBackendConfig::default()
        },
        reimagine_core::event::EventBus::new(),
        Arc::new(StaticWorkerInventoryProvider::new(
            WorkerInventorySnapshot::new(vec![candidate]),
        )),
    )
    .await
    .expect("app-host worker bootstrap");

    let profile = host.compute_profile();
    let selected = profile
        .backend_profiles
        .iter()
        .flat_map(|backend| &backend.instances)
        .find(|profile| profile.instance.as_str() == instance.0)
        .expect("live selected profile");
    assert_eq!(selected.capabilities.len(), 9);
    assert!(selected.diagnostics.is_empty());
    assert_eq!(host.resolved_backend_instance().as_str(), instance.0);
}

#[tokio::test]
async fn app_host_rejects_manifest_and_live_hello_profile_mismatch() {
    use reimagine_app_host::WorkspaceScope;
    use reimagine_app_host::{
        StaticWorkerInventoryProvider, WorkerBackendCandidate, WorkerInventorySnapshot,
        WorkspaceHost,
    };
    use reimagine_config::{AppPaths, InferenceBackendConfig};

    let workspace = tempfile::tempdir().expect("workspace");
    let paths = AppPaths::new(workspace.path());
    paths.ensure_all().await.expect("workspace dirs");
    let launch = process_launch_spec(paths.models_dir(), paths.output_dir());
    let instance = launch.expected.backend_instance_id.clone();
    let candidate = WorkerBackendCandidate::try_new(
        launch,
        WorkerInstanceProfile {
            backend_instance_id: instance.clone(),
            device_label: instance.0.strip_prefix("burn:").unwrap().to_owned(),
            capabilities: vec!["latent.create_empty".to_owned()],
            operation_options: serde_json::json!({}),
        },
    )
    .expect("candidate");
    let host = WorkspaceHost::try_with_backend_config_and_worker_inventory(
        WorkspaceScope::new("mb04-mismatch"),
        workspace.path(),
        InferenceBackendConfig {
            selected_instance: Some(instance.0.clone()),
            ..InferenceBackendConfig::default()
        },
        reimagine_core::event::EventBus::new(),
        Arc::new(StaticWorkerInventoryProvider::new(
            WorkerInventorySnapshot::new(vec![candidate]),
        )),
    )
    .await
    .expect("mismatch becomes readiness state");

    let profile = host.compute_profile();
    let selected = profile
        .backend_profiles
        .iter()
        .flat_map(|backend| &backend.instances)
        .find(|profile| profile.instance.as_str() == instance.0)
        .expect("selected profile");
    assert!(matches!(
        selected.status,
        reimagine_inference::BackendInstanceStatus::Unavailable
    ));
    assert!(selected.diagnostics[0].message().contains("live hello"));
}

#[tokio::test]
async fn workflow_reaches_png_through_process_backed_worker() {
    use reimagine_app_host::WorkspaceScope;
    use reimagine_app_host::{
        RunWorkflowRequest, RunWorkflowResult, StaticWorkerInventoryProvider,
        WorkerBackendCandidate, WorkerInventorySnapshot, WorkspaceHost,
    };
    use reimagine_config::{AppPaths, InferenceBackendConfig};
    use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant, NodeId};
    use reimagine_core::readiness::{RunTarget, RunTargetSelection};
    use reimagine_model_manager::{
        ModelComponentSource, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelRootId,
        ModelSource, ModelSourceStatus,
    };
    use reimagine_runtime::TaskState;

    let workspace = tempfile::tempdir().expect("workspace");
    let paths = AppPaths::new(workspace.path());
    paths.ensure_all().await.expect("workspace dirs");
    let package_root = paths.models_dir().join("tiny-sdxl-burn");
    let _ = tiny_fixture::load_request(&package_root);

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
        ModelId::new("tiny-sdxl-burn"),
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
        .expect("manifest");

    let launch = process_launch_spec(paths.models_dir(), paths.output_dir());
    let instance = launch.expected.backend_instance_id.clone();
    let candidate = WorkerBackendCandidate::try_new(
        launch,
        WorkerInstanceProfile {
            backend_instance_id: instance.clone(),
            device_label: instance.0.strip_prefix("burn:").unwrap().to_owned(),
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
    .expect("candidate");
    let bus = reimagine_core::event::EventBus::new();
    let host = Arc::new(
        WorkspaceHost::try_with_backend_config_and_worker_inventory(
            WorkspaceScope::new("mb04-tiny"),
            workspace.path(),
            InferenceBackendConfig {
                selected_instance: Some(instance.0.clone()),
                ..InferenceBackendConfig::default()
            },
            bus,
            Arc::new(StaticWorkerInventoryProvider::new(
                WorkerInventorySnapshot::new(vec![candidate]),
            )),
        )
        .await
        .expect("host"),
    );

    let workflow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../app-host/tests/fixtures/sdxl-base-burn-smoke-workflow.json");
    let mut workflow: serde_json::Value =
        serde_json::from_slice(&std::fs::read(workflow_path).expect("workflow fixture"))
            .expect("workflow json");
    workflow["id"] = "wf-tiny-sdxl".into();
    workflow["nodes"][0]["params"]["checkpoint"]["value"]["id"] = "tiny-sdxl-burn".into();
    workflow["nodes"][5]["params"]["width"]["value"] = 64.into();
    workflow["nodes"][5]["params"]["height"]["value"] = 64.into();

    let parsed_workflow: reimagine_core::workflow::Workflow =
        serde_json::from_value(workflow).expect("valid workflow");
    let workflow_id = host.workflow_service().register_workflow(parsed_workflow);

    let run_request = RunWorkflowRequest::new(
        workflow_id,
        RunTargetSelection::ExplicitTargets(vec![RunTarget::Node {
            node_id: NodeId::new("node_save_image"),
        }]),
    );
    let run_result = host.run_workflow(run_request).await.expect("start run");
    let RunWorkflowResult::Started { handle, .. } = run_result else {
        panic!("expected run to start, got {run_result:?}");
    };
    let run_id = handle.run_id().clone();

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let summary = loop {
        if let Some(summary) = host.run_summary(&run_id)
            && summary.state.is_terminal()
        {
            break summary;
        }
        assert!(std::time::Instant::now() < deadline, "run timeout");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };
    assert_eq!(
        summary.state,
        TaskState::Completed,
        "worker-backed run failed: {:?}",
        summary.diagnostics
    );
    assert!(!summary.artifacts.is_empty(), "artifacts must not be empty");
    let artifact_id = summary.artifacts[0].id.clone();
    let access = host
        .resolve_artifact(&artifact_id)
        .expect("resolve artifact");
    let bytes = std::fs::read(&access.path).expect("read artifact file");
    let image = image::load_from_memory(&bytes).expect("PNG");
    assert_eq!((image.width(), image.height()), (64, 64));
}

#[tokio::test]
async fn process_adapter_runs_tiny_sdxl_chain() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = Arc::new(
        WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
            .start()
            .await
            .expect("start real Burn worker"),
    );
    let backend = ProcessInferenceBackend::new(worker);

    let loaded = backend
        .load_bundle(tiny_fixture::load_request(root.path()))
        .await
        .expect("load tiny fixture through process");
    let positive = backend
        .text_encode(tiny_fixture::text_request(
            loaded.clip().clone(),
            "small bright city at sunrise",
            "process-positive",
        ))
        .await
        .expect("positive text.encode through process")
        .into_conditioning();
    let negative = backend
        .text_encode(tiny_fixture::text_request(
            loaded.clip().clone(),
            "low quality blur",
            "process-negative",
        ))
        .await
        .expect("negative text.encode through process")
        .into_conditioning();
    let latent = backend
        .create_empty_latent(CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-empty"),
        ))
        .await
        .expect("empty latent through process")
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
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-diffusion"),
        ))
        .await
        .expect("diffusion through process")
        .into_latent();
    let image = backend
        .latent_decode(LatentDecodeRequest::new(
            loaded.vae().clone(),
            sampled,
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-decode"),
        ))
        .await
        .expect("decode through process")
        .into_image();
    let preview = backend
        .image_preview(ImagePreviewRequest::new(
            image.clone(),
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-preview"),
        ))
        .await
        .expect("preview through process")
        .into_artifact();
    let saved = backend
        .image_save(
            ImageSaveRequest::new(
                image,
                tiny_fixture::run_id(),
                tiny_fixture::workflow_id(),
                tiny_fixture::workflow_version(),
                NodeId::new("process-save"),
            )
            .with_filename_prefix("process-tiny-sdxl"),
        )
        .await
        .expect("save through process")
        .into_artifact();

    tiny_fixture::assert_png_artifact(output.path(), preview.as_str());
    tiny_fixture::assert_png_artifact(output.path(), saved.as_str());
}

#[tokio::test]
async fn process_adapter_imports_and_encodes_image() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = Arc::new(
        WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
            .start()
            .await
            .expect("start real Burn worker"),
    );
    let backend = ProcessInferenceBackend::new(worker);

    let loaded = backend
        .load_bundle(tiny_fixture::load_request(root.path()))
        .await
        .expect("load tiny fixture through process");

    // Write a tiny PNG and import it through the process adapter.
    let png_path = root.path().join("import.png");
    let mut rgb = Vec::with_capacity(64 * 64 * 3);
    for index in 0..64 * 64 {
        rgb.extend_from_slice(&[
            (index % 255) as u8,
            ((index / 7) % 255) as u8,
            ((index / 13) % 255) as u8,
        ]);
    }
    {
        let mut out = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut out);
        image::ImageEncoder::write_image(encoder, &rgb, 64, 64, image::ColorType::Rgb8.into())
            .expect("encode fixture PNG");
        std::fs::write(&png_path, &out).expect("write fixture PNG");
    }

    let imported = backend
        .image_import(ImageImportRequest::new(
            ResolvedImageSource::new(&png_path, "image/png", Some("import.png".to_owned())),
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-import"),
        ))
        .await
        .expect("image.import through process")
        .into_image();
    assert_eq!((imported.width(), imported.height()), (64, 64));

    let encoded = backend
        .latent_encode(LatentEncodeRequest::new(
            loaded.vae().clone(),
            imported,
            tiny_fixture::run_id(),
            tiny_fixture::workflow_id(),
            tiny_fixture::workflow_version(),
            NodeId::new("process-encode"),
        ))
        .await
        .expect("latent.encode through process")
        .into_latent();
    assert_eq!(
        encoded.payload().shape().dims(),
        &[1_usize, 4, 8, 8],
        "64x64 image encodes to an SDXL-base 8x8 latent over IPC"
    );
}

// ---------------------------------------------------------------------------
// v2 contract process tests (real worker, supervisor over a Unix socket)
// ---------------------------------------------------------------------------

use reimagine_inference::{CancelToken, InferenceInvocation, NoopProgressSink};

/// The worker's hello (identity + profile + protocol version) must
/// match the launch expectation, which `start` already enforces; this
/// test also asserts the reported shape.
#[tokio::test]
async fn worker_handshake_returns_valid_hello() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
        .start()
        .await
        .expect("real worker starts and greets");

    assert_eq!(worker.hello.protocol.0, 2, "worker must report protocol v2");
    assert_eq!(worker.hello.identity.backend_kind, "burn");
    assert!(
        !worker.hello.identity.incarnation_id.0.is_empty(),
        "incarnation id must be non-empty"
    );
    assert!(
        !worker.hello.identity.worker_version.is_empty(),
        "worker version must be non-empty"
    );
    assert!(
        !worker.hello.profile.instances.is_empty(),
        "profile must have at least one instance"
    );

    worker.shutdown().await.expect("graceful shutdown");
}

/// The direct `latent.create_empty` operation echoes the request shape
/// and hands back a worker token.
#[tokio::test]
async fn worker_create_empty_latent_returns_expected_response() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
        .start()
        .await
        .expect("real worker starts");

    let result = worker
        .request(
            "latent.create_empty",
            serde_json::json!({
                "width": 64,
                "height": 64,
                "batch_size": 1,
            }),
        )
        .await
        .expect("execute over the v2 contract");
    match result.terminal {
        TerminalOutcome::Success { output } => {
            assert!(
                output
                    .get("worker_token")
                    .and_then(|v| v.as_str())
                    .is_some(),
                "response must contain worker_token: {output}"
            );
            assert_eq!(output.get("width").and_then(|v| v.as_u64()), Some(64));
            assert_eq!(output.get("height").and_then(|v| v.as_u64()), Some(64));
            assert_eq!(output.get("batch_size").and_then(|v| v.as_u64()), Some(1));
        }
        other => panic!("expected success outcome, got {other:?}"),
    }

    worker.shutdown().await.expect("graceful shutdown");
}

#[tokio::test]
async fn worker_unknown_operation_returns_error() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
        .start()
        .await
        .expect("real worker starts");

    let result = worker
        .request("nonexistent_op", serde_json::json!({}))
        .await
        .expect("execute over the v2 contract");
    match result.terminal {
        TerminalOutcome::BackendError { error } => {
            assert_eq!(error.code, "unknown_operation");
        }
        other => panic!("expected backend error for unknown operation, got {other:?}"),
    }

    worker.shutdown().await.expect("graceful shutdown");
}

#[tokio::test]
async fn cleanup_reports_actual_run_removals_once() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
        .start()
        .await
        .expect("real worker starts");

    let result = worker
        .request(
            "latent.create_empty",
            serde_json::json!({
                "width": 64,
                "height": 64,
                "batch_size": 1,
                "run_id": "cleanup-run",
            }),
        )
        .await
        .expect("execute over the v2 contract");
    assert!(matches!(result.terminal, TerminalOutcome::Success { .. }));

    let released_first = worker
        .cleanup(Some("cleanup-run".to_owned()), Vec::new())
        .await
        .expect("cleanup call");
    let released_second = worker
        .cleanup(Some("cleanup-run".to_owned()), Vec::new())
        .await
        .expect("cleanup call");
    assert_eq!(released_first, 1, "first cleanup removes the run payload");
    assert_eq!(released_second, 0, "second cleanup finds nothing left");

    worker.shutdown().await.expect("graceful shutdown");
}

/// Cancelling mid-execution drops the Execute call; the host reports
/// the single Cancelled terminal even though the worker's model load
/// would eventually have succeeded.
#[tokio::test]
async fn cancel_during_execution_reports_cancelled() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = Arc::new(
        WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
            .start()
            .await
            .expect("real worker starts"),
    );
    let backend = ProcessInferenceBackend::new(Arc::clone(&worker));

    let cancellation = CancelToken::new();
    let invocation = InferenceInvocation::new(
        tiny_fixture::run_id(),
        NodeId::new("cancel-mid-load"),
        None,
        cancellation.clone(),
        Arc::new(NoopProgressSink),
    );
    let backend_task = tokio::spawn(async move {
        backend
            .load_bundle_with_invocation(&invocation, tiny_fixture::load_request(root.path()))
            .await
    });

    // The tiny bundle load takes well over this on a cold worker; the
    // cancel therefore lands while the operation is in flight.
    tokio::time::sleep(Duration::from_millis(20)).await;
    cancellation.cancel();

    let error = backend_task
        .await
        .expect("task completes")
        .expect_err("cancelled load must fail");
    assert!(
        matches!(error, reimagine_inference::InferenceError::Cancelled),
        "expected Cancelled, got: {error}"
    );

    worker.shutdown().await.expect("graceful shutdown");
}

#[tokio::test]
async fn health_check_reports_healthy() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
        .start()
        .await
        .expect("real worker starts");

    let health = worker.health().await.expect("health RPC");
    assert!(health.healthy);

    worker.shutdown().await.expect("graceful shutdown");
}

/// The graceful shutdown path removes the worker's private socket file
/// and its 0700 directory.
#[tokio::test]
async fn shutdown_removes_the_worker_socket() {
    let root = tempfile::tempdir().expect("package root");
    let output = tempfile::tempdir().expect("output dir");
    let worker = WorkerSupervisor::new(process_launch_spec(root.path(), output.path()))
        .start()
        .await
        .expect("real worker starts");

    let endpoint = worker.transport_endpoint();
    #[cfg(windows)]
    if LocalEndpoint::parse(&endpoint)
        .expect("local endpoint")
        .to_uri()
        .starts_with("pipe://")
    {
        return; // nothing to unlink for Windows named pipes
    }
    #[cfg(not(windows))]
    let socket_path = {
        let path = endpoint
            .strip_prefix("unix://")
            .expect("local workers use unix:// endpoints");
        std::path::PathBuf::from(path)
    };
    assert!(
        socket_path.exists(),
        "socket file must exist while the worker runs"
    );
    let socket_dir = socket_path
        .parent()
        .expect("socket lives in its own directory")
        .to_path_buf();

    worker.shutdown().await.expect("graceful shutdown");
    assert!(
        !socket_path.exists(),
        "socket file must be removed after shutdown"
    );
    assert!(
        !socket_dir.exists(),
        "the private socket directory must be removed after shutdown"
    );
}

// ---------------------------------------------------------------------------
// Lifeline
// ---------------------------------------------------------------------------

/// A dead host closes both ends at once: stdin reaches EOF and stderr
/// becomes a broken pipe, so the worker's own logging fails too. The
/// worker must still exit.
#[test]
fn worker_exits_when_the_host_dies() {
    let socket_dir = tempfile::tempdir().expect("socket dir");
    let socket_path = socket_dir.path().join("worker.sock");
    let mut child = Command::new(env!("CARGO_BIN_EXE_reimagine-inference-burn-worker"))
        .arg("--listen")
        .arg(format!("unix://{}", socket_path.display()))
        .env_clear()
        .env("REIMAGINE_WORKER_TOKEN", "lifeline-token")
        .env("REIMAGINE_MODELS_DIR", "/tmp")
        .env("REIMAGINE_OUTPUT_DIR", "/tmp")
        .env("REIMAGINE_ALLOWED_MODEL_ROOTS", "/tmp")
        .env("REIMAGINE_ALLOWED_OUTPUT_ROOTS", "/tmp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn burn worker process");

    // Wait for the READY line before closing the pipes.
    let mut stdout = std::io::BufReader::new(child.stdout.take().expect("stdout piped"));
    let mut ready = String::new();
    std::io::BufRead::read_line(&mut stdout, &mut ready).expect("read READY line");
    assert!(
        ready.starts_with("READY "),
        "expected a READY line, got: {ready}"
    );
    drop(stdout);
    let stderr = child.stderr.take();
    drop(stderr);

    // Closing stdin is the lifeline going away.
    drop(child.stdin.take());

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if child.try_wait().expect("poll worker").is_some() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let _ = child.kill();
    let _ = child.wait();
    panic!("worker did not exit within 5s of stdin EOF");
}
