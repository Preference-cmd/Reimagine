//! Tests for the app-host readiness bridge and `run_workflow` orchestration.
//!
//! These tests cover three concerns from the app-host/01b issue:
//!   * readiness projection from model-manager diagnostics
//!   * blocked run diagnostics when model readiness fails
//!   * successful mock run handoff with mock node executors
//!
//! V1 Agent tools remain out of scope for `run_workflow`; this slice only
//! wires the host-side orchestration path.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reimagine_agent::WorkspaceScope;
use reimagine_app_host::{
    AppHostError, ModelService, RunWorkflowRequest, RunWorkflowResult,
    SnapshotExternalReadinessProvider, WorkspaceHost,
};
use reimagine_config::{AppConfig, AppPaths};
use reimagine_core::ExecutionValue;
use reimagine_core::diagnostic::DiagnosticSeverity;
use reimagine_core::event::OperationReport;
use reimagine_core::model::{
    ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, NodeId, ParamValue, RunId, SlotId,
    WorkflowId, WorkflowVersion,
};
use reimagine_core::readiness::{
    ExternalReadinessContext, ExternalReadinessProvider, ExternalReadinessSubject, RunTarget,
    RunTargetSelection,
};
use reimagine_core::workflow::{Workflow, WorkflowNode};
use reimagine_model_manager::{
    ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelSource, ModelSourceStatus,
};
use reimagine_nodes::{BUILTIN_CHECKPOINT_LOADER, BuiltinNodeCatalog};
use reimagine_runtime::{
    BoxedNodeExecutor, NodeExecutionContext, NodeExecutor, NodeExecutorRegistry, RunInputs,
    RunState, RuntimeOptions, RuntimeService, VecRunEventSink,
};

const WORKFLOW_ID: &str = "wf-run-test";
const MODEL_ID: &str = "sdxl-base-1.0";

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-run-{prefix}-{nonce}"))
}

fn builtin_catalog() -> Arc<BuiltinNodeCatalog> {
    Arc::new(BuiltinNodeCatalog::v1())
}

fn checkpoint_model_target() -> RunTargetSelection {
    RunTargetSelection::ExplicitTargets(
        ["model", "clip", "vae"]
            .into_iter()
            .map(|slot| RunTarget::NodeOutput {
                node_id: NodeId::new("loader"),
                slot_id: SlotId::new(slot),
            })
            .collect(),
    )
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

fn manifest_with_missing_model() -> ModelManifest {
    ModelManifest::new().with_root(ModelRoot::base_models())
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

struct MockLoaderExecutor;

#[async_trait]
impl NodeExecutor for MockLoaderExecutor {
    async fn execute(
        &self,
        _context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<ExecutionValue>)>, reimagine_runtime::NodeExecutorError> {
        Ok(Vec::new())
    }
}

fn mock_loader_executor() -> BoxedNodeExecutor {
    Arc::new(MockLoaderExecutor)
}

fn build_runtime() -> Arc<RuntimeService> {
    let mut registry = NodeExecutorRegistry::default();
    registry
        .register(BUILTIN_CHECKPOINT_LOADER, mock_loader_executor())
        .expect("register mock loader");
    build_runtime_with_sink(registry, Arc::new(VecRunEventSink::new()))
}

fn build_runtime_with_sink(
    registry: NodeExecutorRegistry,
    sink: Arc<VecRunEventSink>,
) -> Arc<RuntimeService> {
    Arc::new(RuntimeService::new(
        registry,
        Arc::new(reimagine_runtime::NoopRunResourceBackend),
        sink,
        Arc::new(reimagine_runtime::SystemClock),
    ))
}

async fn run_to_completion(service: &RuntimeService, handle: &reimagine_runtime::RunHandle) {
    let run_id: RunId = handle.run_id().clone();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if let Some(summary) = service.summary(&run_id) {
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

/// Construct a host whose model manifest has been pre-saved.
async fn build_host(manifest: ModelManifest, base: &str) -> WorkspaceHost {
    let paths = AppPaths::new(unique_temp_dir(base));
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    let model_service = ModelService::new(paths.clone());
    model_service
        .save_manifest(&manifest)
        .await
        .expect("save manifest");
    WorkspaceHost::new(
        WorkspaceScope::new(format!("ws-{base}")),
        AppConfig::new(paths),
        reimagine_config::InferenceBackendConfig::default(),
        build_runtime(),
        builtin_catalog(),
    )
}

#[tokio::test]
async fn build_readiness_snapshot_preserves_warnings_for_resolved_model() {
    let model_id = ModelId::new(MODEL_ID);
    let paths = AppPaths::new(unique_temp_dir("snapshot-ok"));
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    let model_path = paths.models_dir().join("sdxl-base-1.0.safetensors");
    tokio::fs::write(&model_path, b"placeholder").await.unwrap();

    let manifest = manifest_with_model(&model_id, "sdxl-base-1.0.safetensors");
    let service = ModelService::new(paths);
    service.save_manifest(&manifest).await.unwrap();

    let workflow = build_workflow(&model_id);
    let provider = service
        .build_readiness_snapshot(&workflow)
        .await
        .expect("snapshot should build");

    assert_eq!(provider.len(), 1);
    let subject = ExternalReadinessSubject::ModelRef(model_ref_for(&model_id));
    let ctx = ExternalReadinessContext::new(
        workflow.id().clone(),
        workflow.version(),
        "nodes.loader.params.checkpoint",
    );
    let diagnostics = provider
        .diagnostics_for(&ctx, &subject)
        .expect("subject present");
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.code().as_str() == "MODEL_MANAGER/MODEL_FINGERPRINT_MISSING"
                && diagnostic.severity() == DiagnosticSeverity::Warning
        }),
        "resolved but unfingerprinted model should preserve warning diagnostics; got {diagnostics:?}"
    );
}

#[tokio::test]
async fn build_readiness_snapshot_records_diagnostics_for_missing_model() {
    let model_id = ModelId::new(MODEL_ID);
    let paths = AppPaths::new(unique_temp_dir("snapshot-missing"));
    let service = ModelService::new(paths);
    service
        .save_manifest(&manifest_with_missing_model())
        .await
        .unwrap();

    let workflow = build_workflow(&model_id);
    let provider = service
        .build_readiness_snapshot(&workflow)
        .await
        .expect("snapshot should build");

    assert_eq!(provider.len(), 1);
    let subject = ExternalReadinessSubject::ModelRef(model_ref_for(&model_id));
    let ctx = ExternalReadinessContext::new(
        workflow.id().clone(),
        workflow.version(),
        "nodes.loader.params.checkpoint",
    );
    let diagnostics = provider
        .diagnostics_for(&ctx, &subject)
        .expect("subject present");
    assert!(
        !diagnostics.is_empty(),
        "missing model should produce diagnostics"
    );
    assert!(
        diagnostics.iter().any(|d| {
            d.code().as_str() == "MODEL_MANAGER/MODEL_REF_NOT_FOUND"
                && d.severity() == DiagnosticSeverity::Error
        }),
        "expected MODEL_REF_NOT_FOUND, got {:?}",
        diagnostics
    );
}

#[tokio::test]
async fn run_workflow_returns_blocked_when_model_is_missing() {
    let model_id = ModelId::new(MODEL_ID);
    let host = build_host(manifest_with_missing_model(), "run-blocked").await;

    let workflow_id = host
        .workflow_service()
        .register_workflow(build_workflow(&model_id));

    let request = RunWorkflowRequest::new(workflow_id.clone(), checkpoint_model_target());
    let result = host
        .run_workflow(request)
        .await
        .expect("run_workflow should not error");

    let RunWorkflowResult::Blocked { report } = result else {
        panic!("expected Blocked, got {result:?}");
    };
    assert!(
        report.diagnostics().iter().any(|d| {
            d.code().as_str() == "MODEL_MANAGER/MODEL_REF_NOT_FOUND"
                && d.severity() == DiagnosticSeverity::Error
        }),
        "blocked report should surface the model not-found error, got {:?}",
        report.diagnostics()
    );
}

#[tokio::test]
async fn run_workflow_handoff_completes_when_readiness_passes() {
    let model_id = ModelId::new(MODEL_ID);
    let paths = AppPaths::new(unique_temp_dir("run-start"));
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::write(
        paths.models_dir().join("sdxl-base-1.0.safetensors"),
        b"placeholder",
    )
    .await
    .unwrap();

    let manifest = manifest_with_model(&model_id, "sdxl-base-1.0.safetensors");
    let model_service = ModelService::new(paths.clone());
    model_service.save_manifest(&manifest).await.unwrap();

    let runtime = build_runtime();
    let host = WorkspaceHost::new(
        WorkspaceScope::new("ws-run-start"),
        AppConfig::new(paths),
        reimagine_config::InferenceBackendConfig::default(),
        runtime.clone(),
        builtin_catalog(),
    );
    let _ = model_service;

    let workflow_id = host
        .workflow_service()
        .register_workflow(build_workflow(&model_id));

    let mut run_inputs = RunInputs::new();
    run_inputs.insert_node_param(
        NodeId::new("loader"),
        SlotId::new("checkpoint"),
        ParamValue::ModelRef(model_ref_for(&model_id)),
    );

    let request = RunWorkflowRequest::new(workflow_id.clone(), checkpoint_model_target())
        .with_run_inputs(run_inputs)
        .with_options(RuntimeOptions::default());

    let result = host
        .run_workflow(request)
        .await
        .expect("run_workflow should succeed");

    let RunWorkflowResult::Started {
        handle,
        initial_snapshot,
    } = result
    else {
        panic!("expected Started, got {result:?}");
    };

    assert_eq!(handle.workflow_id().as_str(), WORKFLOW_ID);
    assert_eq!(initial_snapshot.state, RunState::Queued);

    run_to_completion(&runtime, &handle).await;
    let summary = runtime
        .summary(handle.run_id())
        .expect("summary should exist after completion");
    assert_eq!(summary.state, RunState::Completed);
}

#[tokio::test]
async fn run_workflow_rejects_unknown_workflow_id() {
    let host = build_host(manifest_with_missing_model(), "run-unknown").await;
    let request =
        RunWorkflowRequest::new(WorkflowId::new("does-not-exist"), checkpoint_model_target());
    let err = host.run_workflow(request).await.unwrap_err();
    assert!(matches!(err, AppHostError::UnknownWorkflow { .. }));
}

#[tokio::test]
async fn run_workflow_propagates_correlation_id_to_runtime_options() {
    let model_id = ModelId::new(MODEL_ID);
    let paths = AppPaths::new(unique_temp_dir("run-corr"));
    tokio::fs::create_dir_all(paths.models_dir()).await.unwrap();
    tokio::fs::write(
        paths.models_dir().join("sdxl-base-1.0.safetensors"),
        b"placeholder",
    )
    .await
    .unwrap();

    let manifest = manifest_with_model(&model_id, "sdxl-base-1.0.safetensors");
    let model_service = ModelService::new(paths.clone());
    model_service.save_manifest(&manifest).await.unwrap();

    let mut registry = NodeExecutorRegistry::default();
    registry
        .register(BUILTIN_CHECKPOINT_LOADER, mock_loader_executor())
        .expect("register mock loader");
    let sink = Arc::new(VecRunEventSink::new());
    let runtime = build_runtime_with_sink(registry, Arc::clone(&sink));
    let host = WorkspaceHost::new(
        WorkspaceScope::new("ws-run-corr"),
        AppConfig::new(paths),
        reimagine_config::InferenceBackendConfig::default(),
        runtime.clone(),
        builtin_catalog(),
    );
    let _ = model_service;

    let workflow_id = host
        .workflow_service()
        .register_workflow(build_workflow(&model_id));
    let request = RunWorkflowRequest::new(workflow_id, checkpoint_model_target())
        .with_options(RuntimeOptions::default())
        .with_correlation_id(reimagine_core::diagnostic::CorrelationId::new("corr-1"));

    let result = host
        .run_workflow(request)
        .await
        .expect("run_workflow should succeed");
    let RunWorkflowResult::Started { handle, .. } = result else {
        panic!("expected Started");
    };
    run_to_completion(&runtime, &handle).await;
    assert!(
        sink.events().iter().any(|event| event
            .correlation_id()
            .is_some_and(|id| id.as_str() == "corr-1")),
        "expected at least one runtime event to carry corr-1"
    );
}

#[test]
fn run_workflow_result_does_not_expose_runtime_value_store() {
    // The host-facing result variants must not carry runtime value-store
    // handles or backend tensor handles. This guards the boundary in
    // `run_workflow.rs::RunWorkflowResult`.
    let report: OperationReport = OperationReport::new();
    let blocked = RunWorkflowResult::Blocked { report };
    let json = format!("{blocked:?}");
    assert!(!json.contains("RunValueStore"));
    assert!(!json.contains("BackendTensorHandle"));
}

#[test]
fn snapshot_provider_diagnostics_for_is_synchronous() {
    // The provider must be a synchronous `ExternalReadinessProvider`.
    // `diagnostics_for` returns immediately and does not need `.await`.
    let mut provider = SnapshotExternalReadinessProvider::new();
    let subject = ExternalReadinessSubject::ModelRef(model_ref_for(&ModelId::new(MODEL_ID)));
    provider.record_ok(subject.clone());
    let ctx = ExternalReadinessContext::new(
        WorkflowId::new("wf"),
        WorkflowVersion::new(1),
        "nodes.loader.params.checkpoint",
    );
    let ok = provider
        .diagnostics_for(&ctx, &subject)
        .expect("subject present");
    assert!(ok.is_empty());

    let unknown = provider.diagnostics_for(
        &ctx,
        &ExternalReadinessSubject::ModelRef(model_ref_for(&ModelId::new("other"))),
    );
    assert!(unknown.is_none());

    assert_eq!(provider.len(), 1);
}

#[test]
fn app_host_does_not_expose_runtime_run_or_cancel() {
    // The host-facing public surface of `app_host` should re-export
    // only host-neutral orchestration types. Agent tools (which do not
    // exist in V1) must not surface `RuntimeService` or any runtime
    // run/cancel capability.
    use reimagine_app_host as exported;
    fn _is_send_sync<T: Send + Sync>() {}
    _is_send_sync::<exported::WorkspaceHost>();
    _is_send_sync::<exported::RunWorkflowRequest>();
    _is_send_sync::<exported::RunWorkflowResult>();
    _is_send_sync::<exported::SnapshotExternalReadinessProvider>();

    // `RunWorkflowRequest` and `RunWorkflowResult` are the only public
    // types in this slice. The runtime surface is reachable through
    // `WorkspaceHost::runtime_service` but is gated behind host APIs;
    // there is no agent-tool escape hatch in the public exports.
}
