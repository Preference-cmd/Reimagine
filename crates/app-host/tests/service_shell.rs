use reimagine_agent::{AgentMode, AgentSessionId, ProviderName, WorkspaceScope};
use reimagine_app_host::{AppHost, ModelService, WorkflowService, WorkspaceHost};
use reimagine_config::AppPaths;
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResultStatus,
    WorkflowCommand,
};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    CommandBatchId, ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant, NodeCatalog, NodeId,
    NodeTypeId, ParamValue, SlotId, WorkflowVersion,
};
use reimagine_core::workflow::Workflow;
use reimagine_model_manager::{
    ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelSource, ModelSourceStatus,
};
use reimagine_nodes::{BUILTIN_CHECKPOINT_LOADER, BUILTIN_STRING, BuiltinNodeCatalog};

#[test]
fn app_host_owns_single_workspace_arc() {
    let workspace = WorkspaceHost::with_defaults(WorkspaceScope::new("ws-main"), temp_dir("app"));
    let app = AppHost::new(workspace);

    assert_eq!(app.workspace().workspace_scope().as_str(), "ws-main");
    assert_eq!(
        app.workspace().workflow_service().list_workflow_ids().len(),
        0
    );
}

#[tokio::test]
async fn workflow_service_previews_applies_saves_and_loads_workflow_json() {
    let paths = AppPaths::new(temp_dir("workflow"));
    let service = WorkflowService::new(paths.clone());
    let catalog = BuiltinNodeCatalog::v1();
    let workflow = Workflow::new("wf-1", WorkflowVersion::new(0));
    let workflow_id = service.register_workflow(workflow);

    let batch = add_string_node_batch(WorkflowVersion::new(0), "hello");
    let preview = service
        .preview_batch(&workflow_id, &catalog, batch.clone())
        .expect("preview should succeed");
    assert_eq!(preview.status(), CommandResultStatus::Applied);
    assert_eq!(preview.workflow_version(), WorkflowVersion::new(1));
    assert_eq!(
        service.snapshot(&workflow_id).unwrap().version(),
        WorkflowVersion::new(0)
    );

    let apply = service
        .apply_batch(&workflow_id, &catalog, batch)
        .expect("apply should succeed");
    assert_eq!(apply.status(), CommandResultStatus::Applied);
    assert_eq!(
        service.snapshot(&workflow_id).unwrap().version(),
        WorkflowVersion::new(1)
    );

    let saved_path = service
        .save_workflow(&workflow_id)
        .await
        .expect("workflow should save");
    assert_eq!(saved_path, paths.workflows_dir().join("wf-1.json"));

    let reloaded = WorkflowService::new(paths);
    let loaded_id = reloaded
        .load_workflow(&workflow_id)
        .await
        .expect("workflow should reload");
    assert_eq!(loaded_id, workflow_id);
    let loaded = reloaded.snapshot(&workflow_id).unwrap();
    assert_eq!(loaded.nodes().len(), 1);
    assert_eq!(loaded.version(), WorkflowVersion::new(1));
}

#[tokio::test]
async fn model_service_wraps_manifest_store_and_resolver() {
    let paths = AppPaths::new(temp_dir("model"));
    let model_path = paths.models_dir().join("sdxl-base.safetensors");
    tokio::fs::create_dir_all(paths.models_dir())
        .await
        .expect("models dir should be created");
    tokio::fs::write(&model_path, b"model").await.unwrap();

    let service = ModelService::new(paths.clone());
    let model_id = ModelId::new("stable_diffusion.sdxl.base");
    let descriptor = ModelDescriptor::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(
            ModelRoot::base_models().id().clone(),
            "sdxl-base.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available);
    let manifest = ModelManifest::new()
        .with_root(ModelRoot::base_models())
        .with_model(descriptor);

    let save_report = service
        .save_manifest(&manifest)
        .await
        .expect("manifest should save");
    assert!(save_report.diagnostics().is_empty());

    let models = service.list_models().await.expect("models should list");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id(), &model_id);
    assert!(service.cached_manifest().is_some());

    let model_ref = ModelRef::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    );
    let resolution = service
        .resolve_descriptor(&model_ref)
        .await
        .expect("resolution should run");
    assert!(resolution.is_resolved());
    assert_eq!(resolution.value().unwrap().id(), &model_id);
}

#[test]
fn agent_service_creates_workspace_scoped_sessions() {
    let workspace =
        WorkspaceHost::with_defaults(WorkspaceScope::new("ws-agent"), temp_dir("agent"));
    let service = workspace.agent_service();
    let session = service.create_session(
        AgentSessionId::new("session-1"),
        AgentMode::Build,
        ProviderName::new("anthropic"),
        "2026-06-10T00:00:00Z",
    );

    assert_eq!(session.workspace_scope().as_str(), "ws-agent");
    assert_eq!(session.mode(), AgentMode::Build);
    assert_eq!(service.list_sessions().len(), 1);
    assert_eq!(
        service
            .get_session(&AgentSessionId::new("session-1"))
            .unwrap()
            .provider()
            .as_str(),
        "anthropic"
    );
}

#[test]
fn workspace_host_exposes_v1_builtin_catalog_via_host_neutral_helpers() {
    let workspace =
        WorkspaceHost::with_defaults(WorkspaceScope::new("ws-catalog"), temp_dir("catalog"));
    let builtin = workspace.builtin_node_catalog();
    assert_eq!(builtin.len(), BuiltinNodeCatalog::v1().len());

    let defs = workspace.list_node_defs();
    assert_eq!(defs.len(), builtin.len());
    let string_def = workspace
        .find_node_def(&NodeTypeId::new(BUILTIN_STRING))
        .expect("builtin.string should be exposed");
    assert_eq!(string_def.type_id().as_str(), BUILTIN_STRING);

    assert!(
        workspace
            .find_node_def(&NodeTypeId::new("builtin.does_not_exist"))
            .is_none()
    );

    // Catalog must be a `NodeCatalog` so core validation and readiness
    // can consume it through the host service.
    let checkpoint_def = workspace
        .node_catalog()
        .get(&NodeTypeId::new(BUILTIN_CHECKPOINT_LOADER))
        .expect("checkpoint loader should be present");
    assert_eq!(checkpoint_def.type_id().as_str(), BUILTIN_CHECKPOINT_LOADER);
}

#[test]
fn workspace_host_alignment_is_clean_for_v1_default_composition() {
    let workspace =
        WorkspaceHost::with_defaults(WorkspaceScope::new("ws-aligned"), temp_dir("aligned"));
    let report = workspace.check_node_catalog_alignment();
    assert!(
        report.is_aligned(),
        "V1 default composition should be fully aligned; missing={:?} orphan={:?}",
        report.missing_executors(),
        report.orphan_executors()
    );
    assert!(report.diagnostics().is_empty());
}

#[test]
fn host_neutral_catalog_drives_workflow_validation() {
    // Locks in that the host-neutral `NodeCatalogService` exposed by
    // `WorkspaceHost` is the same catalog the readiness/validation
    // path consumes. This prevents accidental drift between the
    // surface UI/Tauri/Axum/Agent read from and the surface the
    // workflow validation reads from.
    let workspace = WorkspaceHost::with_defaults(
        WorkspaceScope::new("ws-host-catalog"),
        temp_dir("host-catalog"),
    );
    let service = workspace.workflow_service().clone();
    let workflow_id =
        service.register_workflow(Workflow::new("wf-host-catalog", WorkflowVersion::new(0)));

    let batch = add_string_node_batch(WorkflowVersion::new(0), "hello");
    let preview = service
        .preview_batch(&workflow_id, workspace.node_catalog().as_ref(), batch)
        .expect("preview should succeed through host-neutral catalog");
    assert_eq!(preview.status(), CommandResultStatus::Applied);

    // `find_node_def` and `node_catalog().get(...)` must agree: both
    // go through the same catalog.
    let via_helper = workspace
        .find_node_def(&NodeTypeId::new(BUILTIN_STRING))
        .expect("builtin.string should be discoverable via host helper");
    let via_trait = workspace
        .node_catalog()
        .get(&NodeTypeId::new(BUILTIN_STRING))
        .expect("builtin.string should be discoverable via NodeCatalog trait");
    assert_eq!(via_helper.type_id().as_str(), via_trait.type_id().as_str());
    assert_eq!(
        via_helper.input_slots().len(),
        via_trait.input_slots().len()
    );
}

fn add_string_node_batch(base_version: WorkflowVersion, value: &str) -> CommandBatch {
    CommandBatch::new(
        CommandBatchId::new(format!("batch-{base_version}")),
        CommandActor::new(CommandActorKind::Human),
        base_version,
        CommandProvenance::Direct,
        Timestamp::new("2026-06-10T00:00:00Z"),
        vec![WorkflowCommand::AddNode {
            node_id: NodeId::new("string-1"),
            type_id: NodeTypeId::new(BUILTIN_STRING),
            label: None,
            params: [(SlotId::new("value"), ParamValue::String(value.to_owned()))].into(),
            position: None,
        }],
    )
}

#[tokio::test]
async fn workspace_host_exposes_candle_backend_instance_snapshot() {
    let workspace =
        WorkspaceHost::with_defaults(WorkspaceScope::new("ws-backend"), temp_dir("backend"));
    let snapshots = workspace.backend_instance_snapshots().await;

    assert_eq!(
        snapshots.len(),
        1,
        "default composition should expose one backend instance"
    );
    let snapshot = &snapshots[0];
    assert_eq!(
        snapshot.backend_instance.to_string(),
        "candle:cpu",
        "instance should be candle:cpu"
    );
    assert_eq!(
        snapshot.plugin.as_ref().map(|p| p.as_str()),
        Some("builtin.candle"),
        "plugin provenance should be preserved"
    );
    assert_eq!(
        snapshot.extension.as_ref().map(|e| e.as_str()),
        Some("backend.candle"),
        "extension provenance should be preserved"
    );
    assert_eq!(
        snapshot.device.as_ref().map(|d| d.label.as_str()),
        Some("cpu"),
        "device label should be cpu"
    );
}

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let tid = std::thread::current().id();
    std::env::temp_dir().join(format!("reimagine-app-host-{prefix}-{nonce:?}-{tid:?}"))
}
