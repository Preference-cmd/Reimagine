#![cfg(feature = "aigc")]

use reimagine_app_host::WorkspaceScope;
use reimagine_app_host::{AppHost, ModelService, WorkflowService, WorkspaceHost};
use reimagine_config::AppPaths;
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandStatus,
    OperationContext, WorkflowCommand,
};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    CommandBatchId, DocumentVersion, ModelId, ModelRef, ModelRole, ModelSeries, ModelVariant,
    NodeCatalog, NodeId, NodeTypeId, ParamValue, ProjectId, SlotId,
};
use reimagine_core::workflow::Workflow;
use reimagine_model_manager::{
    ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelSource, ModelSourceStatus,
};
use reimagine_nodes::{BUILTIN_CHECKPOINT_LOADER, BUILTIN_STRING, BuiltinNodeCatalog};

/// Helper to build a `ModelService` for tests with a mock `ModelAcquisitionService`.
fn test_model_service(paths: AppPaths) -> ModelService {
    use std::sync::Arc;
    let config = reimagine_config::AppConfig::new(paths.clone());
    let acquisition_service = Arc::new(reimagine_app_host::ModelAcquisitionService::new(
        paths.clone(),
        &config,
    ));
    ModelService::new(paths, acquisition_service)
}

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
    let workflow = Workflow::new("wf-1", DocumentVersion::new(0));
    let workflow_id = service.register_workflow(workflow);

    let batch = add_string_node_batch(DocumentVersion::new(0), "hello");
    let preview = service
        .preview_batch(&workflow_id, &catalog, batch.clone())
        .expect("preview should succeed");
    assert_eq!(preview.status(), CommandStatus::Applied);
    assert_eq!(preview.version(), DocumentVersion::new(1));
    assert_eq!(
        service.snapshot(&workflow_id).unwrap().version(),
        DocumentVersion::new(0)
    );

    let apply = service
        .apply_batch(&workflow_id, &catalog, batch)
        .expect("apply should succeed");
    assert_eq!(apply.status(), CommandStatus::Applied);
    assert_eq!(
        service.snapshot(&workflow_id).unwrap().version(),
        DocumentVersion::new(1)
    );

    let saved_path = service
        .save_workflow(&workflow_id)
        .await
        .expect("workflow should save");
    // AR-08: workflows persist under the default project layout.
    assert_eq!(
        saved_path,
        paths.project_workflows_dir("default").join("wf-1.json")
    );

    let reloaded = WorkflowService::new(paths);
    let loaded_id = reloaded
        .load_workflow(&workflow_id)
        .await
        .expect("workflow should reload");
    assert_eq!(loaded_id, workflow_id);
    let loaded = reloaded.snapshot(&workflow_id).unwrap();
    assert_eq!(loaded.nodes().len(), 1);
    assert_eq!(loaded.version(), DocumentVersion::new(1));
}

#[tokio::test]
async fn model_service_wraps_manifest_store_and_resolver() {
    let paths = AppPaths::new(temp_dir("model"));
    let model_path = paths.models_dir().join("sdxl-base.safetensors");
    tokio::fs::create_dir_all(paths.models_dir())
        .await
        .expect("models dir should be created");
    tokio::fs::write(&model_path, b"model").await.unwrap();

    let service = test_model_service(paths.clone());
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
        ModelFormat::SafeTensors,
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

#[tokio::test]
async fn model_service_imports_burn_package_report_and_persists_manifest() {
    let paths = AppPaths::new(temp_dir("burn-package-import"));
    let report_path = write_burn_package(paths.models_dir(), "sdxl-base-1.0", "sha256-abc123");
    let service = test_model_service(paths.clone());

    let (manifest, report, descriptor) = service
        .import_burn_converted_package(&report_path)
        .await
        .expect("burn package import should persist descriptor");

    assert!(report.diagnostics().is_empty());
    assert_eq!(descriptor.id().as_str(), "sdxl-base-1.0-burn");
    assert_eq!(manifest.models(), std::slice::from_ref(&descriptor));
    assert_eq!(descriptor.components().len(), 4);
    assert_eq!(
        descriptor.metadata().get("backend").map(String::as_str),
        Some("burn")
    );

    let reloaded = service.list_models().await.unwrap();
    assert_eq!(reloaded, vec![descriptor.clone()]);

    let model_ref = ModelRef::new(
        ModelId::new("sdxl-base-1.0-burn"),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    );
    let resolved = service
        .resolve_descriptor_with_components(&model_ref)
        .await
        .expect("resolver should run");
    assert!(
        resolved
            .report()
            .diagnostics()
            .iter()
            .all(|diagnostic| !diagnostic.severity().is_error()),
        "resolved burn package should not emit error diagnostics: {:?}",
        resolved.report().diagnostics()
    );
    assert_eq!(resolved.value().unwrap().components().len(), 4);
}

#[tokio::test]
async fn model_service_rejects_burn_package_descriptor_collision_without_saving() {
    let paths = AppPaths::new(temp_dir("burn-package-collision"));
    let first_report = write_burn_package(paths.models_dir(), "sdxl-base-1.0", "sha256-abc123");
    let second_report = write_burn_package(paths.models_dir(), "sdxl-base-1.0", "sha256-def456");
    let service = test_model_service(paths);
    let (_, _, original) = service
        .import_burn_converted_package(&first_report)
        .await
        .expect("first burn package import should persist descriptor");

    let error = service
        .import_burn_converted_package(&second_report)
        .await
        .expect_err("colliding burn package import should fail");

    assert!(
        error.to_string().contains("descriptor id collision"),
        "expected descriptor collision, got: {error}"
    );
    assert_eq!(service.list_models().await.unwrap(), vec![original]);
}

#[tokio::test]
async fn model_service_rejects_burn_package_report_outside_models_dir() {
    let paths = AppPaths::new(temp_dir("burn-package-outside-models"));
    std::fs::create_dir_all(paths.models_dir()).unwrap();
    let outside = temp_dir("burn-package-outside-report");
    let report_path = write_burn_package(&outside.join("models"), "sdxl-base-1.0", "sha256-abc123");
    let service = test_model_service(paths);

    let error = service
        .import_burn_converted_package(&report_path)
        .await
        .expect_err("outside package report should be rejected by app-host path boundary");

    assert!(
        error
            .to_string()
            .contains("Burn package report path must stay under models directory"),
        "expected app-host models directory boundary diagnostic, got: {error}"
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
    // surface UI/Tauri/Axum read from and the surface the
    // workflow validation reads from.
    let workspace = WorkspaceHost::with_defaults(
        WorkspaceScope::new("ws-host-catalog"),
        temp_dir("host-catalog"),
    );
    let service = workspace.workflow_service().clone();
    let workflow_id =
        service.register_workflow(Workflow::new("wf-host-catalog", DocumentVersion::new(0)));

    let batch = add_string_node_batch(DocumentVersion::new(0), "hello");
    let preview = service
        .preview_batch(&workflow_id, workspace.node_catalog().as_ref(), batch)
        .expect("preview should succeed through host-neutral catalog");
    assert_eq!(preview.status(), CommandStatus::Applied);

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

fn add_string_node_batch(
    base_version: DocumentVersion,
    value: &str,
) -> CommandBatch<WorkflowCommand> {
    CommandBatch::new(
        CommandBatchId::new(format!("batch-{base_version}")),
        OperationContext::new(
            CommandActor::new(CommandActorKind::Human),
            CommandProvenance::Direct,
        ),
        base_version,
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

fn write_burn_package(
    models_dir: &std::path::Path,
    source_model_id: &str,
    source_fingerprint: &str,
) -> std::path::PathBuf {
    let package_root = models_dir
        .join("converted/burn")
        .join(source_model_id)
        .join(source_fingerprint);
    for component in ["diffusion", "vae", "text_encoder", "text_encoder_2"] {
        let path = package_root.join(component).join("model.safetensors");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"component").unwrap();
    }

    let report_path = package_root.join("conversion-report.json");
    std::fs::write(
        &report_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "source_identity": source_model_id,
            "source_layout": "diffusers_style_split_safetensors",
            "target_contract_version": 1,
            "output_components": [],
            "mapped_tensor_count": 0,
            "ignored_tensor_families": [],
            "diagnostics": [],
            "package": {
                "schema_version": 1,
                "layout": "burn_native_component_package",
                "converter_version": "burn-sdxl-package-04c-v1",
                "package_root": ".",
                "created_at": 1,
                "source": {
                    "source_model_id": source_model_id,
                    "source_layout": "diffusers_style_split_safetensors",
                    "source_fingerprint": source_fingerprint,
                    "fingerprint_kind": "supplied",
                    "source_files": []
                },
                "target": {
                    "backend": "burn",
                    "contract": "burn.component",
                    "contract_version": 1,
                    "model_series": "stable_diffusion",
                    "variant": "sdxl"
                },
                "components": [
                    {
                        "component_role": "diffusion",
                        "model_role": "DiffusionModel",
                        "relative_path": "diffusion/model.safetensors",
                        "format": "safetensors",
                        "metadata": {
                            "component": "diffusion",
                            "backend": "burn",
                            "converted_layout": "burn_native_component_package",
                            "contract": "burn.component",
                            "contract_version": "1"
                        }
                    },
                    {
                        "component_role": "vae",
                        "model_role": "Vae",
                        "relative_path": "vae/model.safetensors",
                        "format": "safetensors",
                        "metadata": {
                            "component": "vae",
                            "backend": "burn",
                            "converted_layout": "burn_native_component_package",
                            "contract": "burn.component",
                            "contract_version": "1"
                        }
                    },
                    {
                        "component_role": "text_encoder",
                        "model_role": "TextEncoder",
                        "relative_path": "text_encoder/model.safetensors",
                        "format": "safetensors",
                        "metadata": {
                            "component": "text_encoder",
                            "backend": "burn",
                            "converted_layout": "burn_native_component_package",
                            "contract": "burn.component",
                            "contract_version": "1"
                        }
                    },
                    {
                        "component_role": "text_encoder_2",
                        "model_role": "TextEncoder",
                        "relative_path": "text_encoder_2/model.safetensors",
                        "format": "safetensors",
                        "metadata": {
                            "component": "text_encoder_2",
                            "backend": "burn",
                            "converted_layout": "burn_native_component_package",
                            "contract": "burn.component",
                            "contract_version": "1"
                        }
                    }
                ]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    report_path
}

#[tokio::test]
async fn workspace_host_exposes_burn_backend_instance_snapshot() {
    let workspace =
        WorkspaceHost::with_defaults(WorkspaceScope::new("ws-backend"), temp_dir("backend"));
    let snapshots = workspace.backend_instance_snapshots().await;

    // With no Burn worker installed, the runtime exposes no live backend
    // instance; the compute profile carries the available candidates.
    assert_eq!(
        snapshots.len(),
        0,
        "default composition without an installed worker should expose no live instance"
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

// ─── AR-08: project workflow persistence / migration / rollback ───

#[tokio::test]
async fn project_workflow_apply_persists_and_emits_changed_event() {
    let base = temp_dir("workflow-event-project");
    let host = WorkspaceHost::with_defaults(WorkspaceScope::new("workflow-event"), &base);
    let project_id = ProjectId::new("project-a");
    host.project_service()
        .create_project(
            project_id.clone(),
            reimagine_core::project::ProjectMetadata::new(
                "Project A",
                "workflow event test",
                Timestamp::new("2026-08-20T00:00:00Z"),
                Timestamp::new("2026-08-20T00:00:00Z"),
            ),
        )
        .await
        .expect("project creates");
    let service = host.workflow_service_for_project(&project_id);
    let workflow_id = service.register_workflow(Workflow::new("wf-event", DocumentVersion::new(0)));
    let mut events = host.document_events().subscribe();
    let result = service
        .apply_commands(
            &workflow_id,
            host.node_catalog().as_ref(),
            add_string_node_batch(DocumentVersion::new(0), "event"),
        )
        .await
        .expect("workflow apply");
    assert_eq!(result.status(), CommandStatus::Applied);
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), events.recv())
        .await
        .expect("workflow event arrives")
        .expect("event channel remains open");
    assert_eq!(event.kind, "workflow.changed");
    assert_eq!(event.project_id, "project-a");
    assert_eq!(event.document_id, "wf-event");
    assert_eq!(event.version, 1);
    assert!(
        base.join("projects/project-a/workflows/wf-event.json")
            .is_file()
    );
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn external_workflow_save_rejects_non_monotonic_version() {
    let paths = AppPaths::new(temp_dir("workflow-version-guard"));
    let service = WorkflowService::for_project(paths, ProjectId::new("version-project"));
    service
        .save_external_workflow(Workflow::new("wf-version", DocumentVersion::new(1)))
        .await
        .expect("first external save");
    let result = service
        .save_external_workflow(Workflow::new("wf-version", DocumentVersion::new(1)))
        .await;
    assert!(matches!(
        result,
        Err(reimagine_app_host::AppHostError::WorkflowVersionConflict { .. })
    ));
}
#[tokio::test]
async fn apply_commands_persists_accepted_workflow_and_survives_restart() {
    let paths = AppPaths::new(temp_dir("ar08-apply"));
    let service = WorkflowService::new(paths.clone());
    let catalog = BuiltinNodeCatalog::v1();
    let workflow = Workflow::new("wf-persist", DocumentVersion::new(0));
    let workflow_id = service.register_workflow(workflow);

    let batch = add_string_node_batch(DocumentVersion::new(0), "persisted");
    let result = service
        .apply_commands(&workflow_id, &catalog, batch)
        .await
        .expect("apply_commands should succeed");
    assert_eq!(result.status(), CommandStatus::Applied);

    // The accepted snapshot is on disk under the default project.
    let persisted = paths
        .project_workflows_dir("default")
        .join("wf-persist.json");
    assert!(persisted.is_file(), "accepted workflow must be persisted");

    // A fresh service discovers it (restart recovery).
    let reloaded = WorkflowService::new(paths);
    let loaded = reloaded
        .load_workflow(&workflow_id)
        .await
        .expect("workflow reloads after restart");
    assert_eq!(loaded, workflow_id);
    let snapshot = reloaded.snapshot(&workflow_id).unwrap();
    assert_eq!(snapshot.nodes().len(), 1);
    assert_eq!(snapshot.version(), DocumentVersion::new(1));
}

#[tokio::test]
async fn for_project_isolates_cross_project_workflow_paths() {
    let paths = AppPaths::new(temp_dir("ar08-projects"));
    let service_a = WorkflowService::for_project(paths.clone(), ProjectId::new("alpha"));
    let service_b = WorkflowService::for_project(paths.clone(), ProjectId::new("beta"));

    let workflow = Workflow::new("wf-cross", DocumentVersion::new(0));
    let id = service_a.register_workflow(workflow);
    let a_path = service_a.path_for_workflow_id(&id).unwrap();
    let b_path = service_b.path_for_workflow_id(&id).unwrap();
    assert!(
        a_path
            .to_string_lossy()
            .contains("projects/alpha/workflows")
    );
    assert!(b_path.to_string_lossy().contains("projects/beta/workflows"));
    assert_ne!(a_path, b_path);
}
