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
use reimagine_inference_candle::{
    SdxlCheckpointConversionManifest, SdxlCheckpointImportRequest, SdxlConvertedComponent,
};
use reimagine_model_manager::{
    Fingerprint, ModelDescriptor, ModelFormat, ModelManifest, ModelRoot, ModelSource,
    ModelSourceStatus,
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

#[tokio::test]
async fn model_service_import_reuses_existing_candle_split_conversion_and_updates_manifest() {
    let paths = AppPaths::new(temp_dir("model-import-existing"));
    let checkpoint_path = paths.models_dir().join("checkpoints/sdxl-base.safetensors");
    tokio::fs::create_dir_all(checkpoint_path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&checkpoint_path, b"checkpoint")
        .await
        .unwrap();

    let model_id = ModelId::new("sdxl-base-1.0");
    let fingerprint = Fingerprint::sha256("abc123");
    let request = SdxlCheckpointImportRequest::new(
        model_id.as_str(),
        &checkpoint_path,
        "sha256-abc123",
        "safetensors",
        paths.models_dir().join("converted"),
    )
    .with_created_at("2026-06-26T00:00:00Z");
    let conversion_dir = request.conversion_dir();
    let conversion_manifest = SdxlCheckpointConversionManifest::for_request(&request);
    for component in SdxlConvertedComponent::all() {
        let path = conversion_dir.join(conversion_manifest.component_path(component));
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(path, b"component").await.unwrap();
    }
    tokio::fs::write(
        conversion_dir.join("conversion.json"),
        serde_json::to_vec_pretty(&conversion_manifest).unwrap(),
    )
    .await
    .unwrap();

    let descriptor = ModelDescriptor::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(
            ModelRoot::base_models().id().clone(),
            "checkpoints/sdxl-base.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_fingerprint(fingerprint);
    let service = ModelService::new(paths.clone());
    service
        .save_manifest(
            &ModelManifest::new()
                .with_root(ModelRoot::base_models())
                .with_model(descriptor),
        )
        .await
        .unwrap();

    let (manifest, report, import_result) = service
        .import_sdxl_checkpoint_to_candle_split(&model_id)
        .await
        .unwrap();

    assert!(report.diagnostics().is_empty());
    assert!(import_result.reused_existing());
    let converted = manifest.models().first().expect("converted descriptor");
    assert_eq!(converted.components().len(), 4);
    assert!(converted.components().iter().any(|component| {
        component.metadata().get("component").map(String::as_str) == Some("unet")
    }));

    let model_ref = ModelRef::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        ModelRole::CheckpointBundle,
    );
    let resolved = service
        .resolve_descriptor_with_components(&model_ref)
        .await
        .unwrap();
    assert!(resolved.report().diagnostics().is_empty());
    assert_eq!(resolved.value().unwrap().components().len(), 4);
}

#[tokio::test]
async fn model_service_import_failure_does_not_mutate_manifest() {
    // Use an unsupported block index that passes projection but fails
    // at the writer/mapping stage.
    let paths = AppPaths::new(temp_dir("model-import-failure"));
    let checkpoint_path = paths.models_dir().join("checkpoints/sdxl-base.safetensors");
    tokio::fs::create_dir_all(checkpoint_path.parent().unwrap())
        .await
        .unwrap();
    let mut names = complete_original_checkpoint_names();
    names[0] = "model.diffusion_model.input_blocks.99.0.weight";
    write_tiny_safetensors(&checkpoint_path, &names);

    let model_id = ModelId::new("sdxl-base-1.0");
    let descriptor = ModelDescriptor::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(
            ModelRoot::base_models().id().clone(),
            "checkpoints/sdxl-base.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_fingerprint(Fingerprint::sha256("abc123"));
    let service = ModelService::new(paths);
    service
        .save_manifest(
            &ModelManifest::new()
                .with_root(ModelRoot::base_models())
                .with_model(descriptor),
        )
        .await
        .unwrap();

    let error = service
        .import_sdxl_checkpoint_to_candle_split(&model_id)
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("unsupported block index"),
        "expected unsupported block index error, got: {error}"
    );

    // Verify that the manifest was not mutated on failure.
    let models = service.list_models().await.unwrap();
    assert!(models[0].components().is_empty());
}

#[tokio::test]
async fn model_service_import_original_checkpoint_adds_split_components() {
    // Verify that a complete original checkpoint with valid mapping
    // succeeds and produces manifest components.
    let paths = AppPaths::new(temp_dir("model-import-success"));
    let checkpoint_path = paths.models_dir().join("checkpoints/sdxl-base.safetensors");
    tokio::fs::create_dir_all(checkpoint_path.parent().unwrap())
        .await
        .unwrap();
    write_tiny_safetensors(&checkpoint_path, &complete_original_checkpoint_names());

    let model_id = ModelId::new("sdxl-base-1.0");
    let descriptor = ModelDescriptor::new(
        model_id.clone(),
        ModelSeries::new("stable_diffusion"),
        ModelVariant::new("sdxl"),
        vec![ModelRole::CheckpointBundle],
        ModelSource::relative(
            ModelRoot::base_models().id().clone(),
            "checkpoints/sdxl-base.safetensors",
        ),
        ModelFormat::Safetensors,
    )
    .with_source_status(ModelSourceStatus::Available)
    .with_fingerprint(Fingerprint::sha256("abc123"));
    let service = ModelService::new(paths);
    service
        .save_manifest(
            &ModelManifest::new()
                .with_root(ModelRoot::base_models())
                .with_model(descriptor),
        )
        .await
        .unwrap();

    service
        .import_sdxl_checkpoint_to_candle_split(&model_id)
        .await
        .expect("original checkpoint import should now succeed");

    let models = service.list_models().await.unwrap();
    assert_eq!(models[0].components().len(), 4);
}

#[tokio::test]
async fn model_service_imports_burn_package_report_and_persists_manifest() {
    let paths = AppPaths::new(temp_dir("burn-package-import"));
    let report_path = write_burn_package(paths.models_dir(), "sdxl-base-1.0", "sha256-abc123");
    let service = ModelService::new(paths.clone());

    let (manifest, report, descriptor) = service
        .import_burn_converted_package(&report_path)
        .await
        .expect("burn package import should persist descriptor");

    assert!(report.diagnostics().is_empty());
    assert_eq!(descriptor.id().as_str(), "sdxl-base-1.0-burn");
    assert_eq!(manifest.models(), &[descriptor.clone()]);
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
    let service = ModelService::new(paths);
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
    let service = ModelService::new(paths);

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

fn write_tiny_safetensors(path: &std::path::Path, names: &[&str]) {
    let mut offset = 0usize;
    let entries = names
        .iter()
        .map(|name| {
            let start = offset;
            offset += 4;
            format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{start},{offset}]}}"
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let header = format!("{{{entries}}}");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for idx in 0..names.len() {
        bytes.extend_from_slice(&(idx as f32).to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
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
                        "component_role": "Diffusion",
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
                        "component_role": "Vae",
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
                        "component_role": "TextEncoder",
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
                        "component_role": "TextEncoder2",
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

fn complete_original_checkpoint_names() -> Vec<&'static str> {
    let mut names = vec![
        "model.diffusion_model.input_blocks.0.0.weight",
        "model.diffusion_model.time_embed.0.weight",
        "model.diffusion_model.input_blocks.1.0.in_layers.0.weight",
        "model.diffusion_model.input_blocks.4.1.proj_in.weight",
        "model.diffusion_model.input_blocks.3.0.op.weight",
        "model.diffusion_model.middle_block.0.in_layers.0.weight",
        "model.diffusion_model.middle_block.1.proj_in.weight",
        "model.diffusion_model.output_blocks.0.0.skip_connection.weight",
        "model.diffusion_model.output_blocks.0.1.proj_in.weight",
        "model.diffusion_model.output_blocks.2.2.conv.weight",
        "model.diffusion_model.out.0.weight",
        "model.diffusion_model.out.2.weight",
        "model.diffusion_model.label_emb.0.0.weight",
    ];
    names.extend(required_clip_source_names("conditioner.embedders.0."));
    names.extend(required_clip_source_names("conditioner.embedders.1.model."));
    names.extend(required_vae_source_names());
    names
}

fn required_clip_source_names(prefix: &'static str) -> Vec<&'static str> {
    let targets = [
        "transformer.text_model.embeddings.token_embedding.weight",
        "transformer.text_model.embeddings.position_embedding.weight",
        "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
        "transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
        "transformer.text_model.encoder.layers.0.self_attn.v_proj.weight",
        "transformer.text_model.encoder.layers.0.self_attn.out_proj.weight",
        "transformer.text_model.encoder.layers.0.layer_norm1.weight",
        "transformer.text_model.final_layer_norm.weight",
    ];
    targets
        .into_iter()
        .map(|target| Box::leak(format!("{prefix}{target}").into_boxed_str()) as &'static str)
        .collect()
}

fn required_vae_source_names() -> Vec<&'static str> {
    let targets = [
        "encoder.conv_in.weight",
        "encoder.conv_out.weight",
        "encoder.conv_norm_out.weight",
        "decoder.conv_in.weight",
        "decoder.conv_out.weight",
        "decoder.conv_norm_out.weight",
        "quant_conv.weight",
        "post_quant_conv.weight",
    ];
    targets
        .into_iter()
        .map(|target| {
            Box::leak(format!("first_stage_model.{target}").into_boxed_str()) as &'static str
        })
        .collect()
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
