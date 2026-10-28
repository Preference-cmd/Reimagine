use std::fs;
use std::sync::Arc;

use mod_sdk::ModRegistry;
use mod_sdk::schema::check_asset_marker_locations;
use mod_sdk::testing::SampleMod;
use mod_sdk::{ModManifest, OperationAnnotations, OperationSpec};
use reimagine_app_host::{AssetStore, InvokeRequest, WorkspaceHostBuilder, WorkspaceScope};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandProvenance, OperationContext,
};
use reimagine_core::event::EventBus;
use reimagine_core::model::{AssetId, ProjectId};
use reimagine_core::project::ProjectMetadata;
use reimagine_core::step::{StepOutcome, StepRecord};
use reimagine_core::task::CancelToken;
use serde_json::json;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-op-{prefix}-{nonce}"))
}

fn test_context() -> OperationContext {
    OperationContext::new(
        CommandActor::new(CommandActorKind::Human).with_id("test-user"),
        CommandProvenance::Direct,
    )
}

#[tokio::test]
async fn op_invoke_with_invalid_input_returns_diagnostic_and_calls_nothing() {
    let base = temp_dir("invalid-input");
    let event_bus = EventBus::new();
    let mod_registry =
        Arc::new(ModRegistry::start(vec![Arc::new(SampleMod::new())], &event_bus).await);

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .event_bus(event_bus)
        .mod_registry(mod_registry)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-invalid");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    // sample.echo requires { "text": string }
    let req = InvokeRequest {
        name: "sample.echo".to_string(),
        input: json!({ "wrong_field": 123 }),
        context: test_context(),
        cancel: CancelToken::new(),
    };

    let err = host
        .operation_service()
        .invoke(&project_id, req)
        .await
        .expect_err("expected validation failure");

    assert_eq!(err.code().as_str(), "SAMPLE/CALLER_INVALID_INPUT");

    // Verify no step record was written to disk
    let project_dir = host.config().paths().project_dir(project_id.as_str());
    let steps_path = project_dir.join("steps.jsonl");
    assert!(
        !steps_path.exists() || fs::read_to_string(&steps_path).unwrap().trim().is_empty(),
        "invalid input must not write step records"
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn op_invoke_with_malformed_asset_id_returns_caller_diagnostic() {
    let base = temp_dir("bad-asset-id");
    let event_bus = EventBus::new();
    let mod_registry =
        Arc::new(ModRegistry::start(vec![Arc::new(SampleMod::new())], &event_bus).await);

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .event_bus(event_bus)
        .mod_registry(mod_registry)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-bad-asset-id");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    for bad_id in ["../escape", "a\\b", ""] {
        let req = InvokeRequest {
            name: "sample.transform".to_string(),
            input: json!({ "input_file": bad_id }),
            context: test_context(),
            cancel: CancelToken::new(),
        };
        let err = host
            .operation_service()
            .invoke(&project_id, req)
            .await
            .expect_err("a malformed asset id must be rejected, not panic");
        assert_eq!(err.code().as_str(), "SAMPLE/CALLER_INVALID_INPUT");
    }

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn valid_call_writes_step_record_with_actor_and_output_assets_hold_ids() {
    let base = temp_dir("valid-call");
    let event_bus = EventBus::new();
    let sample = Arc::new(SampleMod::new());
    let mod_registry = Arc::new(ModRegistry::start(vec![sample], &event_bus).await);

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .event_bus(event_bus)
        .mod_registry(mod_registry)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-valid");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    // First import a file so we have an input asset
    let input_file = base.join("input.txt");
    fs::write(&input_file, "hello from input asset").unwrap();

    let import_req = InvokeRequest {
        name: "asset.import".to_string(),
        input: json!({ "paths": [input_file.to_str().unwrap()] }),
        context: OperationContext::new(
            CommandActor::new(CommandActorKind::Human).with_id("user-42"),
            CommandProvenance::Direct,
        ),
        cancel: CancelToken::new(),
    };

    let import_result = host
        .operation_service()
        .invoke(&project_id, import_req)
        .await
        .expect("asset.import succeeds");

    let imported_asset_id = import_result.output["assets"][0]
        .as_str()
        .expect("asset id string")
        .to_string();

    // Now call sample.transform with the imported asset id
    let transform_req = InvokeRequest {
        name: "sample.transform".to_string(),
        input: json!({ "input_file": imported_asset_id }),
        context: OperationContext::new(
            CommandActor::new(CommandActorKind::Human).with_id("operator-1"),
            CommandProvenance::Direct,
        ),
        cancel: CancelToken::new(),
    };

    let transform_result = host
        .operation_service()
        .invoke(&project_id, transform_req)
        .await
        .expect("sample.transform succeeds");

    // Verify output asset field holds an AssetId string, NOT a relative path
    let output_asset_id = transform_result.output["output_file"]
        .as_str()
        .expect("asset id");
    assert!(
        output_asset_id.starts_with("asset-"),
        "output asset field must hold asset id: {output_asset_id}"
    );

    // Verify step record has actor and succeeded outcome
    assert_eq!(
        transform_result.step.context.actor().id(),
        Some("operator-1")
    );
    assert_eq!(
        transform_result.step.context.actor().kind(),
        CommandActorKind::Human
    );
    assert!(matches!(
        transform_result.step.outcome,
        StepOutcome::Succeeded { .. }
    ));

    // Verify the step log persisted the step record
    let project_dir = host.config().paths().project_dir(project_id.as_str());
    let steps_file = project_dir.join("steps.jsonl");
    let steps_content = fs::read_to_string(&steps_file).expect("read steps.jsonl");
    let step_lines: Vec<serde_json::Value> = steps_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse step record"))
        .collect();
    // asset.import has start + finish (2 lines), sample.transform has start + finish (2 lines) -> total 4 lines
    assert_eq!(step_lines.len(), 4);
    let last_step: StepRecord = serde_json::from_value(step_lines[3].clone()).unwrap();
    assert_eq!(last_step.id, transform_result.step.id);
    assert_eq!(last_step.context.actor().id(), Some("operator-1"));
    assert!(matches!(last_step.outcome, StepOutcome::Succeeded { .. }));

    // Verify asset store contains the newly generated asset
    let asset_store = AssetStore::new(&project_dir);
    let asset_record = asset_store
        .get_asset(&AssetId::new(output_asset_id))
        .await
        .expect("get asset")
        .expect("asset exists");
    assert_eq!(asset_record.origin, transform_result.step.id);

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn asset_import_origin_and_content_addressable_dedup() {
    let base = temp_dir("asset-import-dedup");
    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-import");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    let test_file = base.join("same_content.txt");
    fs::write(&test_file, "identical content 1234567890").unwrap();

    // First import
    let req1 = InvokeRequest {
        name: "asset.import".to_string(),
        input: json!({ "paths": [test_file.to_str().unwrap()] }),
        context: test_context(),
        cancel: CancelToken::new(),
    };
    let res1 = host
        .operation_service()
        .invoke(&project_id, req1)
        .await
        .expect("import 1 succeeds");
    let asset_id_1 = res1.output["assets"][0].as_str().unwrap();

    // Second import of the same file
    let req2 = InvokeRequest {
        name: "asset.import".to_string(),
        input: json!({ "paths": [test_file.to_str().unwrap()] }),
        context: test_context(),
        cancel: CancelToken::new(),
    };
    let res2 = host
        .operation_service()
        .invoke(&project_id, req2)
        .await
        .expect("import 2 succeeds");
    let asset_id_2 = res2.output["assets"][0].as_str().unwrap();

    // The asset IDs are distinct
    assert_ne!(asset_id_1, asset_id_2);

    let project_dir = host.config().paths().project_dir(project_id.as_str());
    let asset_store = AssetStore::new(&project_dir);

    let rec1 = asset_store
        .get_asset(&AssetId::new(asset_id_1))
        .await
        .unwrap()
        .unwrap();
    let rec2 = asset_store
        .get_asset(&AssetId::new(asset_id_2))
        .await
        .unwrap()
        .unwrap();

    // Check origins: the asset's origin is its respective import step
    assert_eq!(rec1.origin, res1.step.id);
    assert_eq!(rec2.origin, res2.step.id);

    // Hashes match because content is identical
    assert_eq!(rec1.hash, rec2.hash);

    // There should be exactly two lines in assets.jsonl
    let jsonl_content = fs::read_to_string(project_dir.join("assets.jsonl")).unwrap();
    assert_eq!(jsonl_content.lines().count(), 2);

    // But only one stored file on disk in assets/<prefix>/<hash>
    let content_file = asset_store.content_path(&rec1.hash);
    assert!(content_file.is_file());

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn sample_mod_output_escaping_or_symlink_fails() {
    let base = temp_dir("mod-security");
    let event_bus = EventBus::new();
    let mod_registry =
        Arc::new(ModRegistry::start(vec![Arc::new(SampleMod::new())], &event_bus).await);

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .event_bus(event_bus)
        .mod_registry(mod_registry)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-sec");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    // Import a source file
    let input_file = base.join("sample.txt");
    fs::write(&input_file, "data").unwrap();
    let import_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "asset.import".to_string(),
                input: json!({ "paths": [input_file.to_str().unwrap()] }),
                context: test_context(),
                cancel: CancelToken::new(),
            },
        )
        .await
        .unwrap();
    let asset_id = import_res.output["assets"][0].as_str().unwrap();

    // 1. Escaping output path test
    let escape_req = InvokeRequest {
        name: "sample.transform".to_string(),
        input: json!({ "input_file": asset_id, "escape": true }),
        context: test_context(),
        cancel: CancelToken::new(),
    };
    let escape_err = host
        .operation_service()
        .invoke(&project_id, escape_req)
        .await
        .expect_err("escape must fail");
    assert_eq!(
        escape_err.code().as_str(),
        "KERNEL/CALLER_ESCAPE_OUTPUT_DIR"
    );

    // 2. Symlink rejection test
    let symlink_req = InvokeRequest {
        name: "sample.transform".to_string(),
        input: json!({ "input_file": asset_id, "symlink": true }),
        context: test_context(),
        cancel: CancelToken::new(),
    };
    let symlink_err = host
        .operation_service()
        .invoke(&project_id, symlink_req)
        .await
        .expect_err("symlink must fail");
    assert_eq!(
        symlink_err.code().as_str(),
        "KERNEL/CALLER_SYMLINK_REJECTED"
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn sample_mod_output_with_one_good_and_one_escaping_path_finishes_failed_and_cleans_scratch()
{
    let base = temp_dir("mod-partial-escape");
    let event_bus = EventBus::new();
    let mod_registry =
        Arc::new(ModRegistry::start(vec![Arc::new(SampleMod::new())], &event_bus).await);

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .event_bus(event_bus)
        .mod_registry(mod_registry)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-partial-escape");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    // Import a source file
    let input_file = base.join("sample.txt");
    fs::write(&input_file, "data").unwrap();
    let import_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "asset.import".to_string(),
                input: json!({ "paths": [input_file.to_str().unwrap()] }),
                context: test_context(),
                cancel: CancelToken::new(),
            },
        )
        .await
        .unwrap();
    let asset_id = import_res.output["assets"][0].as_str().unwrap();

    let project_dir = host.config().paths().project_dir(project_id.as_str());

    let initial_assets = fs::read_to_string(project_dir.join("assets.jsonl"))
        .unwrap()
        .lines()
        .count();
    assert_eq!(initial_assets, 1);

    // Call sample.transform with escape_second: true (generates transformed.txt AND ../escaped.txt)
    let bad_req = InvokeRequest {
        name: "sample.transform".to_string(),
        input: json!({ "input_file": asset_id, "escape_second": true }),
        context: test_context(),
        cancel: CancelToken::new(),
    };
    let err = host
        .operation_service()
        .invoke(&project_id, bad_req)
        .await
        .expect_err("must fail due to escaping path");
    assert_eq!(err.code().as_str(), "KERNEL/CALLER_ESCAPE_OUTPUT_DIR");

    // Verify step record finished as Failed
    let steps_content = fs::read_to_string(project_dir.join("steps.jsonl")).unwrap();
    let step_lines: Vec<serde_json::Value> = steps_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse step line"))
        .collect();
    assert_eq!(step_lines.len(), 4);
    let failed_step: StepRecord = serde_json::from_value(step_lines[3].clone()).unwrap();
    assert!(matches!(failed_step.outcome, StepOutcome::Failed { .. }));
    let failed_id = failed_step.id;

    // Verify no new asset was ingested into assets.jsonl
    let assets_after = fs::read_to_string(project_dir.join("assets.jsonl"))
        .unwrap()
        .lines()
        .count();
    assert_eq!(
        assets_after, initial_assets,
        "no asset must be registered when a subsequent path is bad"
    );

    // Verify scratch directory for failed step is removed
    let scratch_dir = project_dir.join("scratch").join(failed_id.as_str());
    assert!(
        !scratch_dir.exists(),
        "scratch directory must be cleaned up on failure"
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn preview_key_path_traversal_and_invalid_keys_rejected() {
    let base = temp_dir("preview-key-validation");
    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-key-test");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    let bad_keys = [
        "../escape",
        "/tmp/escape",
        "key with spaces",
        "key/slash",
        "",
        "key@special!",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", // 65 chars
    ];

    for bad_key in bad_keys {
        let err = host
            .operation_service()
            .preview(
                &project_id,
                "asset.import",
                json!({ "paths": [] }),
                bad_key,
                CancelToken::new(),
            )
            .await
            .expect_err(&format!("key `{bad_key}` must be rejected"));

        assert_eq!(
            err.code().as_str(),
            "KERNEL/CALLER_INVALID_PREVIEW_KEY",
            "bad key `{bad_key}` failed with unexpected code"
        );

        let resolve_err = host
            .operation_service()
            .resolve_preview_file(bad_key, "file.txt");
        assert!(
            resolve_err.is_err(),
            "resolve_preview_file must reject `{bad_key}`"
        );
    }

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn preview_writes_no_records_refuses_open_world_and_handles_replace_race() {
    let base = temp_dir("preview-tests");
    let event_bus = EventBus::new();
    let mod_registry =
        Arc::new(ModRegistry::start(vec![Arc::new(SampleMod::new())], &event_bus).await);

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .event_bus(event_bus)
        .mod_registry(mod_registry)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-preview");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    // 1. Import a file to get an asset id
    let input_file = base.join("prev_in.txt");
    fs::write(&input_file, "preview source content").unwrap();
    let import_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "asset.import".to_string(),
                input: json!({ "paths": [input_file.to_str().unwrap()] }),
                context: test_context(),
                cancel: CancelToken::new(),
            },
        )
        .await
        .unwrap();
    let asset_id = import_res.output["assets"][0].as_str().unwrap();

    let project_dir = host.config().paths().project_dir(project_id.as_str());
    let count_steps = || {
        let p = project_dir.join("steps.jsonl");
        if !p.exists() {
            0
        } else {
            fs::read_to_string(&p)
                .unwrap()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count()
        }
    };
    let initial_steps = count_steps();
    assert_eq!(initial_steps, 2);

    // 2. Run preview
    let preview_key = "test-preview-key";
    let preview_out = host
        .operation_service()
        .preview(
            &project_id,
            "sample.transform",
            json!({ "input_file": asset_id }),
            preview_key,
            CancelToken::new(),
        )
        .await
        .expect("preview succeeds");

    // Output asset field holds preview URI
    let uri = preview_out["output_file"].as_str().unwrap();
    assert_eq!(
        uri,
        format!("reimagine-asset://preview/{preview_key}/transformed.txt")
    );

    // Verify preview wrote NO step record
    let steps_after_preview = count_steps();
    assert_eq!(steps_after_preview, initial_steps);

    // Verify preview wrote NO new asset record in assets.jsonl
    let jsonl_lines = fs::read_to_string(project_dir.join("assets.jsonl"))
        .unwrap()
        .lines()
        .count();
    assert_eq!(jsonl_lines, 1); // Only the initial asset.import

    // 3. Refuse open_world op test
    struct OpenWorldOp {
        spec: OperationSpec,
    }
    #[async_trait::async_trait]
    impl reimagine_app_host::KernelOp for OpenWorldOp {
        fn spec(&self) -> &OperationSpec {
            &self.spec
        }
        async fn invoke(
            &self,
            _call: reimagine_app_host::KernelCall,
        ) -> Result<serde_json::Value, reimagine_core::diagnostic::Diagnostic> {
            Ok(json!({}))
        }
    }
    let open_op = Arc::new(OpenWorldOp {
        spec: OperationSpec {
            name: "test.open_world".to_string(),
            description: "open world op".to_string(),
            input_schema: json!({ "type": "object", "properties": {} }),
            output_schema: json!({ "type": "object", "properties": {} }),
            annotations: OperationAnnotations {
                read_only: false,
                destructive: false,
                idempotent: false,
                deterministic: false,
                open_world: true,
            },
        },
    });
    host.operation_service().register_kernel_op(open_op);

    let open_world_err = host
        .operation_service()
        .preview(
            &project_id,
            "test.open_world",
            json!({}),
            "key-open",
            CancelToken::new(),
        )
        .await
        .expect_err("open_world preview must be refused");
    assert_eq!(
        open_world_err.code().as_str(),
        "TEST/CALLER_OPEN_WORLD_PREVIEW_REFUSED"
    );

    // 4. Repeated key cancels earlier preview, and replacement files survive exit of earlier preview
    struct RacePreviewOp {
        spec: OperationSpec,
        started: Arc<tokio::sync::Notify>,
    }
    #[async_trait::async_trait]
    impl reimagine_app_host::KernelOp for RacePreviewOp {
        fn spec(&self) -> &OperationSpec {
            &self.spec
        }
        async fn invoke(
            &self,
            call: reimagine_app_host::KernelCall,
        ) -> Result<serde_json::Value, reimagine_core::diagnostic::Diagnostic> {
            let is_slow = call
                .invocation
                .input
                .get("slow")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);

            if is_slow {
                // Write a file from the first preview
                let first_file = call.invocation.output_dir.join("first.txt");
                let _ = tokio::fs::write(&first_file, "first preview content").await;
                self.started.notify_one();
                // Wait for cancellation token to fire
                call.invocation.cancel.cancelled().await;
                Err(reimagine_core::diagnostic::Diagnostic::new(
                    reimagine_core::model::DiagnosticId::new("op:test.race:CANCELLED"),
                    reimagine_core::diagnostic::DiagnosticCode::new("TEST/CALLER_CANCELLED"),
                    reimagine_core::diagnostic::DiagnosticSeverity::Error,
                    reimagine_core::diagnostic::DiagnosticSourceName::new("test"),
                    "cancelled",
                    reimagine_core::diagnostic::DiagnosticTarget::new(
                        reimagine_core::diagnostic::DiagnosticTargetDomain::new("op.test.race"),
                    ),
                ))
            } else {
                let second_file = call.invocation.output_dir.join("second.txt");
                tokio::fs::write(&second_file, "second preview content")
                    .await
                    .unwrap();
                Ok(json!({ "output_file": "second.txt" }))
            }
        }
    }

    let started = Arc::new(tokio::sync::Notify::new());
    let race_op = Arc::new(RacePreviewOp {
        spec: OperationSpec {
            name: "test.race".to_string(),
            description: "race test op".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": { "slow": { "type": "boolean" } }
            }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "output_file": { "type": "string", "format": "asset" }
                }
            }),
            annotations: OperationAnnotations {
                read_only: false,
                destructive: false,
                idempotent: false,
                deterministic: true,
                open_world: false,
            },
        },
        started: started.clone(),
    });
    host.operation_service().register_kernel_op(race_op);

    let op_svc = host.operation_service().clone();
    let proj = project_id.clone();
    let token1 = CancelToken::new();

    // Launch first preview in flight
    let h1 = tokio::spawn(async move {
        op_svc
            .preview(
                &proj,
                "test.race",
                json!({ "slow": true }),
                "shared-key",
                token1,
            )
            .await
    });

    // Wait until preview 1 is genuinely in flight and waiting on cancel
    started.notified().await;

    // Launch replacement preview with the same key
    let token2 = CancelToken::new();
    let res2 = host
        .operation_service()
        .preview(
            &project_id,
            "test.race",
            json!({ "slow": false }),
            "shared-key",
            token2,
        )
        .await
        .expect("second preview succeeds");

    assert_eq!(
        res2["output_file"].as_str().unwrap(),
        "reimagine-asset://preview/shared-key/second.txt"
    );

    // Await first preview: it must end cancelled
    let res1 = h1.await.unwrap();
    assert!(
        res1.is_err(),
        "first preview must end with cancellation/error"
    );

    // Verify second's files survived the first's exit!
    let preview_dir = host
        .operation_service()
        .preview_dir("shared-key")
        .expect("valid key");
    assert!(
        preview_dir.join("second.txt").is_file(),
        "second preview file must survive first preview's exit"
    );
    assert!(
        !preview_dir.join("first.txt").exists(),
        "first preview file must have been cleared by replacement preview"
    );

    let _ = fs::remove_dir_all(&base);
}

#[cfg(unix)]
#[tokio::test]
async fn symlinked_parent_directory_escapes_output_dir_rejected() {
    let base = temp_dir("symlink-parent");
    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-op"), &base)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-sym-parent");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "P",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    // Outside directory that the symlink points to
    let outside_dir = base.join("outside");
    fs::create_dir_all(&outside_dir).unwrap();
    let outside_file = outside_dir.join("secret.txt");
    fs::write(&outside_file, "secret data").unwrap();

    let outside_dir_for_op = outside_dir.clone();

    struct SymlinkParentOp {
        spec: OperationSpec,
        outside_dir: std::path::PathBuf,
    }

    #[async_trait::async_trait]
    impl reimagine_app_host::KernelOp for SymlinkParentOp {
        fn spec(&self) -> &OperationSpec {
            &self.spec
        }
        async fn invoke(
            &self,
            call: reimagine_app_host::KernelCall,
        ) -> Result<serde_json::Value, reimagine_core::diagnostic::Diagnostic> {
            let sub_link = call.invocation.output_dir.join("sub");
            std::os::unix::fs::symlink(&self.outside_dir, &sub_link).unwrap();
            Ok(json!({ "output_file": "sub/secret.txt" }))
        }
    }

    let op = Arc::new(SymlinkParentOp {
        spec: OperationSpec {
            name: "test.symlink_parent".to_string(),
            description: "symlink parent op".to_string(),
            input_schema: json!({ "type": "object", "properties": {} }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "output_file": { "type": "string", "format": "asset" }
                }
            }),
            annotations: OperationAnnotations {
                read_only: false,
                destructive: false,
                idempotent: false,
                deterministic: true,
                open_world: false,
            },
        },
        outside_dir: outside_dir_for_op,
    });
    host.operation_service().register_kernel_op(op);

    let req = InvokeRequest {
        name: "test.symlink_parent".to_string(),
        input: json!({}),
        context: test_context(),
        cancel: CancelToken::new(),
    };

    let err = host
        .operation_service()
        .invoke(&project_id, req)
        .await
        .expect_err("symlinked parent directory must be rejected");

    assert_eq!(err.code().as_str(), "KERNEL/CALLER_SYMLINK_REJECTED");

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn manifest_check_rejects_asset_marker_under_one_of() {
    let schema_with_one_of = json!({
        "type": "object",
        "properties": {
            "choice": {
                "oneOf": [
                    { "type": "string", "format": "asset" },
                    { "type": "number" }
                ]
            }
        }
    });

    let manifest = ModManifest {
        id: "testmod".to_string(),
        name: "Test Mod".to_string(),
        version: "0.1.0".to_string(),
        api: "0.1.0".to_string(),
        operations: vec![OperationSpec {
            name: "testmod.choice".to_string(),
            description: "test".to_string(),
            input_schema: schema_with_one_of,
            output_schema: json!({ "type": "object" }),
            annotations: OperationAnnotations {
                read_only: false,
                destructive: false,
                idempotent: false,
                deterministic: true,
                open_world: false,
            },
        }],
        events: vec![],
    };

    let result = check_asset_marker_locations(&manifest.operations[0].input_schema);
    assert!(result.is_err(), "asset marker under oneOf must be rejected");
    let msg = result.unwrap_err();
    assert!(msg.contains("asset marker"));
}
