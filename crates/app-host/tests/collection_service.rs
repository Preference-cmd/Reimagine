use std::fs;
use std::sync::Arc;

use async_trait::async_trait;
use mod_sdk::{OperationAnnotations, OperationSpec};
use reimagine_app_host::{
    InvokeRequest, KernelCall, KernelOp, WorkspaceHostBuilder, WorkspaceScope,
};
use reimagine_core::board::BoardItemKind;
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandProvenance, OperationContext,
};
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::{Event, EventBus};
use reimagine_core::model::{CollectionId, ProjectId};
use reimagine_core::project::ProjectMetadata;
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-col-{prefix}-{nonce}"))
}

fn test_context(actor_id: &str) -> OperationContext {
    OperationContext::new(
        CommandActor::new(CommandActorKind::Human).with_id(actor_id),
        CommandProvenance::Direct,
    )
}

struct ReadOnlyWithAssetOp {
    spec: OperationSpec,
}

impl ReadOnlyWithAssetOp {
    fn new() -> Self {
        Self {
            spec: OperationSpec {
                name: "test.read_only_asset".to_owned(),
                description: "Read only op that produces an asset".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false,
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "artifact": {
                            "type": "string",
                            "format": "asset"
                        }
                    },
                    "required": ["artifact"],
                    "additionalProperties": false,
                }),
                annotations: OperationAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    deterministic: true,
                    open_world: false,
                },
            },
        }
    }
}

#[async_trait]
impl KernelOp for ReadOnlyWithAssetOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let out_path = call.invocation.output_dir.join("test_out.txt");
        tokio::fs::write(&out_path, b"test content")
            .await
            .expect("write test file");
        Ok(json!({ "artifact": "test_out.txt" }))
    }
}

#[tokio::test]
async fn collection_create_after_import_creates_slots_board_item_and_persists() {
    let base = temp_dir("col-create");
    let event_bus = EventBus::new();
    let mut rx = event_bus.subscribe();

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-col"), &base)
        .event_bus(event_bus.clone())
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-col");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "Col Project",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    // 1. Create 3 temporary source files to import
    let src1 = base.join("f1.txt");
    let src2 = base.join("f2.txt");
    let src3 = base.join("f3.txt");
    fs::write(&src1, "file 1 content").expect("write src1");
    fs::write(&src2, "file 2 content").expect("write src2");
    fs::write(&src3, "file 3 content").expect("write src3");

    // Import the 3 files via asset.import
    let req = InvokeRequest {
        name: "asset.import".to_string(),
        input: json!({
            "paths": [
                src1.to_string_lossy().to_string(),
                src2.to_string_lossy().to_string(),
                src3.to_string_lossy().to_string(),
            ],
        }),
        context: test_context("importer"),
        cancel: CancelToken::new(),
    };
    let res = host
        .operation_service()
        .invoke(&project_id, req)
        .await
        .expect("import files");
    let asset_ids: Vec<String> = res
        .output
        .get("assets")
        .and_then(Value::as_array)
        .expect("assets array")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(asset_ids.len(), 3);

    // Drain existing events from project creation and import
    while rx.try_recv().is_ok() {}

    // 2. Call collection.create
    let create_req = InvokeRequest {
        name: "collection.create".to_string(),
        input: json!({
            "name": "My 3 Files",
            "assets": asset_ids,
        }),
        context: test_context("creator"),
        cancel: CancelToken::new(),
    };

    let create_res = host
        .operation_service()
        .invoke(&project_id, create_req)
        .await
        .expect("create collection");

    let collection_id_str = create_res
        .output
        .get("collection_id")
        .and_then(Value::as_str)
        .expect("collection_id");
    let board_item_id_str = create_res
        .output
        .get("board_item_id")
        .and_then(Value::as_str)
        .expect("board_item_id");

    let collection_id = CollectionId::new(collection_id_str);

    // Check collection.changed event was published
    let mut received_changed = false;
    while let Ok(event) = rx.try_recv() {
        if let Event::Document {
            kind,
            document_id,
            project_id: p_id,
            ..
        } = event
            && kind == "collection.changed"
            && document_id == collection_id_str
            && p_id == project_id.as_str()
        {
            received_changed = true;
            break;
        }
    }
    assert!(received_changed, "collection.changed event must be emitted");

    // Check collection document snapshot
    let get_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.get".to_string(),
                input: json!({ "collection_id": collection_id.as_str() }),
                context: test_context("checker"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("get collection");

    assert_eq!(get_res.output["name"].as_str().unwrap(), "My 3 Files");
    let slots = get_res.output["slots"].as_array().unwrap();
    assert_eq!(slots.len(), 3);
    assert_eq!(slots[0]["id"].as_str().unwrap(), "slot-1");
    assert_eq!(slots[0]["asset_id"].as_str().unwrap(), &asset_ids[0]);
    assert_eq!(slots[1]["id"].as_str().unwrap(), "slot-2");
    assert_eq!(slots[1]["asset_id"].as_str().unwrap(), &asset_ids[1]);
    assert_eq!(slots[2]["id"].as_str().unwrap(), "slot-3");
    assert_eq!(slots[2]["asset_id"].as_str().unwrap(), &asset_ids[2]);

    // Check Board item was created
    let board = host
        .board_service()
        .snapshot(&project_id)
        .await
        .expect("snapshot board");
    let (_, board_item) = board
        .items()
        .iter()
        .find(|(id, _)| id.as_str() == board_item_id_str)
        .expect("board item found");
    match board_item.kind() {
        BoardItemKind::Collection {
            collection_id: item_col_id,
        } => {
            assert_eq!(*item_col_id, collection_id);
        }
        other => panic!("expected BoardItemKind::Collection, got {other:?}"),
    }

    // Verify file persisted and reloadable by a fresh CollectionService
    let fresh_col_service = reimagine_app_host::CollectionService::with_event_bus(
        host.config().paths().clone(),
        host.event_bus().clone(),
    );
    let reloaded = fresh_col_service
        .snapshot(&project_id, &collection_id)
        .await
        .expect("reload from fresh service");
    assert_eq!(reloaded.id(), &collection_id);
    assert_eq!(reloaded.slots().len(), 3);
    assert_eq!(reloaded.slots()[0].asset.as_str(), &asset_ids[0]);

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn collection_create_with_unknown_asset_id_fails_and_creates_nothing() {
    let base = temp_dir("col-unknown-asset");
    let event_bus = EventBus::new();

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-col"), &base)
        .event_bus(event_bus)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-unknown");
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

    let create_req = InvokeRequest {
        name: "collection.create".to_string(),
        input: json!({
            "name": "Invalid Collection",
            "assets": ["asset-does-not-exist"],
        }),
        context: test_context("creator"),
        cancel: CancelToken::new(),
    };

    let err = host
        .operation_service()
        .invoke(&project_id, create_req)
        .await
        .expect_err("unknown asset must fail");

    assert_eq!(err.code().as_str(), "COLLECTION/CALLER_ASSET_NOT_FOUND");

    // Verify nothing created in collections dir
    let col_dir = host
        .config()
        .paths()
        .project_dir(project_id.as_str())
        .join("collections");
    assert!(
        !col_dir.exists() || fs::read_dir(&col_dir).unwrap().next().is_none(),
        "no collections file should be created"
    );

    // Verify board has no collection item
    let board = host
        .board_service()
        .snapshot(&project_id)
        .await
        .expect("board snapshot");
    assert!(board.items().is_empty());

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn collection_set_slot_undo_redo_via_invoke_flips_and_records_step() {
    let base = temp_dir("col-set-undo-redo");
    let event_bus = EventBus::new();

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-col"), &base)
        .event_bus(event_bus)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-undo-redo");
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

    // Import 2 assets
    let f1 = base.join("f1.txt");
    let f2 = base.join("f2.txt");
    fs::write(&f1, "f1").unwrap();
    fs::write(&f2, "f2").unwrap();

    let res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "asset.import".to_string(),
                input: json!({
                    "paths": [
                        f1.to_string_lossy().to_string(),
                        f2.to_string_lossy().to_string(),
                    ],
                }),
                context: test_context("importer"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("import");
    let asset_ids: Vec<String> = res
        .output
        .get("assets")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    // Create collection with asset_ids[0]
    let create_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.create".to_string(),
                input: json!({
                    "name": "One Slot Collection",
                    "assets": [&asset_ids[0]],
                }),
                context: test_context("actor-create"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("create collection");

    let col_id = create_res
        .output
        .get("collection_id")
        .and_then(Value::as_str)
        .unwrap()
        .to_string();

    // 1. set_slot to asset_ids[1]
    let set_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.set_slot".to_string(),
                input: json!({
                    "collection_id": col_id,
                    "slot_id": "slot-1",
                    "asset": asset_ids[1],
                }),
                context: test_context("actor-set"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("set_slot");

    assert_eq!(set_res.step.context.actor().id(), Some("actor-set"));

    let get_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.get".to_string(),
                input: json!({ "collection_id": col_id }),
                context: test_context("checker"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        get_res.output["slots"][0]["asset_id"].as_str().unwrap(),
        &asset_ids[1]
    );

    // 2. undo
    let undo_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.undo".to_string(),
                input: json!({
                    "collection_id": col_id,
                }),
                context: test_context("actor-undo"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("undo");

    assert_eq!(
        undo_res.output.get("applied").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(undo_res.step.context.actor().id(), Some("actor-undo"));

    let get_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.get".to_string(),
                input: json!({ "collection_id": col_id }),
                context: test_context("checker"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        get_res.output["slots"][0]["asset_id"].as_str().unwrap(),
        &asset_ids[0]
    );

    // 3. redo
    let redo_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.redo".to_string(),
                input: json!({
                    "collection_id": col_id,
                }),
                context: test_context("actor-redo"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("redo");

    assert_eq!(
        redo_res.output.get("applied").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(redo_res.step.context.actor().id(), Some("actor-redo"));

    let get_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.get".to_string(),
                input: json!({ "collection_id": col_id }),
                context: test_context("checker"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        get_res.output["slots"][0]["asset_id"].as_str().unwrap(),
        &asset_ids[1]
    );

    // Verify step log has records for set_slot, undo, and redo with their actors
    let steps_file = host
        .config()
        .paths()
        .project_dir(project_id.as_str())
        .join("steps.jsonl");
    let content = fs::read_to_string(steps_file).expect("read steps.jsonl");
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    // Each op wrote start and finish lines
    assert!(
        lines
            .iter()
            .any(|l| l.contains("collection.set_slot") && l.contains("actor-set"))
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("collection.undo") && l.contains("actor-undo"))
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("collection.redo") && l.contains("actor-redo"))
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn collection_get_writes_no_step_record_while_read_only_with_asset_writes_one() {
    let base = temp_dir("col-get-step-record");
    let event_bus = EventBus::new();

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-col"), &base)
        .event_bus(event_bus)
        .build()
        .await
        .expect("build host");

    // Register our read-only op that outputs an asset
    let op = Arc::new(ReadOnlyWithAssetOp::new());
    host.operation_service().register_kernel_op(op);

    let project_id = ProjectId::new("proj-step-record");
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

    // Create a collection directly via service
    let col_service = reimagine_app_host::CollectionService::with_event_bus(
        host.config().paths().clone(),
        host.event_bus().clone(),
    );
    let col_id = CollectionId::new("col-test");
    col_service
        .create(
            &project_id,
            &col_id,
            "Direct Collection",
            vec![reimagine_core::collection::Slot::new("slot-1", "asset-1")],
        )
        .await
        .expect("create collection directly");

    let steps_file = host
        .config()
        .paths()
        .project_dir(project_id.as_str())
        .join("steps.jsonl");
    assert!(!steps_file.exists(), "no steps recorded yet");

    // 1. Invoke collection.get (read_only, no asset field in output schema)
    let get_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.get".to_string(),
                input: json!({
                    "collection_id": col_id.as_str(),
                }),
                context: test_context("getter"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("collection.get");

    assert_eq!(
        get_res.output.get("name").and_then(Value::as_str),
        Some("Direct Collection")
    );
    assert_eq!(
        get_res
            .output
            .get("slots")
            .and_then(Value::as_array)
            .map(|a| a.len()),
        Some(1)
    );

    // Assert that collection.get did NOT write to steps.jsonl
    assert!(
        !steps_file.exists() || fs::read_to_string(&steps_file).unwrap().trim().is_empty(),
        "collection.get must not write step records"
    );

    // 2. Invoke test.read_only_asset (read_only, but output schema HAS asset field)
    let asset_op_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "test.read_only_asset".to_string(),
                input: json!({}),
                context: test_context("asset-generator"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("read_only_asset op");

    assert!(asset_op_res.output.get("artifact").is_some());

    // Assert that test.read_only_asset DID write to steps.jsonl
    let steps_content = fs::read_to_string(&steps_file).expect("steps.jsonl exists");
    assert!(
        steps_content.contains("test.read_only_asset"),
        "read-only op with asset output must write a step record"
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn preview_document_changing_op_refused_and_unchanged() {
    let base = temp_dir("preview-change");
    let event_bus = EventBus::new();
    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-preview-change"), &base)
        .event_bus(event_bus.clone())
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-preview-doc");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "Preview Project",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    let f1 = base.join("file1.txt");
    fs::write(&f1, "hello preview").expect("write f1");

    let import_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "asset.import".to_string(),
                input: json!({ "paths": [f1.to_str().unwrap()] }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("import");
    let asset_id = import_res.output["assets"][0].as_str().unwrap().to_string();

    let create_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.create".to_string(),
                input: json!({ "name": "Preview Col", "assets": [&asset_id] }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("create");
    let col_id = create_res.output["collection_id"]
        .as_str()
        .unwrap()
        .to_string();

    // 1. Previewing collection.set_slot returns the diagnostic
    let preview_err = host
        .operation_service()
        .preview(
            &project_id,
            "collection.set_slot",
            json!({
                "collection_id": &col_id,
                "slot_id": "slot-1",
                "asset": "asset-nonexistent",
            }),
            "preview-key-slot",
            CancelToken::new(),
        )
        .await
        .expect_err("preview must refuse document mutator");

    assert_eq!(
        preview_err.code().as_str(),
        "COLLECTION/CALLER_DOCUMENT_CHANGE_PREVIEW_REFUSED"
    );

    // 2. Collection is unchanged
    let get_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.get".to_string(),
                input: json!({ "collection_id": &col_id }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("get");
    assert_eq!(get_res.output["version"], 0);
    assert_eq!(get_res.output["slots"][0]["asset_id"], asset_id.as_str());

    // 3. Previewing collection.get works
    let preview_get_res = host
        .operation_service()
        .preview(
            &project_id,
            "collection.get",
            json!({ "collection_id": &col_id }),
            "preview-key-get",
            CancelToken::new(),
        )
        .await
        .expect("preview of collection.get must work");
    assert_eq!(preview_get_res["id"], col_id.as_str());
    assert_eq!(preview_get_res["slots"][0]["asset_id"], asset_id.as_str());

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn collection_create_board_failure_leaves_no_file_session_or_event() {
    let base = temp_dir("board-fail");
    let event_bus = EventBus::new();
    let mut event_rx = event_bus.subscribe();

    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-board-fail"), &base)
        .event_bus(event_bus.clone())
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-board-fail");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "Fail Project",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    let f1 = base.join("f_fail.txt");
    fs::write(&f1, "fail test").expect("write f_fail");
    let import_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "asset.import".to_string(),
                input: json!({ "paths": [f1.to_str().unwrap()] }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("import");
    let asset_id = import_res.output["assets"][0].as_str().unwrap().to_string();

    // Make board.json unwritable by replacing it with a directory, causing write_document to fail
    let board_path = host.board_service().board_path(&project_id);
    fs::remove_file(&board_path).expect("remove board file");
    fs::create_dir(&board_path).expect("create dir at board path");

    // Invoke collection.create
    let create_err = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.create".to_string(),
                input: json!({ "name": "Fail Col", "assets": [&asset_id] }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect_err("create must fail when board append fails");

    assert!(create_err.code().as_str().contains("BOARD"));

    // Verify no collection files remain in projects/<id>/collections
    let col_dir = host
        .config()
        .paths()
        .project_dir(project_id.as_str())
        .join("collections");
    if col_dir.exists() {
        let count = fs::read_dir(&col_dir).unwrap().count();
        assert_eq!(count, 0, "no collection files should remain");
    }

    // Verify no collection.changed events were received
    let mut received_changed = false;
    while let Ok(event) = event_rx.try_recv() {
        if let Event::Document { kind, .. } = event
            && kind == "collection.changed"
        {
            received_changed = true;
        }
    }
    assert!(
        !received_changed,
        "no collection.changed event must be emitted"
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn concurrent_collection_create_both_succeed() {
    let base = temp_dir("concurrent-create");
    let event_bus = EventBus::new();
    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-concurrent"), &base)
        .event_bus(event_bus.clone())
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-concurrent");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "Concurrent Project",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    let f1 = base.join("c1.txt");
    let f2 = base.join("c2.txt");
    fs::write(&f1, "c1").expect("write c1");
    fs::write(&f2, "c2").expect("write c2");
    let import_res = host
        .operation_service()
        .invoke(
            &project_id,
            InvokeRequest {
                name: "asset.import".to_string(),
                input: json!({ "paths": [f1.to_str().unwrap(), f2.to_str().unwrap()] }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("import");
    let a1 = import_res.output["assets"][0].as_str().unwrap().to_string();
    let a2 = import_res.output["assets"][1].as_str().unwrap().to_string();

    let op_svc = host.operation_service();

    let req1 = InvokeRequest {
        name: "collection.create".to_string(),
        input: json!({ "name": "Concurrent 1", "assets": [&a1] }),
        context: test_context("worker-1"),
        cancel: CancelToken::new(),
    };
    let req2 = InvokeRequest {
        name: "collection.create".to_string(),
        input: json!({ "name": "Concurrent 2", "assets": [&a2] }),
        context: test_context("worker-2"),
        cancel: CancelToken::new(),
    };

    let (res1, res2) = tokio::join!(
        op_svc.invoke(&project_id, req1),
        op_svc.invoke(&project_id, req2),
    );

    let out1 = res1.expect("create 1 succeeded");
    let out2 = res2.expect("create 2 succeeded");

    let item_id_1 = out1.output["board_item_id"].as_str().unwrap();
    let item_id_2 = out2.output["board_item_id"].as_str().unwrap();

    // Check Board snapshot has both items
    let board = host
        .board_service()
        .snapshot(&project_id)
        .await
        .expect("snapshot board");
    assert_eq!(board.items().len(), 2);
    let ids: Vec<&str> = board.items().keys().map(|id| id.as_str()).collect();
    assert!(ids.contains(&item_id_1));
    assert!(ids.contains(&item_id_2));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn concurrent_set_slot_on_different_slots_both_apply() {
    let base = temp_dir("concurrent-set-slot");
    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-concurrent-set"), &base)
        .build()
        .await
        .expect("build host");

    let project_id = ProjectId::new("proj-concurrent-set");
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "Concurrent Set Project",
                "desc",
                "2026-01-01T00:00:00Z".into(),
                "2026-01-01T00:00:00Z".into(),
            ),
        )
        .await
        .expect("create project");

    let files: Vec<_> = (0..4)
        .map(|i| {
            let path = base.join(format!("s{i}.txt"));
            fs::write(&path, format!("content {i}")).expect("write file");
            path
        })
        .collect();
    let paths: Vec<_> = files.iter().map(|p| p.to_str().unwrap()).collect();
    let op_svc = host.operation_service();
    let import_res = op_svc
        .invoke(
            &project_id,
            InvokeRequest {
                name: "asset.import".to_string(),
                input: json!({ "paths": paths }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("import");
    let assets: Vec<String> = import_res.output["assets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    let created = op_svc
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.create".to_string(),
                input: json!({ "name": "C", "assets": [&assets[0], &assets[1]] }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("create");
    let collection_id = created.output["collection_id"]
        .as_str()
        .unwrap()
        .to_string();

    let set = |slot: &str, asset: &str| InvokeRequest {
        name: "collection.set_slot".to_string(),
        input: json!({ "collection_id": &collection_id, "slot_id": slot, "asset": asset }),
        context: test_context("worker"),
        cancel: CancelToken::new(),
    };
    let (r1, r2) = tokio::join!(
        op_svc.invoke(&project_id, set("slot-1", &assets[2])),
        op_svc.invoke(&project_id, set("slot-2", &assets[3])),
    );
    r1.expect("set slot-1");
    r2.expect("set slot-2");

    let got = op_svc
        .invoke(
            &project_id,
            InvokeRequest {
                name: "collection.get".to_string(),
                input: json!({ "collection_id": &collection_id }),
                context: test_context("human"),
                cancel: CancelToken::new(),
            },
        )
        .await
        .expect("get");
    assert_eq!(got.output["slots"][0]["asset_id"], json!(assets[2]));
    assert_eq!(got.output["slots"][1]["asset_id"], json!(assets[3]));

    let _ = fs::remove_dir_all(&base);
}
