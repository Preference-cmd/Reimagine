//! Integration tests for ProjectService / BoardService (AR-07):
//! project CRUD with atomic persistence, auto-created board document,
//! cascade deletion, and board.changed domain events.

use reimagine_app_host::WorkspaceScope;
use reimagine_app_host::{AppHostError, WorkspaceHost};
use reimagine_core::board::{BoardCommand, BoardItemKind, BoardItemPosition, BoardItemSize};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandStatus,
    OperationContext,
};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{BoardItemId, CommandBatchId, DocumentVersion, ProjectId, WorkflowId};
use reimagine_core::project::ProjectMetadata;

fn build_host(scope: &str) -> WorkspaceHost {
    WorkspaceHost::with_defaults(WorkspaceScope::new(scope), temp_dir(scope))
}

fn metadata(name: &str, updated_at: &str) -> ProjectMetadata {
    ProjectMetadata::new(
        name,
        "integration test project",
        Timestamp::new("2026-06-12T00:00:00Z"),
        Timestamp::new(updated_at),
    )
}

fn add_item_batch(
    base_version: DocumentVersion,
    item_id: &str,
    label: &str,
) -> CommandBatch<BoardCommand> {
    CommandBatch::new(
        CommandBatchId::new(format!("batch-{label}")),
        OperationContext::new(
            CommandActor::new(CommandActorKind::Human),
            CommandProvenance::Direct,
        ),
        base_version,
        Timestamp::new("2026-06-12T00:00:00Z"),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new(item_id),
            kind: BoardItemKind::WorkflowRef {
                workflow_id: WorkflowId::new(format!("wf-{label}")),
            },
            position: BoardItemPosition::new(100, 200),
            size: BoardItemSize::new(300, 150),
        }],
    )
}

#[tokio::test]
async fn project_creation_persists_project_and_creates_board() {
    let host = build_host("project-creation");
    let project_id = ProjectId::new("proj-1");
    let project = host
        .project_service()
        .create_project(
            project_id.clone(),
            metadata("First", "2026-06-12T00:01:00Z"),
        )
        .await
        .expect("project creates");

    assert_eq!(project.id(), &project_id);
    assert!(host.project_service().project_file(&project_id).is_file());
    assert!(host.board_service().board_path(&project_id).is_file());

    let board = host
        .board_service()
        .snapshot(&project_id)
        .await
        .expect("board loads");
    assert_eq!(board.project_id(), &project_id);
    assert_eq!(board.version(), DocumentVersion::new(0));
    assert!(board.items().is_empty());

    let duplicate = host
        .project_service()
        .create_project(
            project_id.clone(),
            metadata("Again", "2026-06-12T00:02:00Z"),
        )
        .await;
    assert!(matches!(
        duplicate,
        Err(AppHostError::ProjectAlreadyExists { .. })
    ));
}

#[tokio::test]
async fn project_load_list_update_and_delete_round_trip() {
    let host = build_host("project-crud");
    let project_service = host.project_service();

    let first = project_service
        .create_project(
            ProjectId::new("proj-a"),
            metadata("Project A", "2026-06-12T00:01:00Z"),
        )
        .await
        .expect("first project creates");
    let second = project_service
        .create_project(
            ProjectId::new("proj-b"),
            metadata("Project B", "2026-06-12T00:01:00Z"),
        )
        .await
        .expect("second project creates");

    let listed = project_service
        .list_projects()
        .await
        .expect("projects list");
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].id(), first.id());
    assert_eq!(listed[1].id(), second.id());

    let updated = project_service
        .update_project(
            first.id(),
            metadata("Project A renamed", "2026-06-12T00:02:00Z"),
        )
        .await
        .expect("project updates");
    assert_eq!(updated.metadata().name(), "Project A renamed");

    let reloaded = project_service
        .load_project(first.id())
        .await
        .expect("project reloads");
    assert_eq!(reloaded.metadata().name(), "Project A renamed");

    project_service
        .delete_project(second.id())
        .await
        .expect("project deletes");
    assert!(!project_service.project_dir(second.id()).exists());
    assert!(matches!(
        project_service.load_project(second.id()).await,
        Err(AppHostError::UnknownProject { .. })
    ));
}

#[tokio::test]
async fn board_apply_persists_emits_changed_and_rejects_stale_versions() {
    let host = build_host("board-apply");
    host.project_service()
        .create_project(
            ProjectId::new("proj-board"),
            metadata("Board project", "2026-06-12T00:01:00Z"),
        )
        .await
        .expect("project creates");

    let mut events = host.event_bus().subscribe();
    let project_id = ProjectId::new("proj-board");

    let result = host
        .board_service()
        .apply_batch(
            &project_id,
            add_item_batch(DocumentVersion::new(0), "item-1", "a"),
        )
        .await
        .expect("apply succeeds");
    assert_eq!(result.status(), CommandStatus::Applied);
    assert_eq!(result.version(), DocumentVersion::new(1));

    let event = tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
        .await
        .expect("board.changed arrives")
        .expect("event channel open");
    match event {
        reimagine_core::event::Event::Document {
            kind,
            project_id,
            document_id,
            version,
        } => {
            assert_eq!(kind, "board.changed");
            assert_eq!(project_id, "proj-board");
            assert_eq!(document_id, "board-proj-board");
            assert_eq!(version, 1);
        }
        other => panic!("expected Event::Document, got: {other:?}"),
    }
    assert!(
        matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ),
        "one durable Board change must publish exactly one workspace event"
    );

    // The change is durable on disk, not just in the session cache.
    let persisted: reimagine_core::board::BoardDocument = serde_json::from_slice(
        &std::fs::read(host.board_service().board_path(&project_id)).expect("board file readable"),
    )
    .expect("board file is valid JSON");
    assert_eq!(persisted.items().len(), 1);
    assert!(persisted.item(&BoardItemId::new("item-1")).is_some());

    // A stale base version must be rejected without mutation.
    let stale = host
        .board_service()
        .apply_batch(
            &project_id,
            add_item_batch(DocumentVersion::new(0), "item-2", "b"),
        )
        .await
        .expect("apply returns a result");
    assert_eq!(stale.status(), CommandStatus::Rejected);

    // Undo restores the empty board, persists it, and re-emits the event.
    let undo = host
        .board_service()
        .undo(&project_id)
        .await
        .expect("undo succeeds")
        .expect("history entry exists");
    assert_eq!(undo.status(), CommandStatus::Applied);
    assert!(
        host.board_service()
            .snapshot(&project_id)
            .await
            .expect("snapshot loads")
            .items()
            .is_empty()
    );
    let undo_event = tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
        .await
        .expect("undo event arrives")
        .expect("event channel open");
    match undo_event {
        reimagine_core::event::Event::Document {
            kind,
            document_id,
            version,
            ..
        } => {
            assert_eq!(kind, "board.changed");
            assert_eq!(document_id, "board-proj-board");
            assert_eq!(version, 2);
        }
        other => panic!("expected Event::Document, got: {other:?}"),
    }
    assert!(
        matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ),
        "undo must also publish exactly one workspace event"
    );

    // Redo persists the restored item and publishes one more shared event.
    let redo = host
        .board_service()
        .redo(&project_id)
        .await
        .expect("redo succeeds")
        .expect("history entry exists");
    assert_eq!(redo.status(), CommandStatus::Applied);
    let redo_event = tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
        .await
        .expect("redo event arrives")
        .expect("event channel open");
    match redo_event {
        reimagine_core::event::Event::Document {
            kind,
            document_id,
            version,
            ..
        } => {
            assert_eq!(kind, "board.changed");
            assert_eq!(document_id, "board-proj-board");
            assert_eq!(version, 3);
        }
        other => panic!("expected Event::Document, got: {other:?}"),
    }
    assert!(
        matches!(
            events.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ),
        "redo must also publish exactly one workspace event"
    );

    // Deleting the project drops the in-memory board session.
    host.project_service()
        .delete_project(&project_id)
        .await
        .expect("project deletes");
    assert!(matches!(
        host.board_service().snapshot(&project_id).await,
        Err(AppHostError::UnknownBoard { .. })
    ));
}

#[tokio::test]
async fn workflow_noop_does_not_emit_changed_event() {
    let host = build_host("workflow-event");
    let project_id = ProjectId::new("workflow-project");
    host.project_service()
        .create_project(
            project_id.clone(),
            metadata("Workflow project", "2026-06-12T00:01:00Z"),
        )
        .await
        .expect("project creates");
    let workflow_service = host.workflow_service_for_project(&project_id);
    let workflow_id = workflow_service.register_workflow(reimagine_core::workflow::Workflow::new(
        "workflow-event",
        reimagine_core::model::DocumentVersion::new(0),
    ));
    let mut events = host.event_bus().subscribe();
    let batch = reimagine_core::command::CommandBatch::new(
        CommandBatchId::new("workflow-event-batch"),
        reimagine_core::command::OperationContext::new(
            CommandActor::new(CommandActorKind::Human),
            CommandProvenance::Direct,
        ),
        reimagine_core::model::DocumentVersion::new(0),
        Timestamp::new("2026-06-12T00:00:00Z"),
        Vec::new(),
    );
    let result = workflow_service
        .apply_commands(&workflow_id, host.node_catalog().as_ref(), batch)
        .await
        .expect("workflow apply returns");
    assert_eq!(
        result.status(),
        reimagine_core::command::CommandStatus::NoOp
    );
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), events.recv())
            .await
            .is_err()
    );
    let _ = std::fs::remove_dir_all(host.base_path());
}

#[tokio::test]
async fn host_project_delete_clears_workflow_registry_for_recreate() {
    let host = build_host("workflow-registry-delete");
    let project_id = ProjectId::new("registry-project");
    host.project_service()
        .create_project(
            project_id.clone(),
            metadata("Registry", "2026-06-12T00:01:00Z"),
        )
        .await
        .expect("project creates");
    let first = host.workflow_service_for_project(&project_id);
    let workflow_id = first.register_workflow(reimagine_core::workflow::Workflow::new(
        "wf-deleted",
        reimagine_core::model::DocumentVersion::new(0),
    ));
    assert!(first.contains(&workflow_id));
    host.delete_project(&project_id)
        .await
        .expect("host deletes project");
    host.project_service()
        .create_project(
            project_id.clone(),
            metadata("Registry recreated", "2026-06-12T00:02:00Z"),
        )
        .await
        .expect("project recreates");
    let second = host.workflow_service_for_project(&project_id);
    assert!(!second.contains(&workflow_id));
    let _ = std::fs::remove_dir_all(host.base_path());
}

#[tokio::test]
async fn regression_project_document_format_unchanged() {
    let host = build_host("project-regression");
    let project_id = ProjectId::new("proj-regression");
    let project_file = host.project_service().project_file(&project_id);

    // Exact JSON shape written by ProjectService today.
    let fixture_raw = serde_json::json!({
        "id": "proj-regression",
        "metadata": {
            "name": "Regression Project",
            "description": "persisted project document fixture",
            "created_at": "2026-06-12T00:00:00Z",
            "updated_at": "2026-06-12T00:01:00Z"
        }
    });

    let parent = project_file.parent().expect("parent dir");
    tokio::fs::create_dir_all(parent).await.expect("create dir");
    tokio::fs::write(
        &project_file,
        serde_json::to_vec_pretty(&fixture_raw).expect("serialize"),
    )
    .await
    .expect("write fixture");

    // Load through the service.
    let loaded = host
        .project_service()
        .load_project(&project_id)
        .await
        .expect("loads project");
    assert_eq!(loaded.id(), &project_id);
    assert_eq!(loaded.metadata().name(), "Regression Project");

    // Save it back through the service.
    host.project_service()
        .update_project(&project_id, loaded.metadata().clone())
        .await
        .expect("update saves project");

    // Read the re-saved file and assert JSON value equality.
    let resaved_bytes = tokio::fs::read(&project_file).await.expect("read resaved");
    let resaved_val: serde_json::Value =
        serde_json::from_slice(&resaved_bytes).expect("parse resaved json");
    assert_eq!(fixture_raw, resaved_val);

    let _ = std::fs::remove_dir_all(host.base_path());
}

#[tokio::test]
async fn regression_board_document_format_unchanged() {
    let host = build_host("board-regression");
    let project_id = ProjectId::new("proj-board-reg");
    let board_path = host.board_service().board_path(&project_id);

    // Exact JSON shape written by BoardService today.
    let fixture_raw = serde_json::json!({
        "schema_version": "reimagine.board.v1",
        "id": "board-proj-board-reg",
        "project_id": "proj-board-reg",
        "version": 0,
        "items": {
            "item-1": {
                "id": "item-1",
                "kind": {
                    "type": "workflow_ref",
                    "workflow_id": "wf-reg-1"
                },
                "position": {
                    "x": 100,
                    "y": 200
                },
                "size": {
                    "width": 300,
                    "height": 150
                },
                "z": 0,
                "locked": false
            }
        }
    });

    let parent = board_path.parent().expect("parent dir");
    tokio::fs::create_dir_all(parent).await.expect("create dir");
    tokio::fs::write(
        &board_path,
        serde_json::to_vec_pretty(&fixture_raw).expect("serialize"),
    )
    .await
    .expect("write fixture");

    // Load through the service.
    let loaded = host
        .board_service()
        .load_or_create(&project_id)
        .await
        .expect("load board");
    assert_eq!(loaded.project_id(), &project_id);
    assert_eq!(loaded.id().as_str(), "board-proj-board-reg");
    assert_eq!(loaded.items().len(), 1);

    // Save it back through the service by applying one move.
    let move_batch = CommandBatch::new(
        CommandBatchId::new("batch-move-reg"),
        OperationContext::new(
            CommandActor::new(CommandActorKind::Human),
            CommandProvenance::Direct,
        ),
        DocumentVersion::new(0),
        Timestamp::new("2026-06-12T00:00:00Z"),
        vec![BoardCommand::MoveItem {
            item_id: BoardItemId::new("item-1"),
            position: BoardItemPosition::new(110, 200),
        }],
    );
    let result = host
        .board_service()
        .apply_batch(&project_id, move_batch)
        .await
        .expect("apply move");
    assert_eq!(result.status(), CommandStatus::Applied);

    // The re-saved file differs from the fixture only in what the move changed.
    let mut expected = fixture_raw;
    expected["version"] = serde_json::json!(1);
    expected["items"]["item-1"]["position"]["x"] = serde_json::json!(110);
    let resaved_bytes = tokio::fs::read(&board_path).await.expect("read resaved");
    let resaved_val: serde_json::Value =
        serde_json::from_slice(&resaved_bytes).expect("parse resaved json");
    assert_eq!(expected, resaved_val);

    let _ = std::fs::remove_dir_all(host.base_path());
}

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-project-board-{prefix}-{nonce}"))
}
