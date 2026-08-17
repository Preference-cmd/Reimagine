//! Integration tests for ProjectService / BoardService (AR-07):
//! project CRUD with atomic persistence, auto-created board document,
//! cascade deletion, and board.changed domain events.

use reimagine_agent_harness::WorkspaceScope;
use reimagine_app_host::{AppHostError, WorkspaceHost};
use reimagine_core::board::{
    BoardCommand, BoardCommandBatch, BoardCommandResultStatus, BoardItemKind, BoardItemPosition,
    BoardItemSize,
};
use reimagine_core::command::{CommandActor, CommandActorKind, CommandProvenance};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{BoardItemId, BoardVersion, CommandBatchId, ProjectId, WorkflowId};
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

fn add_item_batch(base_version: BoardVersion, item_id: &str, label: &str) -> BoardCommandBatch {
    BoardCommandBatch::new(
        CommandBatchId::new(format!("batch-{label}")),
        CommandActor::new(CommandActorKind::Human),
        base_version,
        CommandProvenance::Direct,
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
    assert_eq!(board.version(), BoardVersion::new(0));
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

    let mut events = host.board_service().subscribe();
    let project_id = ProjectId::new("proj-board");

    let result = host
        .board_service()
        .apply_batch(
            &project_id,
            add_item_batch(BoardVersion::new(0), "item-1", "a"),
        )
        .await
        .expect("apply succeeds");
    assert_eq!(result.status(), BoardCommandResultStatus::Applied);
    assert_eq!(result.board_version(), BoardVersion::new(1));

    let event = tokio::time::timeout(std::time::Duration::from_secs(5), events.recv())
        .await
        .expect("board.changed arrives")
        .expect("event channel open");
    assert_eq!(event.kind, "board.changed");
    assert_eq!(event.project_id, "proj-board");
    assert_eq!(event.board_version, 1);

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
            add_item_batch(BoardVersion::new(0), "item-2", "b"),
        )
        .await
        .expect("apply returns a result");
    assert_eq!(stale.status(), BoardCommandResultStatus::Rejected);

    // Undo restores the empty board, persists it, and re-emits the event.
    let undo = host
        .board_service()
        .undo(&project_id)
        .await
        .expect("undo succeeds")
        .expect("history entry exists");
    assert_eq!(undo.status(), BoardCommandResultStatus::Applied);
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
    assert_eq!(undo_event.board_version, 2);

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

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-project-board-{prefix}-{nonce}"))
}
