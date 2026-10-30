//! Board history guard: an undo or redo that names a history entry other
//! than the current top of its stack is rejected before the board, its
//! persistence or the activity log change.

use std::path::PathBuf;

use reimagine_app_host::{
    AppHostError, AppHostErrorCode, InvokeRequest, WorkspaceHost, WorkspaceScope,
};
use reimagine_core::board::{BoardCommand, BoardItemKind, BoardItemPosition, BoardItemSize};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResult, CommandStatus,
    OperationContext,
};
use reimagine_core::event::{EventReceiver, Timestamp};
use reimagine_core::model::{BoardItemId, CommandBatchId, DocumentVersion, ProjectId};
use reimagine_core::project::ProjectMetadata;
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};

fn temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-board-history-{prefix}-{nonce}"))
}

fn build_host(base: &std::path::Path) -> WorkspaceHost {
    WorkspaceHost::with_defaults(WorkspaceScope::new("board-history"), base)
}

async fn create_project(host: &WorkspaceHost, id: &str) -> ProjectId {
    let project_id = ProjectId::new(id);
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "History test",
                "board history test project",
                Timestamp::new("2026-01-01T00:00:00Z"),
                Timestamp::new("2026-01-01T00:00:00Z"),
            ),
        )
        .await
        .expect("create project");
    project_id
}

fn human_context(label: &str) -> OperationContext {
    OperationContext::new(
        CommandActor::new(CommandActorKind::Human)
            .with_id("human-1")
            .with_label(label),
        CommandProvenance::Direct,
    )
}

fn human_actor() -> CommandActor {
    CommandActor::new(CommandActorKind::Human)
}

fn note_batch(
    base_version: DocumentVersion,
    batch_id: &str,
    item_id: &str,
) -> CommandBatch<BoardCommand> {
    CommandBatch::new(
        CommandBatchId::new(batch_id),
        human_context("You"),
        base_version,
        Timestamp::new("2026-01-01T00:00:00Z"),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new(item_id),
            kind: BoardItemKind::Note {
                content: format!("note {item_id}"),
            },
            position: BoardItemPosition::new(0, 0),
            size: BoardItemSize::new(200, 120),
        }],
    )
}

async fn apply_note(
    host: &WorkspaceHost,
    project: &ProjectId,
    version: u64,
    batch_id: &str,
    item_id: &str,
) -> CommandResult<reimagine_core::board::BoardChange> {
    host.board_service()
        .apply_batch(
            project,
            note_batch(DocumentVersion::new(version), batch_id, item_id),
        )
        .await
        .expect("apply note batch")
}

fn drain_events(events: &mut EventReceiver) {
    while events.try_recv().is_ok() {}
}

async fn activity_kinds(host: &WorkspaceHost, project: &ProjectId) -> Vec<String> {
    let result = host
        .operation_service()
        .invoke(
            project,
            InvokeRequest {
                name: "activity.list".to_owned(),
                input: json!({ "limit": 50 }),
                context: human_context("You"),
                cancel: CancelToken::new(),
                run: None,
            },
        )
        .await
        .expect("activity.list succeeds");
    let output: Value = result.output;
    output["records"]
        .as_array()
        .expect("records array")
        .iter()
        .map(|record| record["kind"].as_str().expect("record kind").to_owned())
        .collect()
}

#[tokio::test]
async fn expected_history_entries_gate_board_undo_and_redo() {
    let base = temp_dir("history-guard");
    let host = build_host(&base);
    let project = create_project(&host, "proj-history").await;
    let mut events = host.event_bus().subscribe();

    let first = apply_note(&host, &project, 0, "batch-1", "note-1").await;
    let second = apply_note(&host, &project, 1, "batch-2", "note-2").await;
    let first_id = first
        .history_entry_id()
        .cloned()
        .expect("first apply names its history entry");
    let second_id = second
        .history_entry_id()
        .cloned()
        .expect("second apply names its history entry");
    assert_ne!(first_id, second_id);
    drain_events(&mut events);

    // The older entry is undoable only after the newer one, so naming it is
    // rejected before the board, its file or the activity log change.
    let error = host
        .board_service()
        .undo(&project, human_actor(), Some(&first_id))
        .await
        .expect_err("the older entry is not the undo top");
    assert_eq!(error.code(), AppHostErrorCode::Conflict);
    let details = error.details().expect("mismatch carries details");
    assert_eq!(
        details,
        json!({
            "project_id": "proj-history",
            "action": "undo",
            "expected_entry_id": first_id.to_string(),
            "undo_entry_id": second_id.to_string(),
            "redo_entry_id": null,
        })
    );
    match error {
        AppHostError::BoardHistoryMismatch {
            project_id,
            action,
            expected_entry_id,
            undo_entry_id,
            redo_entry_id,
        } => {
            assert_eq!(project_id, project);
            assert_eq!(action, "undo");
            assert_eq!(expected_entry_id, first_id);
            assert_eq!(undo_entry_id, Some(second_id.clone()));
            assert_eq!(redo_entry_id, None);
        }
        other => panic!("expected BoardHistoryMismatch, got: {other:?}"),
    }

    let snapshot = host
        .board_service()
        .snapshot(&project)
        .await
        .expect("snapshot loads");
    assert_eq!(snapshot.items().len(), 2);
    assert_eq!(snapshot.version(), DocumentVersion::new(2));

    // The snapshot matches the file only if the rejected undo never wrote the
    // board document.
    let board_bytes = tokio::fs::read(base.join("projects/proj-history/board.json"))
        .await
        .expect("board file is readable");
    let board_file: Value = serde_json::from_slice(&board_bytes).expect("board file is JSON");
    assert_eq!(board_file["version"], json!(2));
    assert_eq!(
        board_file["items"]
            .as_object()
            .expect("board file stores items by id")
            .len(),
        2
    );
    assert!(
        events.try_recv().is_err(),
        "a rejected undo publishes neither board.changed nor activity.appended"
    );
    assert_eq!(
        activity_kinds(&host, &project).await,
        vec!["applied".to_owned(), "applied".to_owned()],
        "a rejected undo appends no record"
    );

    // Naming the actual top succeeds and returns that entry.
    let undone = host
        .board_service()
        .undo(&project, human_actor(), Some(&second_id))
        .await
        .expect("undo with the current top succeeds")
        .expect("a history entry exists");
    assert_eq!(undone.status(), CommandStatus::Applied);
    assert_eq!(undone.history_entry_id(), Some(&second_id));
    let snapshot = host
        .board_service()
        .snapshot(&project)
        .await
        .expect("snapshot loads");
    assert_eq!(snapshot.items().len(), 1);
    assert_eq!(snapshot.version(), DocumentVersion::new(3));
    drain_events(&mut events);

    // The redo top is the entry just undone. Naming the other entry is
    // rejected before the board changes.
    let error = host
        .board_service()
        .redo(&project, human_actor(), Some(&first_id))
        .await
        .expect_err("the older entry is not the redo top");
    assert_eq!(error.code(), AppHostErrorCode::Conflict);
    assert_eq!(
        error.details(),
        Some(json!({
            "project_id": "proj-history",
            "action": "redo",
            "expected_entry_id": first_id.to_string(),
            "undo_entry_id": first_id.to_string(),
            "redo_entry_id": second_id.to_string(),
        }))
    );
    let snapshot = host
        .board_service()
        .snapshot(&project)
        .await
        .expect("snapshot loads");
    assert_eq!(snapshot.items().len(), 1);
    assert_eq!(snapshot.version(), DocumentVersion::new(3));
    assert!(
        events.try_recv().is_err(),
        "a rejected redo publishes neither board.changed nor activity.appended"
    );
    assert_eq!(
        activity_kinds(&host, &project).await,
        vec![
            "undone".to_owned(),
            "applied".to_owned(),
            "applied".to_owned()
        ],
        "a rejected redo appends no record"
    );

    // Naming the actual redo top succeeds.
    let redone = host
        .board_service()
        .redo(&project, human_actor(), Some(&second_id))
        .await
        .expect("redo with the current top succeeds")
        .expect("a redo entry exists");
    assert_eq!(redone.status(), CommandStatus::Applied);
    assert_eq!(redone.history_entry_id(), Some(&second_id));
    let snapshot = host
        .board_service()
        .snapshot(&project)
        .await
        .expect("snapshot loads");
    assert_eq!(snapshot.items().len(), 2);
    assert_eq!(snapshot.version(), DocumentVersion::new(4));

    // No expected entry keeps the plain "act on the current top" behavior.
    let undone = host
        .board_service()
        .undo(&project, human_actor(), None)
        .await
        .expect("unguarded undo succeeds")
        .expect("a history entry exists");
    assert_eq!(undone.history_entry_id(), Some(&second_id));
    let redone = host
        .board_service()
        .redo(&project, human_actor(), None)
        .await
        .expect("unguarded redo succeeds")
        .expect("a redo entry exists");
    assert_eq!(redone.history_entry_id(), Some(&second_id));

    let _ = std::fs::remove_dir_all(&base);
}
