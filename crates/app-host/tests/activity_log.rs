//! Project activity log: board apply/undo/redo records, recorded op steps,
//! `activity.list` paging, the `activity.appended` event, persistence and a
//! torn `activity.jsonl` tail.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use reimagine_app_host::{InvokeRequest, WorkspaceHost, WorkspaceHostBuilder, WorkspaceScope};
use reimagine_core::board::{
    BoardChange, BoardCommand, BoardItemKind, BoardItemPosition, BoardItemSize,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResult,
    OperationContext,
};
use reimagine_core::event::{Event, Timestamp};
use reimagine_core::model::{BoardItemId, CommandBatchId, DocumentVersion, ProjectId};
use reimagine_core::project::ProjectMetadata;
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};

fn temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-activity-{prefix}-{nonce}"))
}

async fn build_host(base: &Path) -> WorkspaceHost {
    WorkspaceHostBuilder::new(WorkspaceScope::new("test-activity"), base)
        .build()
        .await
        .expect("build host")
}

async fn create_project(host: &WorkspaceHost, id: &str) -> ProjectId {
    let project_id = ProjectId::new(id);
    host.project_service()
        .create_project(
            project_id.clone(),
            ProjectMetadata::new(
                "Activity test",
                "activity test project",
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

fn agent_context() -> OperationContext {
    OperationContext::new(
        CommandActor::new(CommandActorKind::Agent)
            .with_id("agent-1")
            .with_label("Agent"),
        CommandProvenance::Direct,
    )
}

fn note_batch(
    base_version: DocumentVersion,
    batch_id: &str,
    item_id: &str,
    context: OperationContext,
) -> CommandBatch<BoardCommand> {
    CommandBatch::new(
        CommandBatchId::new(batch_id),
        context,
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
) -> CommandResult<BoardChange> {
    host.board_service()
        .apply_batch(
            project,
            note_batch(
                DocumentVersion::new(version),
                batch_id,
                item_id,
                human_context("You"),
            ),
        )
        .await
        .expect("apply note batch")
}

async fn invoke(
    host: &WorkspaceHost,
    project: &ProjectId,
    name: &str,
    input: Value,
    context: OperationContext,
) -> Value {
    host.operation_service()
        .invoke(
            project,
            InvokeRequest {
                name: name.to_owned(),
                input,
                context,
                cancel: CancelToken::new(),
                run: None,
            },
        )
        .await
        .unwrap_or_else(|error| panic!("{name} failed: {}", error.message()))
        .output
}

async fn list(
    host: &WorkspaceHost,
    project: &ProjectId,
    before: Option<&str>,
    limit: Option<u64>,
) -> Value {
    let mut input = json!({});
    if let Some(before) = before {
        input["before"] = json!(before);
    }
    if let Some(limit) = limit {
        input["limit"] = json!(limit);
    }
    invoke(host, project, "activity.list", input, human_context("You")).await
}

fn records(output: &Value) -> &[Value] {
    output["records"].as_array().expect("records array")
}

fn ids(output: &Value) -> Vec<String> {
    records(output)
        .iter()
        .map(|record| record["id"].as_str().expect("record id").to_owned())
        .collect()
}

fn project_dir(host: &WorkspaceHost, project: &ProjectId) -> PathBuf {
    host.config().paths().project_dir(project.as_str())
}

#[tokio::test]
async fn board_apply_undo_redo_and_op_record_are_listed_newest_first() {
    let base = temp_dir("records");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-activity").await;

    let applied = apply_note(&host, &project, 0, "batch-note-1", "note-1").await;
    let history_entry_id = applied
        .history_entry_id()
        .cloned()
        .expect("an applied board batch names its history entry")
        .to_string();
    assert!(host.board_service().undo(&project).await.unwrap().is_some());
    assert!(host.board_service().redo(&project).await.unwrap().is_some());

    // A write op with an Agent actor records its own step; the board item it
    // creates through `append_item` must not add a second record.
    let created = invoke(
        &host,
        &project,
        "collection.create",
        json!({ "name": "Agent batch", "assets": [] }),
        agent_context(),
    )
    .await;
    assert!(created["collection_id"].as_str().is_some());

    let listed = list(&host, &project, None, None).await;
    let all = records(&listed);
    assert_eq!(
        all.len(),
        4,
        "apply, undo, redo and the op step each append exactly one record"
    );
    assert_eq!(listed["has_more"], json!(false));

    let op_record = &all[0];
    assert_eq!(op_record["kind"], json!("applied"));
    assert_eq!(op_record["summaryKey"], json!("activity.op"));
    assert_eq!(op_record["summary"], json!("Ran collection.create"));
    assert_eq!(
        op_record["summaryParams"],
        json!({ "op": "collection.create" })
    );
    assert_eq!(op_record["actor"]["kind"], json!("Agent"));
    assert_eq!(op_record["actor"]["id"], json!("agent-1"));
    assert_eq!(op_record["actor"]["label"], json!("Agent"));
    assert_eq!(op_record["targets"], json!([]));
    assert_eq!(op_record["historyEntryId"], Value::Null);

    let redo_record = &all[1];
    assert_eq!(redo_record["kind"], json!("redone"));
    assert_eq!(redo_record["summaryKey"], json!("activity.board.redo"));
    assert_eq!(redo_record["summary"], json!("Redid the last board change"));
    assert_eq!(redo_record["actor"]["kind"], json!("Human"));
    assert_eq!(redo_record["actor"]["id"], Value::Null);
    assert_eq!(redo_record["targets"], json!(["note-1"]));
    assert_eq!(
        redo_record["historyEntryId"],
        json!(history_entry_id.clone())
    );

    let undo_record = &all[2];
    assert_eq!(undo_record["kind"], json!("undone"));
    assert_eq!(undo_record["summaryKey"], json!("activity.board.undo"));
    assert_eq!(undo_record["summary"], json!("Undid the last board change"));
    assert_eq!(undo_record["actor"]["kind"], json!("Human"));
    assert_eq!(undo_record["targets"], json!(["note-1"]));
    assert_eq!(
        undo_record["historyEntryId"],
        json!(history_entry_id.clone())
    );

    let apply_record = &all[3];
    assert_eq!(apply_record["kind"], json!("applied"));
    assert_eq!(apply_record["summaryKey"], json!("activity.board.addNote"));
    assert_eq!(apply_record["summary"], json!("Added a note"));
    assert_eq!(apply_record["summaryParams"], json!({ "count": 1 }));
    assert_eq!(apply_record["actor"]["kind"], json!("Human"));
    assert_eq!(apply_record["actor"]["id"], json!("human-1"));
    assert_eq!(apply_record["actor"]["label"], json!("You"));
    assert_eq!(apply_record["targets"], json!(["note-1"]));
    assert_eq!(apply_record["historyEntryId"], json!(history_entry_id));

    let ids = ids(&listed);
    assert!(ids.iter().all(|id| id.starts_with("act-")), "{ids:?}");
    for record in all {
        let at = record["at"].as_str().expect("record timestamp");
        chrono::DateTime::parse_from_rfc3339(at).expect("RFC 3339 timestamp");
    }

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn activity_records_persist_across_a_host_restart() {
    let base = temp_dir("restart");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-restart").await;

    apply_note(&host, &project, 0, "batch-persist-1", "note-1").await;
    host.board_service().undo(&project).await.unwrap();
    invoke(
        &host,
        &project,
        "collection.create",
        json!({ "name": "Persisted", "assets": [] }),
        agent_context(),
    )
    .await;

    let before = list(&host, &project, None, None).await;
    let before_ids = ids(&before);
    assert_eq!(before_ids.len(), 3);

    host.shutdown().await;
    drop(host);

    // A new host over the same root reads the same append-only log.
    let host = build_host(&base).await;
    let after = list(&host, &project, None, None).await;
    assert_eq!(ids(&after), before_ids);
    assert_eq!(records(&after)[0]["kind"], json!("applied"));
    assert_eq!(records(&after)[1]["kind"], json!("undone"));
    assert_eq!(records(&after)[2]["kind"], json!("applied"));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn activity_list_pages_newest_first_with_cursor_boundaries() {
    let base = temp_dir("paging");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-paging").await;

    for index in 1..=5 {
        apply_note(
            &host,
            &project,
            index - 1,
            &format!("batch-page-{index}"),
            &format!("item-{index}"),
        )
        .await;
    }

    let first_page = list(&host, &project, None, Some(2)).await;
    assert_eq!(first_page["has_more"], json!(true));
    assert_eq!(records(&first_page).len(), 2);
    assert_eq!(records(&first_page)[0]["targets"], json!(["item-5"]));
    assert_eq!(records(&first_page)[1]["targets"], json!(["item-4"]));

    let second_page = list(
        &host,
        &project,
        records(&first_page)[1]["id"].as_str(),
        Some(2),
    )
    .await;
    assert_eq!(second_page["has_more"], json!(true));
    assert_eq!(records(&second_page)[0]["targets"], json!(["item-3"]));
    assert_eq!(records(&second_page)[1]["targets"], json!(["item-2"]));

    let third_page = list(
        &host,
        &project,
        records(&second_page)[1]["id"].as_str(),
        Some(2),
    )
    .await;
    assert_eq!(third_page["has_more"], json!(false));
    assert_eq!(records(&third_page).len(), 1);
    assert_eq!(records(&third_page)[0]["targets"], json!(["item-1"]));

    // A single-record cursor walk: limit=1 with `before`.
    let first_single = list(&host, &project, None, Some(1)).await;
    assert_eq!(first_single["has_more"], json!(true));
    assert_eq!(records(&first_single)[0]["targets"], json!(["item-5"]));
    let second_single = list(
        &host,
        &project,
        records(&first_single)[0]["id"].as_str(),
        Some(1),
    )
    .await;
    assert_eq!(second_single["has_more"], json!(true));
    assert_eq!(records(&second_single)[0]["targets"], json!(["item-4"]));

    // An unknown cursor starts from the newest record.
    let unknown = list(&host, &project, Some("act-unknown"), Some(1)).await;
    assert_eq!(unknown["has_more"], json!(true));
    assert_eq!(records(&unknown)[0]["targets"], json!(["item-5"]));

    // Paging past the oldest record yields an empty final page.
    let oldest_id = records(&third_page)[0]["id"].as_str().expect("oldest id");
    let past_oldest = list(&host, &project, Some(oldest_id), Some(2)).await;
    assert_eq!(past_oldest["has_more"], json!(false));
    assert!(records(&past_oldest).is_empty());

    // Empty and unknown projects list nothing.
    let empty = create_project(&host, "proj-empty").await;
    for project in [&empty, &ProjectId::new("proj-missing")] {
        let listed = list(&host, project, None, None).await;
        assert_eq!(listed["records"], json!([]));
        assert_eq!(listed["has_more"], json!(false));
    }

    // The list operation itself is read-only and appends nothing.
    let all = list(&host, &project, None, Some(200)).await;
    assert_eq!(records(&all).len(), 5);
    assert_eq!(records(&all)[0]["targets"], json!(["item-5"]));

    let infos = host.operation_service().list(&project).await;
    let info = infos
        .iter()
        .find(|info| info.spec.name == "activity.list")
        .expect("activity.list is registered");
    assert!(info.spec.annotations.read_only);
    assert!(info.spec.annotations.idempotent);

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn board_apply_publishes_activity_appended() {
    let base = temp_dir("event");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-event").await;
    let mut events = host.event_bus().subscribe();

    let applied = apply_note(&host, &project, 0, "batch-event", "note-1").await;
    let history_entry_id = applied
        .history_entry_id()
        .cloned()
        .expect("applied batch history entry")
        .to_string();

    let mut appended = None;
    for _ in 0..4 {
        let event = tokio::time::timeout(Duration::from_secs(5), events.recv())
            .await
            .expect("workspace event arrives")
            .expect("event channel open");
        if let Event::Mod { kind, payload } = event
            && kind == "activity.appended"
        {
            appended = Some(payload);
            break;
        }
    }
    let payload = appended.expect("activity.appended is published");

    assert_eq!(payload["projectId"], json!("proj-event"));
    let record = &payload["record"];
    assert_eq!(record["kind"], json!("applied"));
    assert_eq!(record["summaryKey"], json!("activity.board.addNote"));
    assert_eq!(record["targets"], json!(["note-1"]));
    assert_eq!(record["historyEntryId"], json!(history_entry_id));

    let listed = list(&host, &project, None, None).await;
    assert_eq!(
        ids(&listed),
        vec![record["id"].as_str().unwrap().to_owned()]
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn torn_activity_tail_is_ignored_and_repaired_before_the_next_append() {
    let base = temp_dir("torn");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-torn").await;

    apply_note(&host, &project, 0, "batch-torn-1", "note-1").await;
    let jsonl_path = project_dir(&host, &project).join("activity.jsonl");
    let mut contents = fs::read_to_string(&jsonl_path).expect("read activity.jsonl");
    contents.push_str(r#"{"id":"act-torn","at":"#);
    fs::write(&jsonl_path, contents).expect("append torn line");

    let listed = list(&host, &project, None, None).await;
    assert_eq!(
        records(&listed).len(),
        1,
        "a torn trailing line is not a completed record"
    );
    assert_eq!(records(&listed)[0]["targets"], json!(["note-1"]));

    // The next append terminates the torn line, so both complete records
    // survive and the torn fragment is dropped.
    apply_note(&host, &project, 1, "batch-torn-2", "note-2").await;
    let listed = list(&host, &project, None, None).await;
    let all = records(&listed);
    assert_eq!(all.len(), 2, "both complete records survive");
    assert_eq!(all[0]["targets"], json!(["note-2"]));
    assert_eq!(all[1]["targets"], json!(["note-1"]));
    let content = fs::read_to_string(&jsonl_path).expect("read repaired log");
    assert!(
        !content.contains("act-torn"),
        "the torn tail must be dropped: {content}"
    );

    let _ = fs::remove_dir_all(&base);
}
