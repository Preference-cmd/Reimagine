//! Project activity log: board apply/undo/redo records, recorded op steps,
//! `activity.list` paging, the `activity.appended` event, persistence and a
//! torn `activity.jsonl` tail.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use reimagine_app_host::{
    ACTIVITY_LIST_MAX_LIMIT, ActivityKind, ActivityService, InvokeRequest, WorkspaceHost,
    WorkspaceHostBuilder, WorkspaceScope,
};
use reimagine_core::board::{
    BoardChange, BoardCommand, BoardItemKind, BoardItemPosition, BoardItemSize,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResult, CommandStatus,
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

fn human_actor() -> CommandActor {
    CommandActor::new(CommandActorKind::Human)
}

fn agent_actor(label: &str) -> CommandActor {
    CommandActor::new(CommandActorKind::Agent)
        .with_id("agent-1")
        .with_label(label)
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
    assert!(
        host.board_service()
            .undo(&project, human_actor())
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        host.board_service()
            .redo(&project, human_actor())
            .await
            .unwrap()
            .is_some()
    );

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
    // The op's own board append is the change its record must undo: the op
    // output names that history entry and the board item it touched.
    let appended_item_id = created["board_item_id"]
        .as_str()
        .expect("board item id in the op output");
    let appended_history_entry_id = created["history_entry_id"]
        .as_str()
        .expect("history entry id in the op output")
        .to_owned();
    assert_eq!(op_record["targets"], json!([appended_item_id]));
    assert_eq!(
        op_record["historyEntryId"],
        json!(appended_history_entry_id)
    );

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

    // An undo taken from the newest history entry reverses the collection
    // append the op made, not the note that preceded it.
    let undone = host
        .board_service()
        .undo(&project, human_actor())
        .await
        .unwrap()
        .expect("the collection append is the newest history entry");
    assert_eq!(
        undone.history_entry_id().map(ToString::to_string),
        Some(appended_history_entry_id.clone())
    );
    let undone_page = list(&host, &project, None, Some(1)).await;
    let undone_record = &records(&undone_page)[0];
    assert_eq!(undone_record["kind"], json!("undone"));
    assert_eq!(undone_record["targets"], json!([appended_item_id]));
    assert_eq!(
        undone_record["historyEntryId"],
        json!(appended_history_entry_id)
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn activity_records_persist_across_a_host_restart() {
    let base = temp_dir("restart");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-restart").await;

    apply_note(&host, &project, 0, "batch-persist-1", "note-1").await;
    host.board_service()
        .undo(&project, human_actor())
        .await
        .unwrap();
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

#[tokio::test]
async fn rejected_and_no_op_board_batches_append_nothing() {
    let base = temp_dir("no-append");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-no-append").await;

    apply_note(&host, &project, 0, "batch-base", "note-1").await;

    // A stale base version is rejected before anything mutates.
    let rejected = host
        .board_service()
        .apply_batch(
            &project,
            note_batch(
                DocumentVersion::new(0),
                "batch-stale",
                "note-2",
                human_context("You"),
            ),
        )
        .await
        .expect("apply returns a result");
    assert_eq!(rejected.status(), CommandStatus::Rejected);

    // Moving an item onto its current position produces no change.
    let no_op = host
        .board_service()
        .apply_batch(
            &project,
            CommandBatch::new(
                CommandBatchId::new("batch-noop"),
                human_context("You"),
                DocumentVersion::new(1),
                Timestamp::new("2026-01-01T00:00:00Z"),
                vec![BoardCommand::MoveItem {
                    item_id: BoardItemId::new("note-1"),
                    position: BoardItemPosition::new(0, 0),
                }],
            ),
        )
        .await
        .expect("apply returns a result");
    assert_eq!(no_op.status(), CommandStatus::NoOp);

    let listed = list(&host, &project, None, None).await;
    assert_eq!(
        records(&listed).len(),
        1,
        "neither a rejected nor a no-op batch appends a record"
    );
    assert_eq!(records(&listed)[0]["targets"], json!(["note-1"]));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn undo_with_empty_history_appends_nothing_and_records_the_actor() {
    let base = temp_dir("undo-empty");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-undo-empty").await;

    assert!(
        host.board_service()
            .undo(&project, human_actor())
            .await
            .unwrap()
            .is_none()
    );
    let empty = list(&host, &project, None, None).await;
    assert!(
        records(&empty).is_empty(),
        "undo with no history appends nothing"
    );

    apply_note(&host, &project, 0, "batch-undo-actor", "note-1").await;

    let undone = host
        .board_service()
        .undo(&project, agent_actor("Agent"))
        .await
        .unwrap()
        .expect("history entry exists");
    assert_eq!(undone.status(), CommandStatus::Applied);
    assert!(
        host.board_service()
            .undo(&project, human_actor())
            .await
            .unwrap()
            .is_none(),
        "the history is exhausted after one undo"
    );

    let listed = list(&host, &project, None, None).await;
    assert_eq!(
        records(&listed).len(),
        2,
        "the exhausted second undo appends nothing"
    );
    let undo_record = &records(&listed)[0];
    assert_eq!(undo_record["kind"], json!("undone"));
    assert_eq!(undo_record["actor"]["kind"], json!("Agent"));
    assert_eq!(undo_record["actor"]["id"], json!("agent-1"));
    assert_eq!(undo_record["targets"], json!(["note-1"]));
    assert_eq!(
        undo_record["historyEntryId"],
        json!(
            undone
                .history_entry_id()
                .expect("undo names its entry")
                .to_string()
        )
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn multi_item_board_batch_dedupes_targets_in_change_order() {
    let base = temp_dir("targets");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-targets").await;

    let batch = CommandBatch::new(
        CommandBatchId::new("batch-multi"),
        human_context("You"),
        DocumentVersion::new(0),
        Timestamp::new("2026-01-01T00:00:00Z"),
        vec![
            BoardCommand::AddItem {
                item_id: BoardItemId::new("item-a"),
                kind: BoardItemKind::Note {
                    content: "a".to_owned(),
                },
                position: BoardItemPosition::new(0, 0),
                size: BoardItemSize::new(100, 80),
            },
            BoardCommand::AddItem {
                item_id: BoardItemId::new("item-b"),
                kind: BoardItemKind::Note {
                    content: "b".to_owned(),
                },
                position: BoardItemPosition::new(200, 0),
                size: BoardItemSize::new(100, 80),
            },
            BoardCommand::MoveItem {
                item_id: BoardItemId::new("item-a"),
                position: BoardItemPosition::new(40, 40),
            },
        ],
    );
    let applied = host
        .board_service()
        .apply_batch(&project, batch)
        .await
        .expect("apply multi-item batch");
    assert_eq!(applied.status(), CommandStatus::Applied);
    assert_eq!(
        applied.changes().len(),
        3,
        "the last change touches item-a again"
    );

    let listed = list(&host, &project, None, None).await;
    let record = &records(&listed)[0];
    assert_eq!(
        record["targets"],
        json!(["item-a", "item-b"]),
        "targets keep first-touch order without duplicates"
    );
    assert_eq!(
        record["historyEntryId"],
        json!(
            applied
                .history_entry_id()
                .expect("applied batch names its entry")
                .to_string()
        )
    );

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn activity_logs_are_isolated_per_project() {
    let base = temp_dir("isolation");
    let host = build_host(&base).await;
    let project_a = create_project(&host, "proj-iso-a").await;
    let project_b = create_project(&host, "proj-iso-b").await;

    apply_note(&host, &project_a, 0, "batch-iso-a", "item-a").await;
    apply_note(&host, &project_b, 0, "batch-iso-b", "item-b").await;

    let listed_a = list(&host, &project_a, None, None).await;
    assert_eq!(records(&listed_a).len(), 1);
    assert_eq!(records(&listed_a)[0]["targets"], json!(["item-a"]));

    let listed_b = list(&host, &project_b, None, None).await;
    assert_eq!(records(&listed_b).len(), 1);
    assert_eq!(records(&listed_b)[0]["targets"], json!(["item-b"]));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn activity_list_clamps_the_limit_at_both_ends() {
    let base = temp_dir("limit");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-limit").await;

    // Write past the maximum page directly; the list op is what clamps.
    let writer = ActivityService::new(host.config().paths().clone(), host.event_bus().clone());
    for index in 1..=ACTIVITY_LIST_MAX_LIMIT + 5 {
        writer
            .record(
                &project,
                ActivityKind::Applied,
                human_actor(),
                format!("event {index}"),
                "activity.test",
                json!({}),
                Vec::new(),
                None,
            )
            .await
            .expect("append record");
    }

    let zero = list(&host, &project, None, Some(0)).await;
    assert_eq!(
        records(&zero).len(),
        1,
        "a limit of 0 clamps up to the 1-record minimum"
    );
    assert_eq!(zero["has_more"], json!(true));

    let huge = list(&host, &project, None, Some(500)).await;
    assert_eq!(
        records(&huge).len(),
        ACTIVITY_LIST_MAX_LIMIT,
        "a limit of 500 clamps down to the maximum page"
    );
    assert_eq!(huge["has_more"], json!(true));

    let _ = fs::remove_dir_all(&base);
}

#[tokio::test]
async fn concurrent_records_publish_in_file_order() {
    let base = temp_dir("order");
    let host = build_host(&base).await;
    let project = create_project(&host, "proj-order").await;
    let mut events = host.event_bus().subscribe();

    let activity = Arc::new(ActivityService::new(
        host.config().paths().clone(),
        host.event_bus().clone(),
    ));
    let mut tasks = Vec::new();
    for index in 0..16 {
        let activity = Arc::clone(&activity);
        let project = project.clone();
        tasks.push(tokio::spawn(async move {
            activity
                .record(
                    &project,
                    ActivityKind::Applied,
                    human_actor(),
                    format!("concurrent {index}"),
                    "activity.test",
                    json!({}),
                    Vec::new(),
                    None,
                )
                .await
                .expect("append record");
        }));
    }
    for task in tasks {
        task.await.expect("recorder finishes");
    }

    let mut published = Vec::new();
    while published.len() < 16 {
        let event = tokio::time::timeout(Duration::from_secs(5), events.recv())
            .await
            .expect("activity event arrives")
            .expect("event channel open");
        if let Event::Mod { kind, payload } = event
            && kind == "activity.appended"
        {
            published.push(
                payload["record"]["id"]
                    .as_str()
                    .expect("record id")
                    .to_owned(),
            );
        }
    }

    let (mut appended, _) = activity
        .list(&project, None, ACTIVITY_LIST_MAX_LIMIT)
        .await
        .expect("read the activity log");
    appended.reverse();
    let file_order: Vec<String> = appended.into_iter().map(|record| record.id).collect();
    assert_eq!(
        published, file_order,
        "published order matches append order"
    );

    let _ = fs::remove_dir_all(&base);
}
