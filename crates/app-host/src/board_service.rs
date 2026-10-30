//! Board service: load/save the persisted project canvas document and
//! run board commands through `DocumentSession`.
//!
//! Every board is 1:1 with a project and lives at
//! `projects/{project_id}/board.json`. Successful command application
//! saves the document atomically and emits one `board.changed` event on
//! the shared workspace document event bus.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use reimagine_config::AppPaths;
use reimagine_core::board::{
    BoardChange, BoardCommand, BoardDocument, BoardItemKind, BoardItemPosition, BoardItemSize,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandResult, CommandStatus, OperationContext,
};
use reimagine_core::document::DocumentSession;
use reimagine_core::event::{Event, EventBus, Timestamp};
use reimagine_core::model::{
    BoardId, BoardItemId, CommandBatchId, DocumentVersion, HistoryEntryId, ProjectId,
};
use serde_json::json;
use tokio::sync::Mutex as AsyncMutex;

use crate::activity::{ActivityKind, ActivityService};
use crate::document_store::{read_document, write_document};
use crate::{AppHostError, AppHostResult};

pub struct BoardService {
    paths: AppPaths,
    sessions: RwLock<BTreeMap<ProjectId, Arc<AsyncMutex<DocumentSession<BoardDocument>>>>>,
    event_bus: EventBus,
    activity: Option<Arc<ActivityService>>,
}

impl std::fmt::Debug for BoardService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let session_count = self
            .sessions
            .read()
            .map(|sessions| sessions.len())
            .unwrap_or_default();
        f.debug_struct("BoardService")
            .field("paths", &self.paths)
            .field("session_count", &session_count)
            .field("activity", &self.activity.is_some())
            .finish()
    }
}

impl BoardService {
    pub fn new(paths: AppPaths) -> Self {
        Self::with_event_bus(paths, EventBus::new())
    }

    pub fn with_event_bus(paths: AppPaths, event_bus: EventBus) -> Self {
        Self::with_event_bus_and_activity(paths, event_bus, None)
    }

    pub fn with_event_bus_and_activity(
        paths: AppPaths,
        event_bus: EventBus,
        activity: Option<Arc<ActivityService>>,
    ) -> Self {
        Self {
            paths,
            sessions: RwLock::new(BTreeMap::new()),
            event_bus,
            activity,
        }
    }

    pub fn board_path(&self, project_id: &ProjectId) -> std::path::PathBuf {
        self.paths
            .project_dir(project_id.as_str())
            .join("board.json")
    }

    /// Load the board for `project_id`. The document is created by
    /// project creation (`ensure_board_file`); asking for a project
    /// without a board file yields [`AppHostError::UnknownBoard`]. The
    /// returned document is the current in-memory snapshot shared by
    /// this service.
    pub async fn load_or_create(&self, project_id: &ProjectId) -> AppHostResult<BoardDocument> {
        let cached = self
            .sessions
            .read()
            .expect("board session registry poisoned")
            .get(project_id)
            .cloned();
        if let Some(session) = cached {
            return Ok(session.lock().await.document().clone());
        }

        let path = self.board_path(project_id);
        let board: BoardDocument =
            read_document(&path, "board")
                .await?
                .ok_or_else(|| AppHostError::UnknownBoard {
                    project_id: project_id.clone(),
                })?;
        let session = Arc::new(AsyncMutex::new(DocumentSession::new(board)));

        // Double-check: another task may have created the session while
        // we read from disk. No registry guard is held across the
        // awaited session lock below.
        let winner = {
            let mut sessions = self
                .sessions
                .write()
                .expect("board session registry poisoned");
            if let Some(existing) = sessions.get(project_id) {
                Some(existing.clone())
            } else {
                sessions.insert(project_id.clone(), Arc::clone(&session));
                None
            }
        };
        match winner {
            Some(existing) => Ok(existing.lock().await.document().clone()),
            None => Ok(session.lock().await.document().clone()),
        }
    }

    /// Current snapshot for a project. Unknown projects yield
    /// [`AppHostError::UnknownBoard`].
    pub async fn snapshot(&self, project_id: &ProjectId) -> AppHostResult<BoardDocument> {
        self.session_for(project_id).await?;
        self.load_or_create(project_id).await
    }

    /// Preview a command batch without mutating the board.
    pub async fn preview_batch(
        &self,
        project_id: &ProjectId,
        batch: CommandBatch<BoardCommand>,
    ) -> AppHostResult<CommandResult<BoardChange>> {
        let session = self.session_for(project_id).await?;
        Ok(session.lock().await.preview_batch(&(), batch))
    }

    /// Apply a command batch, persist the board, and emit
    /// `board.changed` when the batch was applied. An applied batch also
    /// appends one `activity.appended` record.
    pub async fn apply_batch(
        &self,
        project_id: &ProjectId,
        batch: CommandBatch<BoardCommand>,
    ) -> AppHostResult<CommandResult<BoardChange>> {
        let session = self.session_for(project_id).await?;
        let mut guard = session.lock().await;
        let actor = batch.context().actor().clone();
        let (summary_key, summary) = apply_summary(batch.commands());
        let command_count = batch.commands().len();
        let result = guard.apply_batch(&(), batch);
        if result.status() == CommandStatus::Applied {
            self.save_locked(project_id, &guard).await?;
            self.emit_changed(&guard);
            self.record_board_activity(
                project_id,
                ActivityKind::Applied,
                actor,
                summary_key,
                summary,
                json!({ "count": command_count }),
                result.changes(),
                result.history_entry_id().cloned(),
            )
            .await;
        }
        Ok(result)
    }

    /// Append a single item to the board under session lock.
    ///
    /// Reads the current document version while holding the session lock so
    /// concurrent additions on the same board do not conflict on version.
    pub async fn append_item(
        &self,
        project_id: &ProjectId,
        context: &OperationContext,
        item_id: BoardItemId,
        kind: BoardItemKind,
        position: BoardItemPosition,
        size: BoardItemSize,
    ) -> AppHostResult<CommandResult<BoardChange>> {
        let session = self.session_for(project_id).await?;
        let mut guard = session.lock().await;
        let base_version = guard.document().version();
        let batch_id = CommandBatchId::new(format!("batch-add-item-{}", item_id.as_str()));
        let batch = CommandBatch::new(
            batch_id,
            context.clone(),
            base_version,
            Timestamp::new(chrono::Utc::now().to_rfc3339()),
            vec![BoardCommand::AddItem {
                item_id,
                kind,
                position,
                size,
            }],
        );
        let result = guard.apply_batch(&(), batch);
        if result.status() == CommandStatus::Applied {
            self.save_locked(project_id, &guard).await?;
            self.emit_changed(&guard);
        }
        Ok(result)
    }

    /// Undo the last applied batch, persisting the restored document.
    pub async fn undo(
        &self,
        project_id: &ProjectId,
    ) -> AppHostResult<Option<CommandResult<BoardChange>>> {
        let session = self.session_for(project_id).await?;
        let mut guard = session.lock().await;
        let Some(result) = guard.undo() else {
            return Ok(None);
        };
        self.save_locked(project_id, &guard).await?;
        self.emit_changed(&guard);
        self.record_board_activity(
            project_id,
            ActivityKind::Undone,
            CommandActor::new(CommandActorKind::Human),
            "activity.board.undo",
            "Undid the last board change",
            json!({}),
            result.changes(),
            result.history_entry_id().cloned(),
        )
        .await;
        Ok(Some(result))
    }

    /// Redo the last undone batch, persisting the restored document.
    pub async fn redo(
        &self,
        project_id: &ProjectId,
    ) -> AppHostResult<Option<CommandResult<BoardChange>>> {
        let session = self.session_for(project_id).await?;
        let mut guard = session.lock().await;
        let Some(result) = guard.redo() else {
            return Ok(None);
        };
        self.save_locked(project_id, &guard).await?;
        self.emit_changed(&guard);
        self.record_board_activity(
            project_id,
            ActivityKind::Redone,
            CommandActor::new(CommandActorKind::Human),
            "activity.board.redo",
            "Redid the last board change",
            json!({}),
            result.changes(),
            result.history_entry_id().cloned(),
        )
        .await;
        Ok(Some(result))
    }

    /// Drop the in-memory session for a project (project deletion).
    pub fn remove_project(&self, project_id: &ProjectId) {
        self.sessions
            .write()
            .expect("board session registry poisoned")
            .remove(project_id);
    }

    async fn session_for(
        &self,
        project_id: &ProjectId,
    ) -> AppHostResult<Arc<AsyncMutex<DocumentSession<BoardDocument>>>> {
        self.load_or_create(project_id).await?;
        self.sessions
            .read()
            .expect("board session registry poisoned")
            .get(project_id)
            .cloned()
            .ok_or_else(|| AppHostError::UnknownBoard {
                project_id: project_id.clone(),
            })
    }

    fn emit_changed(&self, session: &DocumentSession<BoardDocument>) {
        let board = session.document();
        self.event_bus.publish(Event::Document {
            kind: "board.changed".to_owned(),
            project_id: board.project_id().to_string(),
            document_id: board.id().to_string(),
            version: board.version().get(),
        });
    }

    /// Append an activity record, never failing the board operation: a broken
    /// activity log must not undo an already persisted board change.
    #[allow(clippy::too_many_arguments)]
    async fn record_board_activity(
        &self,
        project_id: &ProjectId,
        kind: ActivityKind,
        actor: CommandActor,
        summary_key: impl Into<String>,
        summary: impl Into<String>,
        summary_params: serde_json::Value,
        changes: &[BoardChange],
        history_entry_id: Option<HistoryEntryId>,
    ) {
        let Some(activity) = &self.activity else {
            return;
        };

        // One entry per touched item, in change order.
        let mut targets: Vec<String> = Vec::new();
        for change in changes {
            let item_id = change.item_id().to_string();
            if !targets.contains(&item_id) {
                targets.push(item_id);
            }
        }

        if let Err(error) = activity
            .record(
                project_id,
                kind,
                actor,
                summary,
                summary_key,
                summary_params,
                targets,
                history_entry_id,
            )
            .await
        {
            tracing::warn!(
                project_id = %project_id,
                ?kind,
                %error,
                "failed to append board activity record"
            );
        }
    }

    async fn save_locked(
        &self,
        project_id: &ProjectId,
        session: &DocumentSession<BoardDocument>,
    ) -> AppHostResult<()> {
        let path = self.board_path(project_id);
        write_document(&path, session.document(), "board").await
    }
}

/// Summary key and English fallback for an applied board batch.
fn apply_summary(commands: &[BoardCommand]) -> (&'static str, String) {
    let count = commands.len();
    let all_adds = commands
        .iter()
        .all(|command| matches!(command, BoardCommand::AddItem { .. }));
    let all_notes = commands.iter().all(|command| {
        matches!(
            command,
            BoardCommand::AddItem {
                kind: BoardItemKind::Note { .. },
                ..
            }
        )
    });

    if all_adds && all_notes {
        (
            "activity.board.addNote",
            match count {
                1 => "Added a note".to_owned(),
                _ => format!("Added {count} notes"),
            },
        )
    } else if all_adds {
        (
            "activity.board.addItem",
            match count {
                1 => "Added an item".to_owned(),
                _ => format!("Added {count} items"),
            },
        )
    } else {
        (
            "activity.board.apply",
            match count {
                1 => "Applied a board change".to_owned(),
                _ => format!("Applied {count} board changes"),
            },
        )
    }
}

fn default_board(project_id: &ProjectId) -> BoardDocument {
    BoardDocument::new(
        BoardId::new(format!("board-{project_id}")),
        project_id.clone(),
        DocumentVersion::new(0),
    )
}

/// Ensure the board file exists on disk without loading a session.
/// Used by project creation so a fresh project always has its canvas
/// document from the first write.
pub(crate) async fn ensure_board_file(
    paths: &AppPaths,
    project_id: &ProjectId,
) -> AppHostResult<()> {
    let path = paths.project_dir(project_id.as_str()).join("board.json");
    if path.exists() {
        return Ok(());
    }
    write_document(&path, &default_board(project_id), "board").await
}
