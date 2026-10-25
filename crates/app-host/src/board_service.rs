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
use reimagine_core::board::{BoardChange, BoardCommand, BoardDocument};
use reimagine_core::command::{CommandBatch, CommandResult, CommandStatus};
use reimagine_core::document::DocumentSession;
use reimagine_core::model::{BoardId, DocumentVersion, ProjectId};
use tokio::sync::Mutex as AsyncMutex;

use crate::document_store::{read_document, write_document};
use crate::{AppHostError, AppHostResult};
use reimagine_core::event::{Event, EventBus};

pub struct BoardService {
    paths: AppPaths,
    sessions: RwLock<BTreeMap<ProjectId, Arc<AsyncMutex<DocumentSession<BoardDocument>>>>>,
    event_bus: EventBus,
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
            .finish()
    }
}

impl BoardService {
    pub fn new(paths: AppPaths) -> Self {
        Self::with_event_bus(paths, EventBus::new())
    }

    pub fn with_event_bus(paths: AppPaths, event_bus: EventBus) -> Self {
        Self {
            paths,
            sessions: RwLock::new(BTreeMap::new()),
            event_bus,
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
    /// `board.changed` when the batch was applied.
    pub async fn apply_batch(
        &self,
        project_id: &ProjectId,
        batch: CommandBatch<BoardCommand>,
    ) -> AppHostResult<CommandResult<BoardChange>> {
        let session = self.session_for(project_id).await?;
        let mut guard = session.lock().await;
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

    async fn save_locked(
        &self,
        project_id: &ProjectId,
        session: &DocumentSession<BoardDocument>,
    ) -> AppHostResult<()> {
        let path = self.board_path(project_id);
        write_document(&path, session.document(), "board").await
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
