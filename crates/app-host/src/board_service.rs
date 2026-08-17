//! Board service: load/save the persisted project canvas document and
//! run board commands through `BoardSession`.
//!
//! Every board is 1:1 with a project and lives at
//! `projects/{project_id}/board.json`. Successful command application
//! saves the document atomically and emits a `board.changed` event on
//! the service's broadcast channel.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use reimagine_config::{AppPaths, atomic_write};
use reimagine_core::board::{
    BoardCommandBatch, BoardCommandResult, BoardCommandResultStatus, BoardDocument, BoardSession,
};
use reimagine_core::model::{BoardId, BoardVersion, ProjectId};
use tokio::sync::{Mutex as AsyncMutex, broadcast};

use crate::{AppHostError, AppHostResult};

/// Domain event emitted after a board document changes on disk.
///
/// The payload is deliberately flat so UI listeners never depend on
/// `reimagine-core` board internals.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoardChangedEvent {
    pub kind: String,
    pub project_id: String,
    pub board_id: String,
    pub board_version: u64,
}

pub struct BoardService {
    paths: AppPaths,
    sessions: RwLock<BTreeMap<ProjectId, Arc<AsyncMutex<BoardSession>>>>,
    events: broadcast::Sender<BoardChangedEvent>,
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
        let (events, _) = broadcast::channel(64);
        Self {
            paths,
            sessions: RwLock::new(BTreeMap::new()),
            events,
        }
    }

    /// Subscribe to `board.changed` events. Events are broadcast; a
    /// lagging subscriber is dropped by `tokio::sync::broadcast`
    /// semantics.
    pub fn subscribe(&self) -> broadcast::Receiver<BoardChangedEvent> {
        self.events.subscribe()
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
            return Ok(session.lock().await.board().clone());
        }

        let board = self.read_board_from_disk(project_id).await?;
        let session = Arc::new(AsyncMutex::new(BoardSession::new(board)));

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
            Some(existing) => Ok(existing.lock().await.board().clone()),
            None => Ok(session.lock().await.board().clone()),
        }
    }

    async fn read_board_from_disk(&self, project_id: &ProjectId) -> AppHostResult<BoardDocument> {
        let path = self.board_path(project_id);
        match tokio::fs::read(&path).await {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|error| AppHostError::Io {
                path: path.clone(),
                message: format!("invalid board document: {error}"),
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(AppHostError::UnknownBoard {
                    project_id: project_id.clone(),
                })
            }
            Err(error) => Err(AppHostError::Io {
                path,
                message: error.to_string(),
            }),
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
        batch: BoardCommandBatch,
    ) -> AppHostResult<BoardCommandResult> {
        let session = self.session_for(project_id).await?;
        Ok(session.lock().await.preview_batch(batch))
    }

    /// Apply a command batch, persist the board, and emit
    /// `board.changed` when the batch was applied.
    pub async fn apply_batch(
        &self,
        project_id: &ProjectId,
        batch: BoardCommandBatch,
    ) -> AppHostResult<BoardCommandResult> {
        let session = self.session_for(project_id).await?;
        let mut guard = session.lock().await;
        let result = guard.apply_batch(batch);
        if result.status() == BoardCommandResultStatus::Applied {
            self.save_locked(project_id, &guard).await?;
            self.emit_changed(&guard);
        }
        Ok(result)
    }

    /// Undo the last applied batch, persisting the restored document.
    pub async fn undo(&self, project_id: &ProjectId) -> AppHostResult<Option<BoardCommandResult>> {
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
    pub async fn redo(&self, project_id: &ProjectId) -> AppHostResult<Option<BoardCommandResult>> {
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
    ) -> AppHostResult<Arc<AsyncMutex<BoardSession>>> {
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

    fn emit_changed(&self, session: &BoardSession) {
        let board = session.board();
        let _ = self.events.send(BoardChangedEvent {
            kind: "board.changed".to_owned(),
            project_id: board.project_id().to_string(),
            board_id: board.id().to_string(),
            board_version: board.version().get(),
        });
    }

    async fn save_locked(
        &self,
        project_id: &ProjectId,
        session: &BoardSession,
    ) -> AppHostResult<()> {
        let path = self.board_path(project_id);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| AppHostError::Io {
                    path: parent.to_path_buf(),
                    message: error.to_string(),
                })?;
        }
        let bytes =
            serde_json::to_vec_pretty(session.board()).map_err(|error| AppHostError::Io {
                path: path.clone(),
                message: format!("failed to serialize board: {error}"),
            })?;
        atomic_write(&path, bytes)
            .await
            .map_err(|error| AppHostError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        Ok(())
    }
}

fn default_board(project_id: &ProjectId) -> BoardDocument {
    BoardDocument::new(
        BoardId::new(format!("board-{project_id}")),
        project_id.clone(),
        BoardVersion::new(0),
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
    write_board_atomic(&path, &default_board(project_id)).await
}

pub(crate) async fn write_board_atomic(path: &Path, board: &BoardDocument) -> AppHostResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| AppHostError::Io {
                path: parent.to_path_buf(),
                message: error.to_string(),
            })?;
    }
    let bytes = serde_json::to_vec_pretty(board).map_err(|error| AppHostError::Io {
        path: path.to_path_buf(),
        message: format!("failed to serialize board: {error}"),
    })?;
    atomic_write(path, bytes)
        .await
        .map_err(|error| AppHostError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    Ok(())
}
