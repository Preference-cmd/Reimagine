//! Collection service managing document sessions for collections.
//!
//! Each collection in each project has a single in-memory `DocumentSession`
//! persisted under `projects/<id>/collections/<collection id>.json`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use reimagine_config::AppPaths;
use reimagine_core::collection::{CollectionChange, CollectionCommand, CollectionDocument, Slot};
use reimagine_core::command::{CommandBatch, CommandResult, CommandStatus, OperationContext};
use reimagine_core::document::DocumentSession;
use reimagine_core::event::{Event, EventBus, Timestamp};
use reimagine_core::model::{
    AssetId, CollectionId, CommandBatchId, DocumentVersion, ProjectId, SlotId,
};
use tokio::sync::Mutex as AsyncMutex;

use crate::document_store::{read_document, write_document};
use crate::{AppHostError, AppHostResult};

type CollectionSessionHandle = Arc<AsyncMutex<DocumentSession<CollectionDocument>>>;

/// The fields of a collection document that listing needs, without its slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionSummary {
    pub id: CollectionId,
    pub name: String,
    pub version: u64,
    pub slot_count: usize,
}

impl CollectionSummary {
    fn of(document: &CollectionDocument) -> Self {
        Self {
            id: document.id().clone(),
            name: document.name().to_owned(),
            version: document.version().get(),
            slot_count: document.slots().len(),
        }
    }
}

pub struct CollectionService {
    paths: AppPaths,
    sessions: RwLock<BTreeMap<(ProjectId, CollectionId), CollectionSessionHandle>>,
    event_bus: EventBus,
}

impl std::fmt::Debug for CollectionService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let session_count = self
            .sessions
            .read()
            .map(|sessions| sessions.len())
            .unwrap_or_default();
        f.debug_struct("CollectionService")
            .field("paths", &self.paths)
            .field("session_count", &session_count)
            .finish()
    }
}

impl CollectionService {
    pub fn with_event_bus(paths: AppPaths, event_bus: EventBus) -> Self {
        Self {
            paths,
            sessions: RwLock::new(BTreeMap::new()),
            event_bus,
        }
    }

    fn collection_path(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
    ) -> std::path::PathBuf {
        self.collections_dir(project_id)
            .join(format!("{}.json", collection_id.as_str()))
    }

    fn collections_dir(&self, project_id: &ProjectId) -> std::path::PathBuf {
        self.paths
            .project_dir(project_id.as_str())
            .join("collections")
    }

    /// Summaries of every collection document in `project_id`, sorted by id.
    ///
    /// A missing `collections/` directory lists nothing. An unreadable
    /// document is skipped so one bad file does not hide the rest.
    pub async fn list(&self, project_id: &ProjectId) -> AppHostResult<Vec<CollectionSummary>> {
        let dir = self.collections_dir(project_id);
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(AppHostError::Io {
                    path: dir,
                    message: error.to_string(),
                });
            }
        };

        let mut summaries = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| AppHostError::Io {
                path: dir.clone(),
                message: error.to_string(),
            })?
        {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            match read_document::<CollectionDocument>(&path, "collection").await {
                Ok(Some(document)) => summaries.push(CollectionSummary::of(&document)),
                Ok(None) => {}
                Err(error) => tracing::warn!(
                    path = %path.display(),
                    error = %error,
                    "skipping unreadable collection document"
                ),
            }
        }
        summaries.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
        Ok(summaries)
    }

    /// Create and persist a new collection document, initializing its session.
    ///
    /// Does not publish `collection.changed` yet; the caller is responsible for
    /// emitting the event after any associated board items are created.
    pub async fn create(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
        name: impl Into<String>,
        slots: Vec<Slot>,
    ) -> AppHostResult<CollectionDocument> {
        let doc = CollectionDocument::new(
            collection_id.clone(),
            project_id.clone(),
            name,
            DocumentVersion::new(0),
            slots,
        );
        let path = self.collection_path(project_id, collection_id);
        write_document(&path, &doc, "collection").await?;

        let session = Arc::new(AsyncMutex::new(DocumentSession::new(doc.clone())));
        self.sessions
            .write()
            .expect("collection session registry poisoned")
            .insert(
                (project_id.clone(), collection_id.clone()),
                Arc::clone(&session),
            );

        Ok(doc)
    }

    /// Remove a collection document file and its in-memory session.
    pub async fn remove(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
    ) -> AppHostResult<()> {
        self.sessions
            .write()
            .expect("collection session registry poisoned")
            .remove(&(project_id.clone(), collection_id.clone()));

        let path = self.collection_path(project_id, collection_id);
        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| AppHostError::Io {
                    path,
                    message: e.to_string(),
                })?;
        }
        Ok(())
    }

    /// The session for `(project_id, collection_id)`, loading it from disk on
    /// first use. Concurrent first loads agree on one session.
    async fn session_for(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
    ) -> AppHostResult<CollectionSessionHandle> {
        let cached = self
            .sessions
            .read()
            .expect("collection session registry poisoned")
            .get(&(project_id.clone(), collection_id.clone()))
            .cloned();
        if let Some(session) = cached {
            return Ok(session);
        }

        let path = self.collection_path(project_id, collection_id);
        let doc: CollectionDocument =
            read_document(&path, "collection").await?.ok_or_else(|| {
                AppHostError::UnknownCollection {
                    project_id: project_id.clone(),
                    collection_id: collection_id.clone(),
                }
            })?;
        let session = Arc::new(AsyncMutex::new(DocumentSession::new(doc)));

        let winner = {
            let mut sessions = self
                .sessions
                .write()
                .expect("collection session registry poisoned");
            if let Some(existing) = sessions.get(&(project_id.clone(), collection_id.clone())) {
                Some(existing.clone())
            } else {
                sessions.insert(
                    (project_id.clone(), collection_id.clone()),
                    Arc::clone(&session),
                );
                None
            }
        };
        Ok(winner.unwrap_or(session))
    }

    /// Current snapshot for a collection.
    pub async fn snapshot(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
    ) -> AppHostResult<CollectionDocument> {
        let session = self.session_for(project_id, collection_id).await?;
        Ok(session.lock().await.document().clone())
    }

    /// Point one slot at `asset`, persist the collection, and emit
    /// `collection.changed` when the command applied.
    ///
    /// The base version is read while holding the session lock, so concurrent
    /// writers to different slots of one collection do not conflict.
    pub async fn set_slot_asset(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
        context: &OperationContext,
        slot_id: SlotId,
        asset: AssetId,
    ) -> AppHostResult<CommandResult<CollectionChange>> {
        let session = self.session_for(project_id, collection_id).await?;
        let mut guard = session.lock().await;
        static BATCH_COUNTER: AtomicU64 = AtomicU64::new(1);
        let count = BATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let batch = CommandBatch::new(
            CommandBatchId::new(format!("batch-set-slot-{nanos:x}-{count:x}")),
            context.clone(),
            guard.document().version(),
            Timestamp::new(chrono::Utc::now().to_rfc3339()),
            vec![CollectionCommand::SetSlotAsset { slot_id, asset }],
        );
        let result = guard.apply_batch(&(), batch);
        if result.status() == CommandStatus::Applied {
            self.save_locked(project_id, collection_id, &guard).await?;
            self.emit_changed(guard.document());
        }
        Ok(result)
    }

    /// Undo the last applied batch, persisting the restored document.
    pub async fn undo(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
    ) -> AppHostResult<Option<CommandResult<CollectionChange>>> {
        let session = self.session_for(project_id, collection_id).await?;
        let mut guard = session.lock().await;
        let Some(result) = guard.undo() else {
            return Ok(None);
        };
        self.save_locked(project_id, collection_id, &guard).await?;
        self.emit_changed(guard.document());
        Ok(Some(result))
    }

    /// Redo the last undone batch, persisting the restored document.
    pub async fn redo(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
    ) -> AppHostResult<Option<CommandResult<CollectionChange>>> {
        let session = self.session_for(project_id, collection_id).await?;
        let mut guard = session.lock().await;
        let Some(result) = guard.redo() else {
            return Ok(None);
        };
        self.save_locked(project_id, collection_id, &guard).await?;
        self.emit_changed(guard.document());
        Ok(Some(result))
    }

    /// Drop all in-memory collection sessions for a project (called on project deletion).
    pub fn remove_project(&self, project_id: &ProjectId) {
        let mut sessions = self
            .sessions
            .write()
            .expect("collection session registry poisoned");
        sessions.retain(|(p, _), _| p != project_id);
    }

    pub fn emit_changed(&self, doc: &CollectionDocument) {
        self.event_bus.publish(Event::Document {
            kind: "collection.changed".to_owned(),
            project_id: doc.project_id().to_string(),
            document_id: doc.id().to_string(),
            version: doc.version().get(),
        });
    }

    async fn save_locked(
        &self,
        project_id: &ProjectId,
        collection_id: &CollectionId,
        session: &DocumentSession<CollectionDocument>,
    ) -> AppHostResult<()> {
        let path = self.collection_path(project_id, collection_id);
        write_document(&path, session.document(), "collection").await
    }
}
