//! Dynamic workflow documents and batch run scheduling.
//!
//! Dynamic workflows and runs are ordinary project files rewritten atomically;
//! while a run task is alive the in-memory document is authoritative. A run
//! document is never edited by hand, so no `DocumentSession` is involved.

mod execute;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock, Weak};

use reimagine_config::AppPaths;
use reimagine_core::command::OperationContext;
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::dynamic_workflow::{
    DynamicWorkflowDocument, RunDocument, RunItem, RunState, RunTarget, WorkflowSpec,
};
use reimagine_core::event::{Event, EventBus, Timestamp};
use reimagine_core::model::{
    CollectionId, DiagnosticId, DynamicWorkflowId, ProjectId, RunId, SlotId,
};
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};
use tokio::sync::{Mutex as AsyncMutex, Notify};
use tokio::task::{AbortHandle, JoinHandle};

use crate::collection_service::CollectionService;
use crate::document_store::{read_document, write_document};
use crate::operation_service::OperationService;

/// How many collection slots / parameter groups one run executes at once.
pub const DYNAMIC_WORKFLOW_SLOT_CONCURRENCY: usize = 4;

/// How long a cancel waits for the running task to settle before returning
/// the state it can see.
const CANCEL_SETTLE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// Result type for dynamic workflow calls: failures are caller-facing
/// diagnostics, the same shape the operations return.
pub type DynamicWorkflowResult<T> = Result<T, Diagnostic>;

fn dynamic_workflow_caller_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("op:dynamic_workflow:{code}")),
        DiagnosticCode::new(format!("DYNAMIC_WORKFLOW/CALLER_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("dynamic_workflow_service"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("op.dynamic_workflow")),
    )
}

fn dynamic_workflow_internal_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("op:dynamic_workflow:{code}")),
        DiagnosticCode::new(format!("DYNAMIC_WORKFLOW/INTERNAL_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("dynamic_workflow_service"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("op.dynamic_workflow")),
    )
}

fn io_error(error: crate::AppHostError) -> Diagnostic {
    dynamic_workflow_internal_error("IO", error.to_string())
}

fn first_validation_error(errors: Vec<Diagnostic>) -> Diagnostic {
    errors.into_iter().next().unwrap_or_else(|| {
        dynamic_workflow_internal_error("INVALID_WORKFLOW", "workflow validation failed")
    })
}

/// State of one in-flight run, shared between its task and readers.
#[derive(Clone)]
pub(crate) struct ActiveRun {
    pub(crate) cancel: CancelToken,
    pub(crate) done: Arc<Notify>,
    pub(crate) doc: Arc<AsyncMutex<RunDocument>>,
    pub(crate) abort: Option<AbortHandle>,
}

/// Owns dynamic workflow documents and schedules their batch runs.
pub struct DynamicWorkflowService {
    paths: AppPaths,
    event_bus: EventBus,
    /// Weak by design: the operation service registry holds the kernel ops,
    /// which hold this service, so a strong edge back would form a cycle.
    operations: Weak<OperationService>,
    collections: Arc<CollectionService>,
    active: RwLock<HashMap<RunId, ActiveRun>>,
}

impl std::fmt::Debug for DynamicWorkflowService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let active = self
            .active
            .read()
            .map(|active| active.len())
            .unwrap_or_default();
        f.debug_struct("DynamicWorkflowService")
            .field("paths", &self.paths)
            .field("active_runs", &active)
            .finish_non_exhaustive()
    }
}

impl DynamicWorkflowService {
    pub fn with_event_bus(
        paths: AppPaths,
        event_bus: EventBus,
        operations: Weak<OperationService>,
        collections: Arc<CollectionService>,
    ) -> Self {
        Self {
            paths,
            event_bus,
            operations,
            collections,
            active: RwLock::new(HashMap::new()),
        }
    }

    fn project_dir(&self, project_id: &ProjectId) -> PathBuf {
        self.paths.project_dir(project_id.as_str())
    }

    fn workflow_path(&self, project_id: &ProjectId, id: &DynamicWorkflowId) -> PathBuf {
        self.project_dir(project_id)
            .join("dynamic_workflows")
            .join(format!("{}.json", id.as_str()))
    }

    fn run_path(&self, project_id: &ProjectId, run_id: &RunId) -> PathBuf {
        self.project_dir(project_id)
            .join("dynamic_workflow_runs")
            .join(format!("{}.json", run_id.as_str()))
    }

    fn runs_dir(&self, project_id: &ProjectId) -> PathBuf {
        self.project_dir(project_id).join("dynamic_workflow_runs")
    }

    /// Validate and persist a new dynamic workflow document.
    pub async fn create(
        &self,
        project_id: &ProjectId,
        name: String,
        workflow: WorkflowSpec,
    ) -> DynamicWorkflowResult<DynamicWorkflowDocument> {
        if name.trim().is_empty() {
            return Err(dynamic_workflow_caller_error(
                "INVALID_INPUT",
                "dynamic workflow name must not be empty",
            ));
        }
        workflow.validate().map_err(first_validation_error)?;

        static ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let count = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let id = DynamicWorkflowId::new(format!("dwf-{nanos:x}-{count:x}"));
        let document = DynamicWorkflowDocument::new(id.clone(), project_id.clone(), name, workflow);
        let path = self.workflow_path(project_id, &id);
        write_document(&path, &document, "dynamic workflow")
            .await
            .map_err(io_error)?;
        Ok(document)
    }

    /// Load one dynamic workflow document.
    pub async fn get(
        &self,
        project_id: &ProjectId,
        id: &DynamicWorkflowId,
    ) -> DynamicWorkflowResult<Option<DynamicWorkflowDocument>> {
        let path = self.workflow_path(project_id, id);
        read_document(&path, "dynamic workflow")
            .await
            .map_err(io_error)
    }

    /// Create a run document and start executing it in the background.
    pub async fn run(
        self: &Arc<Self>,
        project_id: &ProjectId,
        dynamic_workflow_id: &DynamicWorkflowId,
        target: RunTarget,
        params: Option<Value>,
        context: OperationContext,
    ) -> DynamicWorkflowResult<RunDocument> {
        let document = self
            .get(project_id, dynamic_workflow_id)
            .await?
            .ok_or_else(|| {
                dynamic_workflow_caller_error(
                    "UNKNOWN_DYNAMIC_WORKFLOW",
                    format!("dynamic workflow `{dynamic_workflow_id}` not found"),
                )
            })?;
        document
            .workflow
            .validate()
            .map_err(first_validation_error)?;
        self.build_run_document(project_id, document, target, params, context)
            .await
    }

    async fn build_run_document(
        self: &Arc<Self>,
        project_id: &ProjectId,
        document: DynamicWorkflowDocument,
        target: RunTarget,
        params: Option<Value>,
        context: OperationContext,
    ) -> DynamicWorkflowResult<RunDocument> {
        let workflow = document.workflow.clone();
        workflow
            .validate_target(&target)
            .map_err(first_validation_error)?;

        let (items, stored_params) = match &target {
            RunTarget::Collection {
                collection_id,
                slots,
            } => {
                let params = params.unwrap_or_else(|| json!({}));
                if !params.is_object() {
                    return Err(dynamic_workflow_caller_error(
                        "INVALID_INPUT",
                        "params must be a JSON object for a collection target",
                    ));
                }
                Self::validate_params(&workflow.params, &params)?;

                let collection = self
                    .collections
                    .snapshot(project_id, collection_id)
                    .await
                    .map_err(|error| match error {
                        crate::AppHostError::UnknownCollection { .. } => {
                            dynamic_workflow_caller_error(
                                "UNKNOWN_COLLECTION",
                                format!("collection `{collection_id}` not found"),
                            )
                        }
                        other => dynamic_workflow_internal_error("COLLECTION", other.to_string()),
                    })?;

                let selected: Vec<SlotId> = match slots {
                    Some(slots) => {
                        let mut seen = std::collections::BTreeSet::new();
                        for slot_id in slots {
                            if !seen.insert(slot_id.clone()) {
                                return Err(dynamic_workflow_caller_error(
                                    "DUPLICATE_SLOT",
                                    format!("slot `{slot_id}` appears more than once"),
                                ));
                            }
                            if collection.slot(slot_id).is_none() {
                                return Err(dynamic_workflow_caller_error(
                                    "UNKNOWN_SLOT",
                                    format!(
                                        "slot `{slot_id}` not found in collection `{collection_id}`"
                                    ),
                                ));
                            }
                        }
                        slots.clone()
                    }
                    None => collection
                        .slots()
                        .iter()
                        .map(|slot| slot.id.clone())
                        .collect(),
                };
                let items = selected
                    .into_iter()
                    .map(|slot_id| RunItem::new(slot_id.to_string(), Some(slot_id)))
                    .collect();
                (items, Some(params))
            }
            RunTarget::Groups { groups, .. } => {
                for (index, group) in groups.iter().enumerate() {
                    Self::validate_params(&workflow.params, group).map_err(|error| {
                        dynamic_workflow_caller_error(
                            "INVALID_PARAMS",
                            format!(
                                "parameter group {} is invalid: {}",
                                index + 1,
                                error.message()
                            ),
                        )
                    })?;
                }
                let items = (0..groups.len())
                    .map(|index| RunItem::new(format!("group-{}", index + 1), None))
                    .collect();
                (items, None)
            }
        };

        static ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let count = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let run_id = RunId::new(format!("run-{nanos:x}-{count:x}"));
        let run = RunDocument::new(
            run_id,
            project_id.clone(),
            document.id,
            workflow,
            target,
            stored_params,
            context,
            items,
            Timestamp::new(chrono::Utc::now().to_rfc3339()),
        );

        self.write_run(project_id, &run).await?;
        self.start_task(project_id.clone(), run.clone())
            .await
            .map(|_| ())?;
        Ok(run)
    }

    #[allow(clippy::result_large_err)]
    fn validate_params(schema: &Value, value: &Value) -> DynamicWorkflowResult<()> {
        let validator = jsonschema::validator_for(schema).map_err(|error| {
            dynamic_workflow_internal_error(
                "INVALID_PARAMS_SCHEMA",
                format!("invalid params schema: {error}"),
            )
        })?;
        if let Some(violations) = mod_sdk::schema::violations(&validator, value) {
            return Err(dynamic_workflow_caller_error(
                "INVALID_PARAMS",
                format!("parameter value validation failed: {violations}"),
            ));
        }
        Ok(())
    }

    /// Write a run document and spawn its task. The returned handle is for
    /// callers that want to wait (resume); `run` drops it and returns early.
    async fn start_task(
        self: &Arc<Self>,
        project_id: ProjectId,
        run: RunDocument,
    ) -> DynamicWorkflowResult<JoinHandle<()>> {
        let run_id = run.id.clone();
        let doc = Arc::new(AsyncMutex::new(run));
        let cancel = CancelToken::new();
        let done = Arc::new(Notify::new());
        self.active
            .write()
            .expect("dynamic workflow run registry poisoned")
            .insert(
                run_id.clone(),
                ActiveRun {
                    cancel: cancel.clone(),
                    done: done.clone(),
                    doc: Arc::clone(&doc),
                    abort: None,
                },
            );

        let service = Arc::clone(self);
        let handle = tokio::spawn(async move {
            service.execute_run(project_id, doc, cancel, done).await;
        });
        if let Some(active) = self
            .active
            .write()
            .expect("dynamic workflow run registry poisoned")
            .get_mut(&run_id)
        {
            active.abort = Some(handle.abort_handle());
        }
        Ok(handle)
    }

    async fn write_run(
        &self,
        project_id: &ProjectId,
        run: &RunDocument,
    ) -> DynamicWorkflowResult<()> {
        let path = self.run_path(project_id, &run.id);
        write_document(&path, run, "dynamic workflow run")
            .await
            .map_err(io_error)
    }

    fn emit(&self, kind: &str, project_id: &ProjectId, run: &RunDocument) {
        self.event_bus.publish(Event::Document {
            kind: kind.to_owned(),
            project_id: project_id.to_string(),
            document_id: run.id.to_string(),
            version: run.version,
        });
    }

    pub(crate) fn emit_slot_changed(&self, project_id: &ProjectId, run: &RunDocument) {
        self.emit("dynamic_workflow.slot_changed", project_id, run);
    }

    pub(crate) fn emit_run_changed(&self, project_id: &ProjectId, run: &RunDocument) {
        self.emit("dynamic_workflow.run_changed", project_id, run);
    }

    fn active_run(&self, run_id: &RunId) -> Option<ActiveRun> {
        self.active
            .read()
            .expect("dynamic workflow run registry poisoned")
            .get(run_id)
            .cloned()
    }

    /// Read a run, preferring the in-memory copy of an active task.
    pub async fn get_run(
        &self,
        project_id: &ProjectId,
        run_id: &RunId,
    ) -> DynamicWorkflowResult<RunDocument> {
        if let Some(active) = self.active_run(run_id) {
            return Ok(active.doc.lock().await.clone());
        }
        let run = self.load_run(project_id, run_id).await?;
        if run.state.is_active() {
            return self.mark_interrupted(project_id, run).await;
        }
        Ok(run)
    }

    async fn load_run(
        &self,
        project_id: &ProjectId,
        run_id: &RunId,
    ) -> DynamicWorkflowResult<RunDocument> {
        let path = self.run_path(project_id, run_id);
        read_document(&path, "dynamic workflow run")
            .await
            .map_err(io_error)?
            .ok_or_else(|| {
                dynamic_workflow_caller_error(
                    "UNKNOWN_RUN",
                    format!("dynamic workflow run `{run_id}` not found"),
                )
            })
    }

    /// A queued/running document with no live task in this process belongs to
    /// a previous session; it is interrupted, not still running.
    async fn mark_interrupted(
        &self,
        project_id: &ProjectId,
        mut run: RunDocument,
    ) -> DynamicWorkflowResult<RunDocument> {
        run.state = RunState::Interrupted;
        run.version += 1;
        self.write_run(project_id, &run).await?;
        self.emit_run_changed(project_id, &run);
        Ok(run)
    }

    /// Runs of a project, newest first, optionally only those touching one
    /// collection (as their target or as their created collection).
    pub async fn list_runs(
        &self,
        project_id: &ProjectId,
        collection_id: Option<&CollectionId>,
    ) -> DynamicWorkflowResult<Vec<RunDocument>> {
        let dir = self.runs_dir(project_id);
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(dynamic_workflow_internal_error(
                    "IO",
                    format!("read run dir `{}`: {error}", dir.display()),
                ));
            }
        };

        let mut paths = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(|error| {
            dynamic_workflow_internal_error("IO", format!("read run dir: {error}"))
        })? {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
                paths.push(path);
            }
        }

        let mut runs = Vec::new();
        for path in paths {
            let Some(run_id) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| RunId::try_new(stem).ok())
            else {
                continue;
            };
            let run = match self.get_run(project_id, &run_id).await {
                Ok(run) => run,
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %error.message(),
                        "skipping unreadable dynamic workflow run document"
                    );
                    continue;
                }
            };
            if let Some(collection_id) = collection_id
                && run.target.collection_id() != Some(collection_id)
                && run.collection_id.as_ref() != Some(collection_id)
            {
                continue;
            }
            runs.push(run);
        }
        runs.sort_by(|a, b| {
            b.created_at
                .as_str()
                .cmp(a.created_at.as_str())
                .then_with(|| b.id.as_str().cmp(a.id.as_str()))
        });
        Ok(runs)
    }

    /// Cancel a run. Cancelling an idle run only reports its current state.
    pub async fn cancel(
        &self,
        project_id: &ProjectId,
        run_id: &RunId,
    ) -> DynamicWorkflowResult<RunState> {
        if let Some(active) = self.active_run(run_id) {
            let notified = active.done.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            active.cancel.cancel();
            let _ = tokio::time::timeout(CANCEL_SETTLE_DEADLINE, notified).await;
            let run = active.doc.lock().await.clone();
            return Ok(run.state);
        }
        Ok(self.get_run(project_id, run_id).await?.state)
    }

    /// Resume an interrupted run from the last successful step of each item.
    ///
    /// Resuming runs synchronously so the returned state describes the
    /// resumed execution, unlike `run` which returns after scheduling.
    pub async fn resume(
        self: &Arc<Self>,
        project_id: &ProjectId,
        run_id: &RunId,
    ) -> DynamicWorkflowResult<RunState> {
        let mut run = self.get_run(project_id, run_id).await?;
        if run.state != RunState::Interrupted {
            return Err(dynamic_workflow_caller_error(
                "NOT_INTERRUPTED",
                format!(
                    "run `{run_id}` is `{}`, only interrupted runs can resume",
                    run.state
                ),
            ));
        }
        run.state = RunState::Running;
        run.version += 1;
        self.write_run(project_id, &run).await?;
        self.emit_run_changed(project_id, &run);

        let handle = self.start_task(project_id.clone(), run).await?;
        if let Err(error) = handle.await {
            return Err(dynamic_workflow_internal_error(
                "RESUME_TASK",
                format!("resumed run task failed: {error}"),
            ));
        }
        Ok(self.get_run(project_id, run_id).await?.state)
    }

    /// Abort every in-flight run task. Run documents are left as they are, so
    /// the next process sees them as interrupted.
    pub fn shutdown(&self) {
        let mut active = self
            .active
            .write()
            .expect("dynamic workflow run registry poisoned");
        for (_, run) in active.drain() {
            if let Some(abort) = run.abort {
                abort.abort();
            }
        }
    }

    pub(crate) fn operation_service(&self) -> Option<Arc<OperationService>> {
        self.operations.upgrade()
    }

    pub(crate) fn project_directory(&self, project_id: &ProjectId) -> PathBuf {
        self.project_dir(project_id)
    }

    pub(crate) fn collection_service(&self) -> &Arc<CollectionService> {
        &self.collections
    }
}
