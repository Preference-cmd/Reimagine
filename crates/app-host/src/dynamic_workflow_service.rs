//! Dynamic workflow documents and batch run scheduling.
//!
//! Dynamic workflows and runs are ordinary project files rewritten atomically;
//! while a run task is alive the in-memory document is authoritative. A run
//! document is never edited by hand, so no `DocumentSession` is involved.

mod execute;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

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
    AssetId, CollectionId, DiagnosticId, DynamicWorkflowId, ProjectId, RunId, SlotId,
};
use reimagine_core::task::CancelToken;
use serde_json::{Value, json};
use tokio::sync::{Mutex as AsyncMutex, Notify, OwnedMutexGuard};
use tokio::task::AbortHandle;

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

fn run_already_active(run_id: &RunId) -> Diagnostic {
    dynamic_workflow_caller_error(
        "RUN_ALREADY_ACTIVE",
        format!("run `{run_id}` already has a live task"),
    )
}

fn not_interrupted(run_id: &RunId, state: RunState) -> Diagnostic {
    dynamic_workflow_caller_error(
        "NOT_INTERRUPTED",
        format!("run `{run_id}` is `{state}`, only interrupted runs can resume"),
    )
}

fn shutting_down() -> Diagnostic {
    dynamic_workflow_caller_error(
        "SHUTTING_DOWN",
        "the host is shutting down; new dynamic workflow tasks are rejected",
    )
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
    pub(crate) abort: AbortHandle,
}

/// A held run transition guard. Dropping it releases the guard and removes
/// the run's entry once no one else holds or awaits it, so the table only
/// covers runs with a transition in flight.
struct RunTransition<'a> {
    transitions: &'a Mutex<HashMap<RunId, Arc<AsyncMutex<()>>>>,
    run_id: RunId,
    guard: Option<OwnedMutexGuard<()>>,
}

impl Drop for RunTransition<'_> {
    fn drop(&mut self) {
        // Release first, so the count below is the table's own handle plus
        // the callers still waiting for this guard.
        self.guard.take();
        let mut transitions = self
            .transitions
            .lock()
            .expect("dynamic workflow run transition registry poisoned");
        if transitions
            .get(&self.run_id)
            .is_some_and(|lock| Arc::strong_count(lock) == 1)
        {
            transitions.remove(&self.run_id);
        }
    }
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
    /// Per-run guards for the decisions that depend on "this run has no live
    /// task": normalizing an idle document and claiming it for a task. A claim
    /// holds its guard from the registry insert through the first `running`
    /// write, so no reader can observe the run as idle in between. The guards
    /// are per run, so a write for one run never stalls another.
    run_transitions: Mutex<HashMap<RunId, Arc<AsyncMutex<()>>>>,
    /// Set once the service is shutting down: no new task may be registered,
    /// even if a claim was already past its first registry check.
    shutting_down: AtomicBool,
    /// Per project, what `list_runs` needs to filter and order runs without
    /// reading every run document. A project is scanned once, on its first
    /// listing; `write_run` keeps a scanned project's entries current.
    run_index: AsyncMutex<HashMap<ProjectId, HashMap<RunId, RunSummary>>>,
}

/// The fields of a run document that `list_runs` filters and orders by.
#[derive(Debug, Clone)]
struct RunSummary {
    created_at: Timestamp,
    target_collection: Option<CollectionId>,
    created_collection: Option<CollectionId>,
}

impl RunSummary {
    fn of(run: &RunDocument) -> Self {
        Self {
            created_at: run.created_at.clone(),
            target_collection: run.target.collection_id().cloned(),
            created_collection: run.collection_id.clone(),
        }
    }

    fn touches(&self, collection_id: &CollectionId) -> bool {
        self.target_collection.as_ref() == Some(collection_id)
            || self.created_collection.as_ref() == Some(collection_id)
    }
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
            run_transitions: Mutex::new(HashMap::new()),
            shutting_down: AtomicBool::new(false),
            run_index: AsyncMutex::new(HashMap::new()),
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

                let selected: Vec<(SlotId, AssetId)> = match slots {
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
                        slots
                            .iter()
                            .filter_map(|slot_id| {
                                collection
                                    .slot(slot_id)
                                    .map(|slot| (slot_id.clone(), slot.asset.clone()))
                            })
                            .collect()
                    }
                    None => collection
                        .slots()
                        .iter()
                        .map(|slot| (slot.id.clone(), slot.asset.clone()))
                        .collect(),
                };
                // Pin each item's input now: a resume after an interruption
                // must read the asset the run started with, even if the slot
                // was changed (or undone) in the meantime.
                let items = selected
                    .into_iter()
                    .map(|(slot_id, asset)| {
                        RunItem::new(slot_id.to_string(), Some(slot_id)).with_input_asset(asset)
                    })
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

        // The task is claimed before the document is first written: a reader
        // that arrives between the two must see the registered run, not an
        // idle document it can normalize to `interrupted`.
        self.start_task(project_id.clone(), run.clone()).await?;
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

    /// Write a run document and spawn its task.
    async fn start_task(
        self: &Arc<Self>,
        project_id: ProjectId,
        run: RunDocument,
    ) -> DynamicWorkflowResult<RunState> {
        let _transition = self.lock_run_transition(&run.id).await;
        self.begin_task(project_id, run).await
    }

    /// Claim `run` in the active registry, persist its `running` state, and
    /// let its task start. Returns the claimed state. The caller holds this
    /// run's transition guard.
    ///
    /// A run that already has a live task is rejected here rather than
    /// silently replacing the registry entry: two tasks over one run would
    /// execute the same items twice and leave the first task un-abortable.
    ///
    /// The spawned task waits for `start` until the `running` state is on
    /// disk, and the registry entry is inserted with its abort handle in the
    /// same critical section, so shutdown can never drain an entry it cannot
    /// abort.
    async fn begin_task(
        self: &Arc<Self>,
        project_id: ProjectId,
        mut run: RunDocument,
    ) -> DynamicWorkflowResult<RunState> {
        let run_id = run.id.clone();
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err(shutting_down());
        }
        if self.active_run(&run_id).is_some() {
            return Err(run_already_active(&run_id));
        }

        // A registered run is running the moment it is readable: readers take
        // the in-memory document and must never see the pre-task state as an
        // interrupted run.
        run.state = RunState::Running;
        run.version = run.version.saturating_add(1);
        run.finished_at = None;

        let doc = Arc::new(AsyncMutex::new(run));
        let cancel = CancelToken::new();
        let done = Arc::new(Notify::new());
        let start = Arc::new(Notify::new());

        let service = Arc::clone(self);
        let task_project = project_id.clone();
        let task_doc = Arc::clone(&doc);
        let task_cancel = cancel.clone();
        let task_done = Arc::clone(&done);
        let task_start = Arc::clone(&start);
        let handle = tokio::spawn(async move {
            // No item may run before the claimed `running` state is durable.
            task_start.notified().await;
            service
                .execute_run(task_project, task_doc, task_cancel, task_done)
                .await;
        });
        let abort = handle.abort_handle();

        {
            let mut active = self
                .active
                .write()
                .expect("dynamic workflow run registry poisoned");
            if self.shutting_down.load(Ordering::SeqCst) {
                drop(active);
                abort.abort();
                return Err(shutting_down());
            }
            if active.contains_key(&run_id) {
                drop(active);
                abort.abort();
                return Err(run_already_active(&run_id));
            }
            active.insert(
                run_id.clone(),
                ActiveRun {
                    cancel,
                    done: Arc::clone(&done),
                    doc: Arc::clone(&doc),
                    abort,
                },
            );
        }

        let persisted = {
            let run = doc.lock().await;
            self.write_run(&project_id, &run).await
        };
        if let Err(error) = persisted {
            // Nothing will run or settle this claim, so release the registry
            // entry and wake anyone already waiting on its `done` first.
            self.fail_claim(&run_id, &doc).await;
            return Err(error);
        }
        let state = {
            let run = doc.lock().await;
            self.emit_run_changed(&project_id, &run);
            run.state
        };
        start.notify_one();
        Ok(state)
    }

    /// Drop a claim whose `running` state could not be persisted: mark the
    /// in-memory document failed, remove it from the registry, abort the
    /// still-gated task and wake cancel waiters instead of leaving them to
    /// time out.
    async fn fail_claim(&self, run_id: &RunId, doc: &Arc<AsyncMutex<RunDocument>>) {
        {
            let mut run = doc.lock().await;
            if run.state.is_active() {
                run.state = RunState::Failed;
                run.version = run.version.saturating_add(1);
                run.finished_at = Some(Timestamp::new(chrono::Utc::now().to_rfc3339()));
            }
        }
        let active = self
            .active
            .write()
            .expect("dynamic workflow run registry poisoned")
            .remove(run_id);
        if let Some(active) = active {
            active.abort.abort();
            active.done.notify_waiters();
        }
    }

    async fn write_run(
        &self,
        project_id: &ProjectId,
        run: &RunDocument,
    ) -> DynamicWorkflowResult<()> {
        // The index lock is held across the write so a first scan of this
        // project cannot miss a document that is being created.
        let mut index = self.run_index.lock().await;
        let path = self.run_path(project_id, &run.id);
        write_document(&path, run, "dynamic workflow run")
            .await
            .map_err(io_error)?;
        if let Some(runs) = index.get_mut(project_id) {
            runs.insert(run.id.clone(), RunSummary::of(run));
        }
        Ok(())
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

    /// Take the transition guard for one run. Claims and normalizations of
    /// the same run serialize on it; unrelated runs never wait for each other.
    async fn lock_run_transition(&self, run_id: &RunId) -> RunTransition<'_> {
        let lock = {
            let mut transitions = self
                .run_transitions
                .lock()
                .expect("dynamic workflow run transition registry poisoned");
            Arc::clone(
                transitions
                    .entry(run_id.clone())
                    .or_insert_with(|| Arc::new(AsyncMutex::new(()))),
            )
        };
        RunTransition {
            transitions: &self.run_transitions,
            run_id: run_id.clone(),
            guard: Some(lock.lock_owned().await),
        }
    }

    /// Read a run, preferring the in-memory copy of an active task.
    ///
    /// A queued/running document with no live task in this process belongs to
    /// a previous session: it is normalized to `interrupted` and persisted.
    /// The normalization holds only this run's transition guard, so it cannot
    /// write over a run that has just been (re)started while an unrelated run
    /// reads or claims independently.
    pub async fn get_run(
        &self,
        project_id: &ProjectId,
        run_id: &RunId,
    ) -> DynamicWorkflowResult<RunDocument> {
        if let Some(active) = self.active_run(run_id) {
            return Ok(active.doc.lock().await.clone());
        }
        let _transition = self.lock_run_transition(run_id).await;
        if let Some(active) = self.active_run(run_id) {
            return Ok(active.doc.lock().await.clone());
        }
        let mut run = self.load_run(project_id, run_id).await?;
        if !run.state.is_active() {
            return Ok(run);
        }
        run.state = RunState::Interrupted;
        run.version = run.version.saturating_add(1);
        self.write_run(project_id, &run).await?;
        self.emit_run_changed(project_id, &run);
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

    /// Runs of a project, newest first, optionally only those touching one
    /// collection (as their target or as their created collection) and at
    /// most `limit` of them. Only the returned documents are read.
    pub async fn list_runs(
        &self,
        project_id: &ProjectId,
        collection_id: Option<&CollectionId>,
        limit: Option<usize>,
    ) -> DynamicWorkflowResult<Vec<RunDocument>> {
        let mut selected: Vec<(RunId, Timestamp)> = {
            let mut index = self.run_index.lock().await;
            if !index.contains_key(project_id) {
                let scanned = self.scan_runs(project_id).await?;
                index.insert(project_id.clone(), scanned);
            }
            index[project_id]
                .iter()
                .filter(|(_, summary)| collection_id.is_none_or(|id| summary.touches(id)))
                .map(|(run_id, summary)| (run_id.clone(), summary.created_at.clone()))
                .collect()
        };
        selected.sort_by(|(a_id, a_created), (b_id, b_created)| {
            b_created
                .as_str()
                .cmp(a_created.as_str())
                .then_with(|| b_id.as_str().cmp(a_id.as_str()))
        });
        selected.truncate(limit.unwrap_or(usize::MAX));

        let mut runs = Vec::with_capacity(selected.len());
        for (run_id, _) in selected {
            match self.get_run(project_id, &run_id).await {
                Ok(run) => runs.push(run),
                Err(error) => tracing::warn!(
                    run_id = %run_id,
                    error = %error.message(),
                    "skipping unreadable dynamic workflow run document"
                ),
            }
        }
        Ok(runs)
    }

    /// Read every run document of a project once to build its index.
    async fn scan_runs(
        &self,
        project_id: &ProjectId,
    ) -> DynamicWorkflowResult<HashMap<RunId, RunSummary>> {
        let dir = self.runs_dir(project_id);
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(HashMap::new());
            }
            Err(error) => {
                return Err(dynamic_workflow_internal_error(
                    "IO",
                    format!("read run dir `{}`: {error}", dir.display()),
                ));
            }
        };

        let mut runs = HashMap::new();
        while let Some(entry) = entries.next_entry().await.map_err(|error| {
            dynamic_workflow_internal_error("IO", format!("read run dir: {error}"))
        })? {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            match read_document::<RunDocument>(&path, "dynamic workflow run").await {
                Ok(Some(run)) => {
                    runs.insert(run.id.clone(), RunSummary::of(&run));
                }
                Ok(None) => {}
                Err(error) => tracing::warn!(
                    path = %path.display(),
                    error = %error,
                    "skipping unreadable dynamic workflow run document"
                ),
            }
        }
        Ok(runs)
    }

    /// Cancel a run. Cancelling an idle run only reports its current state.
    pub async fn cancel(
        &self,
        project_id: &ProjectId,
        run_id: &RunId,
    ) -> DynamicWorkflowResult<RunState> {
        let Some(active) = self.active_run(run_id) else {
            return Ok(self.get_run(project_id, run_id).await?.state);
        };
        // The task may have settled between looking it up and waiting. A
        // completed task calls `notify_waiters`, which leaves no permit for a
        // waiter that registers afterwards, so check the state first and
        // again once the waiter is armed.
        let settled = {
            let run = active.doc.lock().await;
            (!run.state.is_active()).then_some(run.state)
        };
        if let Some(state) = settled {
            return Ok(state);
        }
        let notified = active.done.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let settled = {
            let run = active.doc.lock().await;
            (!run.state.is_active()).then_some(run.state)
        };
        if let Some(state) = settled {
            return Ok(state);
        }
        active.cancel.cancel();
        let _ = tokio::time::timeout(CANCEL_SETTLE_DEADLINE, notified).await;
        let run = active.doc.lock().await.clone();
        Ok(run.state)
    }

    /// Resume an interrupted run from the last successful step of each item.
    ///
    /// Like `run`, this returns once the resumed task is claimed; the returned
    /// state is the claimed `running` state, and progress arrives as events.
    pub async fn resume(
        self: &Arc<Self>,
        project_id: &ProjectId,
        run_id: &RunId,
    ) -> DynamicWorkflowResult<RunState> {
        let run = self.get_run(project_id, run_id).await?;
        if run.state != RunState::Interrupted {
            return Err(not_interrupted(run_id, run.state));
        }

        // Check the state again inside the claim window: between the read
        // above and here another resume may have claimed the run and even
        // finished it. Two concurrent resumes must not both start a task.
        let _transition = self.lock_run_transition(run_id).await;
        if self.active_run(run_id).is_some() {
            return Err(run_already_active(run_id));
        }
        let run = self.load_run(project_id, run_id).await?;
        if run.state != RunState::Interrupted {
            return Err(not_interrupted(run_id, run.state));
        }
        self.begin_task(project_id.clone(), run).await
    }

    /// Abort every in-flight run task. Run documents are left as they are, so
    /// the next process sees them as interrupted. No new task may be
    /// registered afterwards, including one whose claim was already in flight.
    pub fn shutdown(&self) {
        // Store the flag before draining: a claim that inserts after this
        // store re-checks it under the registry lock and aborts itself; one
        // that inserted before is drained here.
        self.shutting_down.store(true, Ordering::SeqCst);
        let mut active = self
            .active
            .write()
            .expect("dynamic workflow run registry poisoned");
        for (_, run) in active.drain() {
            run.abort.abort();
            // No task will run to its own `notify_waiters`; a cancel that
            // already holds this entry must not wait out its deadline.
            run.done.notify_waiters();
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

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::command::{CommandActor, CommandActorKind, CommandProvenance};
    use std::time::Duration;
    use tokio::task::JoinHandle;

    fn test_service() -> Arc<DynamicWorkflowService> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let paths = AppPaths::new(
            std::env::temp_dir().join(format!("reimagine-dynamic-workflow-service-{nonce}")),
        );
        Arc::new(DynamicWorkflowService::with_event_bus(
            paths.clone(),
            EventBus::new(),
            Weak::new(),
            Arc::new(CollectionService::with_event_bus(paths, EventBus::new())),
        ))
    }

    fn run_document(id: &RunId, state: RunState) -> RunDocument {
        let mut run = RunDocument::new(
            id.clone(),
            ProjectId::new("default"),
            DynamicWorkflowId::new("dwf-1"),
            WorkflowSpec {
                inputs: std::collections::BTreeMap::new(),
                params: json!({ "type": "object" }),
                steps: Vec::new(),
                output: json!({}),
            },
            RunTarget::Groups {
                groups: Vec::new(),
                name: "batch".to_owned(),
            },
            None,
            OperationContext::new(
                CommandActor::new(CommandActorKind::Human),
                CommandProvenance::Direct,
            ),
            Vec::new(),
            Timestamp::new("2026-01-01T00:00:00Z"),
        );
        run.state = state;
        run
    }

    fn register_pending(
        service: &DynamicWorkflowService,
        run_id: &RunId,
    ) -> (Arc<Notify>, JoinHandle<()>) {
        let done = Arc::new(Notify::new());
        let doc = Arc::new(AsyncMutex::new(run_document(run_id, RunState::Running)));
        let handle = tokio::spawn(std::future::pending::<()>());
        service.active.write().unwrap().insert(
            run_id.clone(),
            ActiveRun {
                cancel: CancelToken::new(),
                done: Arc::clone(&done),
                doc,
                abort: handle.abort_handle(),
            },
        );
        (done, handle)
    }

    #[tokio::test]
    async fn a_failed_claim_wakes_cancel_waiters() {
        let service = test_service();
        let run_id = RunId::new("run-1");
        let (done, handle) = register_pending(&service, &run_id);
        let doc = service.active_run(&run_id).expect("registered").doc;

        let notified = done.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();

        let result = tokio::time::timeout(Duration::from_secs(1), async {
            service.fail_claim(&run_id, &doc).await;
            notified.await;
        })
        .await;
        assert!(
            result.is_ok(),
            "a failed claim must notify the cancel waiter instead of leaving it to time out"
        );

        assert!(service.active.read().unwrap().is_empty());
        assert_eq!(doc.lock().await.state, RunState::Failed);
        assert!(handle.await.unwrap_err().is_cancelled());
    }

    #[tokio::test]
    async fn shutdown_aborts_registered_tasks_and_rejects_new_claims() {
        let service = test_service();
        let run_id = RunId::new("run-1");
        let (done, handle) = register_pending(&service, &run_id);

        let notified = done.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();

        service.shutdown();

        tokio::time::timeout(Duration::from_secs(1), notified)
            .await
            .expect("shutdown must wake a cancel waiter");
        assert!(handle.await.unwrap_err().is_cancelled());
        assert!(service.active.read().unwrap().is_empty());

        let error = service
            .begin_task(
                ProjectId::new("default"),
                run_document(&RunId::new("run-2"), RunState::Queued),
            )
            .await
            .expect_err("a claim after shutdown must be rejected");
        assert_eq!(
            error.code().as_str(),
            "DYNAMIC_WORKFLOW/CALLER_SHUTTING_DOWN"
        );
    }

    #[tokio::test]
    async fn a_run_transition_does_not_block_another_run() {
        let service = test_service();
        let project = ProjectId::new("default");
        let run_a = RunId::new("run-a");
        let run_b = RunId::new("run-b");
        service
            .write_run(&project, &run_document(&run_a, RunState::Running))
            .await
            .expect("write run a");
        service
            .write_run(&project, &run_document(&run_b, RunState::Running))
            .await
            .expect("write run b");

        // Simulate an in-flight normalization of run a.
        let guard = service.lock_run_transition(&run_a).await;

        let reader = {
            let service = Arc::clone(&service);
            let project = project.clone();
            let run_b = run_b.clone();
            tokio::spawn(async move { service.get_run(&project, &run_b).await })
        };
        let normalized = tokio::time::timeout(Duration::from_secs(1), reader)
            .await
            .expect("an unrelated run must not wait for run a's transition")
            .expect("reader task")
            .expect("normalize run b");
        assert_eq!(normalized.state, RunState::Interrupted);

        drop(guard);
        let run = service
            .get_run(&project, &run_a)
            .await
            .expect("normalize run a");
        assert_eq!(run.state, RunState::Interrupted);
        assert!(
            service.run_transitions.lock().unwrap().is_empty(),
            "released transition guards are forgotten"
        );

        let _ = std::fs::remove_dir_all(service.paths.base_path());
    }
}
