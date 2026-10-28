//! Execution engine for dynamic workflow runs.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use serde_json::{Value, json};
use tokio::sync::{Mutex as AsyncMutex, Notify, Semaphore};
use tokio::task::JoinSet;

use reimagine_core::command::OperationContext;
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::dynamic_workflow::{
    BindingScope, RunDocument, RunState, RunTarget, SlotState, WorkflowSpec, WorkflowStep,
    resolve_value,
};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    AssetId, CollectionId, DiagnosticId, ProjectId, RunId, SlotId, StepId,
};
use reimagine_core::step::{StepOutcome, StepRecord, StepRun};
use reimagine_core::task::CancelToken;

use crate::asset_store::AssetStore;
use crate::operation_service::{InvokeRequest, OperationInfo, OperationService};
use crate::step_log::StepLog;

use super::{
    DYNAMIC_WORKFLOW_SLOT_CONCURRENCY, DynamicWorkflowService, dynamic_workflow_internal_error,
};

fn item_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("dynamic-workflow-run:{code}")),
        DiagnosticCode::new(format!("DYNAMIC_WORKFLOW/{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("dynamic_workflow_service"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("dynamic_workflow.run")),
    )
}

/// Per-item input prepared before the item tasks start.
enum ItemInput {
    Collection {
        asset: AssetId,
        slot_id: SlotId,
        collection_id: CollectionId,
    },
    Group {
        params: Value,
    },
}

/// Execution state shared by all item tasks of one run.
struct RunEnv {
    project_id: ProjectId,
    run_id: RunId,
    context: OperationContext,
    workflow: WorkflowSpec,
    steps: Vec<WorkflowStep>,
    target: RunTarget,
    params: Option<Value>,
    operations: Arc<OperationService>,
    operation_infos: Vec<OperationInfo>,
    /// Merged step records for this run, keyed by item id and step id.
    records: HashMap<String, BTreeMap<String, StepRecord>>,
    doc: Arc<AsyncMutex<RunDocument>>,
    cancel: CancelToken,
}

impl DynamicWorkflowService {
    pub(super) async fn execute_run(
        self: Arc<Self>,
        project_id: ProjectId,
        doc: Arc<AsyncMutex<RunDocument>>,
        cancel: CancelToken,
        done: Arc<Notify>,
    ) {
        let run_id = doc.lock().await.id.clone();
        if let Err(error) = self.execute_inner(&project_id, &doc, &cancel).await {
            tracing::warn!(
                run_id = %run_id,
                error = %error.message(),
                "dynamic workflow run could not finish"
            );
        }
        self.active
            .write()
            .expect("dynamic workflow run registry poisoned")
            .remove(&run_id);
        done.notify_waiters();
    }

    async fn execute_inner(
        self: &Arc<Self>,
        project_id: &ProjectId,
        doc: &Arc<AsyncMutex<RunDocument>>,
        cancel: &CancelToken,
    ) -> Result<(), Diagnostic> {
        {
            let mut run = doc.lock().await;
            if run.state != RunState::Running {
                run.state = RunState::Running;
                run.version += 1;
                self.write_run(project_id, &run).await?;
                self.emit_run_changed(project_id, &run);
            }
        }

        let (run_id, workflow, target, params, context, existing_collection) = {
            let run = doc.lock().await;
            (
                run.id.clone(),
                run.workflow.clone(),
                run.target.clone(),
                run.params.clone(),
                run.context.clone(),
                run.collection_id.clone(),
            )
        };

        let steps: Vec<WorkflowStep> = workflow
            .ordered_steps()
            .map_err(|errors| {
                errors.into_iter().next().unwrap_or_else(|| {
                    dynamic_workflow_internal_error("INVALID_WORKFLOW", "invalid step graph")
                })
            })?
            .into_iter()
            .cloned()
            .collect();

        let operations = self.operation_service().ok_or_else(|| {
            dynamic_workflow_internal_error(
                "SERVICE_GONE",
                "operation service is no longer available",
            )
        })?;
        let operation_infos = operations.list(project_id).await;

        let records = self.step_records(project_id, &run_id).await?;

        let collection = match &target {
            RunTarget::Collection { collection_id, .. } => Some(
                self.collection_service()
                    .snapshot(project_id, collection_id)
                    .await
                    .map_err(|error| {
                        dynamic_workflow_internal_error("COLLECTION", error.to_string())
                    })?,
            ),
            RunTarget::Groups { .. } => None,
        };

        let items = doc.lock().await.items.clone();
        let mut plans: Vec<(usize, Result<ItemInput, Diagnostic>)> = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let input = match (&target, &collection) {
                (RunTarget::Collection { collection_id, .. }, Some(collection)) => {
                    let diagnostic = || {
                        item_error(
                            "ITEM_UNKNOWN_SLOT",
                            format!(
                                "slot for item `{}` is no longer in the collection",
                                item.item_id
                            ),
                        )
                    };
                    match item.slot_id.as_ref().and_then(|slot_id| {
                        collection
                            .slot(slot_id)
                            .map(|slot| (slot_id.clone(), slot.asset.clone()))
                    }) {
                        Some((slot_id, asset)) => Ok(ItemInput::Collection {
                            asset,
                            slot_id,
                            collection_id: collection_id.clone(),
                        }),
                        None => Err(diagnostic()),
                    }
                }
                (RunTarget::Groups { groups, .. }, _) => match groups.get(index) {
                    Some(group) => Ok(ItemInput::Group {
                        params: group.clone(),
                    }),
                    None => Err(item_error(
                        "ITEM_UNKNOWN_GROUP",
                        format!("parameter group for item `{}` is missing", item.item_id),
                    )),
                },
                _ => Err(item_error(
                    "ITEM_UNKNOWN_SLOT",
                    "collection target with no collection snapshot",
                )),
            };
            plans.push((index, input));
        }

        let env = Arc::new(RunEnv {
            project_id: project_id.clone(),
            run_id,
            context,
            workflow,
            steps,
            target: target.clone(),
            params,
            operations,
            operation_infos,
            records,
            doc: Arc::clone(doc),
            cancel: cancel.clone(),
        });

        let mut outputs: Vec<Option<AssetId>> = vec![None; items.len()];
        let mut set = JoinSet::new();
        let slots = Arc::new(Semaphore::new(DYNAMIC_WORKFLOW_SLOT_CONCURRENCY));
        for (index, input) in plans {
            let item_state = items[index].state;
            if matches!(
                item_state,
                SlotState::Failed | SlotState::Cancelled | SlotState::NeedsConfirmation
            ) {
                continue;
            }
            if cancel.is_cancelled() && item_state != SlotState::Succeeded {
                self.update_item(project_id, doc, index, SlotState::Cancelled, None)
                    .await;
                continue;
            }
            let Ok(permit) = Arc::clone(&slots).acquire_owned().await else {
                break;
            };
            let service = Arc::clone(self);
            let env = Arc::clone(&env);
            set.spawn(async move {
                let _permit = permit;
                let output = service.execute_item(&env, index, input).await;
                (index, output)
            });
        }

        while let Some(joined) = set.join_next().await {
            match joined {
                Ok((index, output)) => outputs[index] = output,
                Err(error) => {
                    tracing::warn!(error = %error, "dynamic workflow item task failed");
                }
            }
        }

        // A parameter-group run builds its collection once every item settled;
        // failed items do not contribute an asset.
        let mut collection_id = None;
        let mut state = {
            let run = doc.lock().await;
            derive_run_state(&run.items, cancel.is_cancelled())
        };
        if let RunTarget::Groups { name, .. } = &env.target
            && !cancel.is_cancelled()
            && existing_collection.is_none()
        {
            let assets: Vec<AssetId> = outputs.iter().flatten().cloned().collect();
            if !assets.is_empty() {
                match self
                    .create_group_collection(project_id, name, &assets, &env.context)
                    .await
                {
                    Ok(id) => {
                        // Record the collection before the run settles so a
                        // later resume does not create a second one.
                        {
                            let mut run = doc.lock().await;
                            run.collection_id = Some(id.clone());
                            run.version += 1;
                            self.write_run(project_id, &run).await?;
                        }
                        collection_id = Some(id);
                    }
                    Err(error) => {
                        tracing::warn!(error = %error.message(), "collection.create failed");
                        state = RunState::Failed;
                    }
                }
            }
        }

        {
            let mut run = doc.lock().await;
            run.state = state;
            if let Some(collection_id) = collection_id {
                run.collection_id = Some(collection_id);
            }
            run.finished_at = Some(Timestamp::new(chrono::Utc::now().to_rfc3339()));
            run.version += 1;
            self.write_run(project_id, &run).await?;
            self.emit_run_changed(project_id, &run);
        }

        Ok(())
    }

    async fn step_records(
        &self,
        project_id: &ProjectId,
        run_id: &RunId,
    ) -> Result<HashMap<String, BTreeMap<String, StepRecord>>, Diagnostic> {
        let step_log = StepLog::new(self.project_directory(project_id));
        let records = step_log.records().await?;
        let mut by_item: HashMap<String, BTreeMap<String, StepRecord>> = HashMap::new();
        for record in records {
            let Some(run) = &record.run else {
                continue;
            };
            if &run.run_id != run_id {
                continue;
            }
            by_item
                .entry(run.item_id.clone())
                .or_default()
                .insert(run.step_id.clone(), record);
        }
        Ok(by_item)
    }

    async fn create_group_collection(
        &self,
        project_id: &ProjectId,
        name: &str,
        assets: &[AssetId],
        context: &OperationContext,
    ) -> Result<CollectionId, Diagnostic> {
        let operations = self.operation_service().ok_or_else(|| {
            dynamic_workflow_internal_error(
                "SERVICE_GONE",
                "operation service is no longer available",
            )
        })?;
        let result = operations
            .invoke(
                project_id,
                InvokeRequest {
                    name: "collection.create".to_owned(),
                    input: json!({
                        "name": name,
                        "assets": assets
                            .iter()
                            .map(|asset| asset.as_str())
                            .collect::<Vec<_>>(),
                    }),
                    context: context.clone(),
                    cancel: CancelToken::new(),
                    run: None,
                },
            )
            .await?;
        let collection_id = result
            .output
            .get("collection_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                dynamic_workflow_internal_error(
                    "COLLECTION_OUTPUT",
                    "collection.create returned no collection_id",
                )
            })?;
        CollectionId::try_new(collection_id)
            .map_err(|reason| dynamic_workflow_internal_error("COLLECTION_OUTPUT", reason))
    }

    async fn execute_item(
        &self,
        env: &Arc<RunEnv>,
        index: usize,
        input: Result<ItemInput, Diagnostic>,
    ) -> Option<AssetId> {
        let item = {
            let doc = env.doc.lock().await;
            doc.items[index].clone()
        };
        if matches!(
            item.state,
            SlotState::Failed | SlotState::Cancelled | SlotState::NeedsConfirmation
        ) {
            return None;
        }
        let input = match input {
            Ok(input) => input,
            Err(diagnostic) => {
                self.update_item(
                    &env.project_id,
                    &env.doc,
                    index,
                    SlotState::Failed,
                    Some(diagnostic),
                )
                .await;
                return None;
            }
        };
        if env.cancel.is_cancelled() {
            self.update_item(&env.project_id, &env.doc, index, SlotState::Cancelled, None)
                .await;
            return None;
        }

        // A slot that already succeeded before an interruption stays in that
        // state; it is only re-read to recover its output for group runs.
        if item.state != SlotState::Succeeded {
            self.update_item(&env.project_id, &env.doc, index, SlotState::Running, None)
                .await;
        }

        let (inputs, params) = match &input {
            ItemInput::Collection { asset, .. } => {
                let mut inputs = BTreeMap::new();
                if let Some(name) = env.workflow.asset_inputs().first() {
                    inputs.insert((*name).to_owned(), Value::String(asset.to_string()));
                }
                (inputs, env.params.clone().unwrap_or_else(|| json!({})))
            }
            ItemInput::Group { params } => (BTreeMap::new(), params.clone()),
        };

        let mut step_outputs: BTreeMap<String, Value> = BTreeMap::new();
        let mut produced: BTreeSet<StepId> = BTreeSet::new();
        let records = env.records.get(&item.item_id).cloned().unwrap_or_default();

        for step in &env.steps {
            if env.cancel.is_cancelled() {
                self.update_item(&env.project_id, &env.doc, index, SlotState::Cancelled, None)
                    .await;
                return None;
            }

            if let Some(record) = records.get(&step.id) {
                match &record.outcome {
                    StepOutcome::Succeeded { output } => {
                        step_outputs.insert(step.id.clone(), output.clone());
                        produced.insert(record.id.clone());
                        continue;
                    }
                    StepOutcome::Failed { diagnostic } => {
                        self.update_item(
                            &env.project_id,
                            &env.doc,
                            index,
                            SlotState::Failed,
                            Some((**diagnostic).clone()),
                        )
                        .await;
                        return None;
                    }
                    StepOutcome::Cancelled => {
                        self.update_item(
                            &env.project_id,
                            &env.doc,
                            index,
                            SlotState::Cancelled,
                            None,
                        )
                        .await;
                        return None;
                    }
                    StepOutcome::Running => {
                        if !self.interrupted_step_may_rerun(&env.operation_infos, &step.op) {
                            self.update_item(
                                &env.project_id,
                                &env.doc,
                                index,
                                SlotState::NeedsConfirmation,
                                Some(item_error(
                                    "NEEDS_CONFIRMATION",
                                    format!(
                                        "operation `{}` was interrupted and may have external effects",
                                        step.op
                                    ),
                                )),
                            )
                            .await;
                            return None;
                        }
                    }
                }
            }

            let scope = BindingScope {
                inputs: &inputs,
                params: &params,
                steps: &step_outputs,
            };
            let call_input = match resolve_value(&step.input, &scope) {
                Ok(input) => input,
                Err(error) => {
                    self.update_item(
                        &env.project_id,
                        &env.doc,
                        index,
                        SlotState::Failed,
                        Some(item_error("BINDING", error.to_string())),
                    )
                    .await;
                    return None;
                }
            };

            let request = InvokeRequest {
                name: step.op.clone(),
                input: call_input,
                context: env.context.clone(),
                cancel: env.cancel.clone(),
                run: Some(StepRun {
                    run_id: env.run_id.clone(),
                    item_id: item.item_id.clone(),
                    step_id: step.id.clone(),
                }),
            };
            match env.operations.invoke(&env.project_id, request).await {
                Ok(result) => {
                    step_outputs.insert(step.id.clone(), result.output);
                    produced.insert(result.step.id);
                }
                Err(diagnostic) => {
                    if env.cancel.is_cancelled() {
                        self.update_item(
                            &env.project_id,
                            &env.doc,
                            index,
                            SlotState::Cancelled,
                            None,
                        )
                        .await;
                    } else {
                        self.update_item(
                            &env.project_id,
                            &env.doc,
                            index,
                            SlotState::Failed,
                            Some(diagnostic),
                        )
                        .await;
                    }
                    return None;
                }
            }
        }

        let scope = BindingScope {
            inputs: &inputs,
            params: &params,
            steps: &step_outputs,
        };
        let output = match resolve_value(&env.workflow.output, &scope) {
            Ok(output) => output,
            Err(error) => {
                self.update_item(
                    &env.project_id,
                    &env.doc,
                    index,
                    SlotState::Failed,
                    Some(item_error("OUTPUT_BINDING", error.to_string())),
                )
                .await;
                return None;
            }
        };
        let Some(asset) = output.as_str().and_then(|raw| AssetId::try_new(raw).ok()) else {
            self.update_item(
                &env.project_id,
                &env.doc,
                index,
                SlotState::Failed,
                Some(item_error(
                    "OUTPUT_NOT_ASSET",
                    format!("output binding resolved to `{output}`, not an asset id"),
                )),
            )
            .await;
            return None;
        };

        let asset_store = AssetStore::new(self.project_directory(&env.project_id));
        let record = match asset_store.get_asset(&asset).await {
            Ok(Some(record)) => record,
            Ok(None) => {
                self.update_item(
                    &env.project_id,
                    &env.doc,
                    index,
                    SlotState::Failed,
                    Some(item_error(
                        "OUTPUT_ASSET_MISSING",
                        format!("output asset `{asset}` is not in the project"),
                    )),
                )
                .await;
                return None;
            }
            Err(error) => {
                self.update_item(
                    &env.project_id,
                    &env.doc,
                    index,
                    SlotState::Failed,
                    Some(error),
                )
                .await;
                return None;
            }
        };
        if !produced.contains(&record.origin) {
            self.update_item(
                &env.project_id,
                &env.doc,
                index,
                SlotState::Failed,
                Some(item_error(
                    "OUTPUT_NOT_PRODUCED",
                    format!("output asset `{asset}` was not produced by this run's steps"),
                )),
            )
            .await;
            return None;
        }

        if let ItemInput::Collection {
            slot_id,
            collection_id,
            ..
        } = &input
        {
            // A slot that already succeeded before an interruption already
            // points at its result; only resolve it again for the run's
            // bookkeeping.
            if item.state != SlotState::Succeeded
                && let Err(error) = self
                    .collection_service()
                    .set_slot_asset(
                        &env.project_id,
                        collection_id,
                        &env.context,
                        slot_id.clone(),
                        asset.clone(),
                    )
                    .await
            {
                self.update_item(
                    &env.project_id,
                    &env.doc,
                    index,
                    SlotState::Failed,
                    Some(item_error("SET_SLOT", error.to_string())),
                )
                .await;
                return None;
            }
        }

        self.update_item(&env.project_id, &env.doc, index, SlotState::Succeeded, None)
            .await;
        Some(asset)
    }

    fn interrupted_step_may_rerun(&self, infos: &[OperationInfo], op: &str) -> bool {
        match infos.iter().find(|info| info.spec.name == op) {
            Some(info) if info.health.is_available() => {
                !info.spec.annotations.open_world || info.spec.annotations.idempotent
            }
            // Unknown or unhealthy operations cannot be replayed safely.
            _ => false,
        }
    }

    async fn update_item(
        &self,
        project_id: &ProjectId,
        doc: &Arc<AsyncMutex<RunDocument>>,
        index: usize,
        state: SlotState,
        diagnostic: Option<Diagnostic>,
    ) {
        let mut run = doc.lock().await;
        let Some(item) = run.items.get_mut(index) else {
            return;
        };
        if item.state == state && item.diagnostic == diagnostic {
            return;
        }
        item.state = state;
        item.diagnostic = diagnostic;
        run.version += 1;
        if let Err(error) = self.write_run(project_id, &run).await {
            tracing::warn!(error = %error.message(), "failed to persist run item state");
            return;
        }
        self.emit_slot_changed(project_id, &run);
    }
}

fn derive_run_state(
    items: &[reimagine_core::dynamic_workflow::RunItem],
    cancelled: bool,
) -> RunState {
    if cancelled {
        return RunState::Cancelled;
    }
    if items.iter().any(|item| item.state == SlotState::Failed) {
        return RunState::Failed;
    }
    if items
        .iter()
        .any(|item| item.state == SlotState::NeedsConfirmation)
    {
        return RunState::NeedsConfirmation;
    }
    if items.iter().all(|item| item.state == SlotState::Succeeded) {
        return RunState::Completed;
    }
    RunState::Failed
}
