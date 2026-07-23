//! Host-side run workflow orchestration.
//!
//! `run_workflow` is the single host-facing entry point that turns a
//! `WorkflowService` snapshot into either a started run (returning the
//! host-safe run handle and initial snapshot) or a `Blocked` diagnostic
//! report when readiness fails.

use std::sync::Arc;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::event::OperationReport;
use reimagine_core::model::{RunId, WorkflowId};
use reimagine_core::readiness::{ExecutionPlanResult, RunTargetSelection, build_execution_plan};
use reimagine_core::workflow::Workflow;
use reimagine_runtime::{RunHandle, RunInputs, RunSnapshot, RuntimeOptions};

use crate::{AppHostResult, WorkspaceHost};

/// Request payload for [`WorkspaceHost::run_workflow`].
///
/// `target_selection`, `run_inputs`, `options`, and `correlation_id` are
/// all host-supplied. V1 host adapters (Tauri, future Axum) construct one
/// of these per run request; Agent tools are not allowed to call
/// `run_workflow` directly.
#[derive(Debug, Clone)]
pub struct RunWorkflowRequest {
    pub workflow_id: WorkflowId,
    pub target_selection: RunTargetSelection,
    pub run_inputs: RunInputs,
    pub options: RuntimeOptions,
    pub correlation_id: Option<CorrelationId>,
}

impl RunWorkflowRequest {
    pub fn new(workflow_id: WorkflowId, target_selection: RunTargetSelection) -> Self {
        Self {
            workflow_id,
            target_selection,
            run_inputs: RunInputs::new(),
            options: RuntimeOptions::default(),
            correlation_id: None,
        }
    }

    pub fn with_run_inputs(mut self, run_inputs: RunInputs) -> Self {
        self.run_inputs = run_inputs;
        self
    }

    pub fn with_options(mut self, options: RuntimeOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.options.correlation_id = Some(correlation_id.clone());
        self.correlation_id = Some(correlation_id);
        self
    }
}

/// Outcome of [`WorkspaceHost::run_workflow`].
///
/// `Blocked` carries the readiness `OperationReport` so the host can
/// surface the diagnostics. `Started` carries the host-safe [`RunHandle`]
/// and the initial [`RunSnapshot`]; it never exposes the runtime value
/// store or backend tensor handles.
///
/// `Started` also carries the readiness `OperationReport` so the host
/// can surface non-blocking warnings (for example, the catalog/executor
/// orphan warnings produced by the alignment check). An empty report
/// means the run is fully clean.
#[derive(Debug, Clone)]
pub enum RunWorkflowResult {
    Blocked {
        report: OperationReport,
    },
    Started {
        handle: RunHandle,
        initial_snapshot: Box<RunSnapshot>,
        report: OperationReport,
    },
}

impl WorkspaceHost {
    /// Run a prepared workflow.
    ///
    /// Flow:
    /// 1. Snapshot the workflow session through `WorkflowService`.
    /// 2. Asynchronously build a `SnapshotExternalReadinessProvider`
    ///    through `ModelService` for every `ModelRef` referenced by the
    ///    workflow.
    /// 3. Synchronously call core readiness with that snapshot. If the
    ///    readiness report contains error diagnostics, return
    ///    [`RunWorkflowResult::Blocked`].
    /// 4. Hand the prepared `ExecutionPlan` to `RuntimeService::run` and
    ///    return [`RunWorkflowResult::Started`].
    pub async fn run_workflow(
        &self,
        request: RunWorkflowRequest,
    ) -> AppHostResult<RunWorkflowResult> {
        let workflow = self.workflow_service().snapshot(&request.workflow_id)?;
        let RunWorkflowRequest {
            workflow_id: _,
            target_selection,
            mut run_inputs,
            mut options,
            correlation_id,
        } = request;

        if let Some(correlation_id) = correlation_id {
            options.correlation_id = Some(correlation_id);
        }

        // Seed run inputs from the workflow's own node params so that
        // HTTP/Tauri callers do not have to resend values already
        // declared in the workflow JSON. Explicit run inputs still win.
        for node in workflow.nodes() {
            for (slot_id, value) in node.params() {
                if run_inputs.node_param(node.id(), slot_id).is_none() {
                    run_inputs.insert_node_param(node.id().clone(), slot_id.clone(), value.clone());
                }
            }
        }

        let plan_result = self.build_plan(&workflow, target_selection).await?;

        let Some(plan) = plan_result.plan() else {
            return Ok(RunWorkflowResult::Blocked {
                report: plan_result.report().clone(),
            });
        };

        let started_report = plan_result.report().clone();

        let plan = Arc::new(plan.clone());
        let handle = self.runtime_service().run(plan, run_inputs, options)?;

        let initial_snapshot = self
            .runtime_service()
            .snapshot(handle.run_id())
            .ok_or_else(|| crate::AppHostError::UnknownRun {
                run_id: handle.run_id().clone(),
            })?;

        Ok(RunWorkflowResult::Started {
            handle,
            initial_snapshot: Box::new(initial_snapshot),
            report: started_report,
        })
    }

    async fn build_plan(
        &self,
        workflow: &Workflow,
        target_selection: RunTargetSelection,
    ) -> AppHostResult<ExecutionPlanResult> {
        let provider = self
            .model_service()
            .build_readiness_snapshot(workflow)
            .await?;
        let result = build_execution_plan(
            workflow,
            self.node_catalog().as_ref(),
            target_selection,
            Some(&provider),
        );

        // Catalog / executor alignment is a workspace bootstrap invariant
        // and should always be checked, not only for workflows that
        // referenced a specific node type. Orphan executors are surfaced
        // as warnings; missing executors referenced by the workflow
        // turn the run into a `Blocked` report so callers can fix the
        // catalog before retrying.
        let alignment = self.check_node_catalog_alignment();
        let alignment_diagnostics = alignment.diagnostics();
        if alignment_diagnostics.is_empty() {
            return Ok(result);
        }

        let mut report = result.report().clone();
        for diagnostic in alignment_diagnostics {
            report.push_diagnostic(diagnostic);
        }

        let has_blocking = alignment
            .missing_executors()
            .iter()
            .any(|id| workflow_uses_node_type(workflow, id));
        if has_blocking {
            return Ok(ExecutionPlanResult::new(None, report));
        }

        let plan = result.plan().cloned();
        Ok(ExecutionPlanResult::new(plan, report))
    }
}

fn workflow_uses_node_type(
    workflow: &Workflow,
    type_id: &reimagine_core::model::NodeTypeId,
) -> bool {
    workflow
        .nodes()
        .iter()
        .any(|node| node.type_id() == type_id)
}

/// Recover the [`RunId`] of a started run, if the host already holds a
/// `RunHandle`. Convenience for adapters that only need the id.
pub fn run_id_of(result: &RunWorkflowResult) -> Option<&RunId> {
    match result {
        RunWorkflowResult::Started { handle, .. } => Some(handle.run_id()),
        RunWorkflowResult::Blocked { .. } => None,
    }
}
