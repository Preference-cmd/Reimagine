//! `POST /workflows/open` and `POST /workflows/:id/run` routes.
//!
//! Both routes are thin: they shape HTTP requests into app-host facade
//! calls and shape app-host results back into HTTP responses. They
//! must not reimplement workflow validation, readiness projection, or
//! run orchestration.

use axum::Json;
use axum::extract::{Path, State};
use reimagine_app_host::{RunWorkflowRequest, RunWorkflowResult};
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::WorkflowId;
use reimagine_core::readiness::RunTargetSelection;
use reimagine_core::workflow::Workflow;
use tracing::Span;

use crate::dto::{
    OpenWorkflowRequest, OpenWorkflowResponse, RunWorkflowRequestDto, RunWorkflowResponse,
    WorkflowSource,
};
use crate::error::{AxumHostError, AxumHostResult};
use crate::state::AxumHostState;

/// `POST /workflows/open`.
///
/// V1 accepts either:
/// - `{ "id": "<workflow_id>" }` — load the JSON from the workspace's
///   `workflows_dir` and register it under that id.
/// - `{ "workflow": { ... } }` — parse the inline JSON document,
///   register it under the id declared in the document, and return
///   that id.
/// - `{}` — neither id nor inline body; the request is rejected with
///   `400 Bad Request`.
///
/// Idempotency: when the requested id is already registered, the
/// route returns the existing id with `source: "existing"` so the
/// client can continue without a duplicate registration error.
pub async fn open(
    State(state): State<AxumHostState>,
    Json(body): Json<OpenWorkflowRequest>,
) -> AxumHostResult<Json<OpenWorkflowResponse>> {
    let OpenWorkflowRequest { id, workflow } = body;
    match (id, workflow) {
        (Some(id), None) => {
            if state.workspace().workflow_service().contains(&id) {
                return Ok(Json(OpenWorkflowResponse {
                    workflow_id: id,
                    source: WorkflowSource::Existing,
                }));
            }
            // The host facade loads the JSON from disk and registers
            // it; we surface any IO / JSON failure as a host error.
            let resolved = state
                .workspace()
                .workflow_service()
                .load_workflow(&id)
                .await?;
            Ok(Json(OpenWorkflowResponse {
                workflow_id: resolved,
                source: WorkflowSource::Disk,
            }))
        }
        (None, Some(value)) => {
            let workflow: Workflow =
                serde_json::from_value(value).map_err(|err| AxumHostError::BadRequest {
                    message: format!("invalid workflow JSON: {err}"),
                })?;
            // `WorkflowService::register_workflow` always uses the
            // workflow's own id as the registry key, so the resolved
            // id is by construction equal to `workflow.id()`. The
            // route still asserts the invariant as a guard against
            // future facade changes.
            let resolved = state
                .workspace()
                .workflow_service()
                .register_workflow(workflow.clone());
            debug_assert_eq!(resolved, *workflow.id());
            Ok(Json(OpenWorkflowResponse {
                workflow_id: workflow.id().clone(),
                source: WorkflowSource::Inline,
            }))
        }
        (Some(_), Some(_)) => Err(AxumHostError::BadRequest {
            message: "specify exactly one of `id` or `workflow`, not both".to_string(),
        }),
        (None, None) => Err(AxumHostError::BadRequest {
            message: "must specify `id` or `workflow`".to_string(),
        }),
    }
}

/// `POST /workflows/:id/run`.
///
/// Bridges to `WorkspaceHost::run_workflow`. Returns the
/// `RunWorkflowResponse` DTO so the client can branch on `outcome`
/// without parsing opaque payloads.
pub async fn run(
    State(state): State<AxumHostState>,
    Path(id): Path<String>,
    body: Option<Json<RunWorkflowRequestDto>>,
) -> AxumHostResult<Json<RunWorkflowResponse>> {
    let workflow_id = WorkflowId::new(id);
    Span::current().record("workflow_id", workflow_id.as_str());
    if !state.workspace().workflow_service().contains(&workflow_id) {
        return Err(AxumHostError::UnknownWorkflow {
            workflow_id: workflow_id.clone(),
        });
    }

    let body = body.map(|Json(b)| b).unwrap_or_default();
    let target_selection: RunTargetSelection = body
        .target_selection
        .map(Into::into)
        .unwrap_or(RunTargetSelection::AllDefaultTargets);

    let correlation_id_for_log = body.correlation_id.clone();
    if let Some(ref correlation_id) = correlation_id_for_log {
        Span::current().record("correlation_id", correlation_id.as_str());
    }

    let mut request = RunWorkflowRequest::new(workflow_id.clone(), target_selection);
    if let Some(correlation_id) = body.correlation_id {
        request = request.with_correlation_id(CorrelationId::new(correlation_id));
    }

    let result = state.workspace().run_workflow(request).await?;
    let response = match result {
        RunWorkflowResult::Started {
            handle,
            initial_snapshot,
            report,
        } => {
            let run_id = handle.run_id().clone();
            Span::current().record("run_id", run_id.as_str());
            tracing::info!(
                run_id = %run_id,
                workflow_id = %workflow_id,
                correlation_id = ?correlation_id_for_log,
                "workflow run started",
            );
            RunWorkflowResponse::Started {
                run_id,
                workflow_id: handle.workflow_id().clone(),
                workflow_version: handle.workflow_version(),
                initial_snapshot: Box::new((*initial_snapshot).into()),
                diagnostics: report
                    .diagnostics()
                    .iter()
                    .map(|d| d.clone().into())
                    .collect(),
            }
        }
        RunWorkflowResult::Blocked { report } => {
            tracing::info!(
                workflow_id = %workflow_id,
                correlation_id = ?correlation_id_for_log,
                "workflow run blocked",
            );
            RunWorkflowResponse::Blocked {
                workflow_id,
                diagnostics: report
                    .diagnostics()
                    .iter()
                    .map(|d| d.clone().into())
                    .collect(),
            }
        }
    };
    Ok(Json(response))
}
