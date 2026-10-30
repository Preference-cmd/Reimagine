//! Kernel operations for dynamic workflows and their batch runs:
//! - `dynamic_workflow.create`
//! - `dynamic_workflow.get`
//! - `dynamic_workflow.list`
//! - `dynamic_workflow.run`
//! - `dynamic_workflow.cancel`
//! - `dynamic_workflow.resume`
//! - `dynamic_workflow.get_run`
//! - `dynamic_workflow.list_runs`

use std::sync::Arc;

use async_trait::async_trait;
use mod_sdk::{OperationAnnotations, OperationSpec};
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::dynamic_workflow::{RunTarget, WorkflowSpec};
use reimagine_core::model::{CollectionId, DiagnosticId, DynamicWorkflowId, RunId};
use serde_json::{Value, json};

use crate::dynamic_workflow_service::DynamicWorkflowService;
use crate::operation_service::{KernelCall, KernelOp, OperationService};

fn dynamic_workflow_caller_error(code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("op:dynamic_workflow:{code}")),
        DiagnosticCode::new(format!("DYNAMIC_WORKFLOW/CALLER_{code}")),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("dynamic_workflow_ops"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("op.dynamic_workflow")),
    )
}

#[allow(clippy::result_large_err)]
fn parse_workflow_id(call: &KernelCall) -> Result<DynamicWorkflowId, Diagnostic> {
    let raw = call
        .caller_input
        .get("dynamic_workflow_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            dynamic_workflow_caller_error("INVALID_INPUT", "missing 'dynamic_workflow_id'")
        })?;
    DynamicWorkflowId::try_new(raw)
        .map_err(|reason| dynamic_workflow_caller_error("INVALID_DYNAMIC_WORKFLOW_ID", reason))
}

#[allow(clippy::result_large_err)]
fn parse_run_id(call: &KernelCall) -> Result<RunId, Diagnostic> {
    let raw = call
        .caller_input
        .get("run_id")
        .and_then(Value::as_str)
        .ok_or_else(|| dynamic_workflow_caller_error("INVALID_INPUT", "missing 'run_id'"))?;
    RunId::try_new(raw).map_err(|reason| dynamic_workflow_caller_error("INVALID_RUN_ID", reason))
}

/// `dynamic_workflow.create`
pub struct DynamicWorkflowCreateOp {
    spec: OperationSpec,
    service: Arc<DynamicWorkflowService>,
}

impl DynamicWorkflowCreateOp {
    pub fn new(service: Arc<DynamicWorkflowService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "dynamic_workflow.create".to_owned(),
                description: "Save a dynamic workflow document".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "workflow": { "type": "object" }
                    },
                    "required": ["name", "workflow"],
                    "additionalProperties": false,
                    "examples": [{
                        "name": "Normalize and outline",
                        "workflow": {
                            "inputs": { "doc": { "type": "string", "format": "asset" } },
                            "params": { "type": "object" },
                            "steps": [],
                            "output": { "$from": "inputs.doc" }
                        }
                    }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "dynamic_workflow_id": { "type": "string" }
                    },
                    "required": ["dynamic_workflow_id"],
                    "additionalProperties": false,
                    "examples": [{ "dynamic_workflow_id": "dwf-1" }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    // Every call allocates a fresh dynamic workflow id.
                    deterministic: false,
                    open_world: false,
                },
            },
            service,
        }
    }
}

#[async_trait]
impl KernelOp for DynamicWorkflowCreateOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    fn changes_documents(&self) -> bool {
        true
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let name = call
            .caller_input
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| dynamic_workflow_caller_error("INVALID_INPUT", "missing 'name'"))?;
        let workflow = call
            .caller_input
            .get("workflow")
            .ok_or_else(|| dynamic_workflow_caller_error("INVALID_INPUT", "missing 'workflow'"))?;
        let workflow: WorkflowSpec = serde_json::from_value(workflow.clone()).map_err(|error| {
            dynamic_workflow_caller_error(
                "INVALID_WORKFLOW",
                format!("invalid workflow document: {error}"),
            )
        })?;

        let document = self
            .service
            .create(&call.project_id, name.to_owned(), workflow)
            .await?;
        Ok(json!({ "dynamic_workflow_id": document.id.as_str() }))
    }
}

/// `dynamic_workflow.get`
pub struct DynamicWorkflowGetOp {
    spec: OperationSpec,
    service: Arc<DynamicWorkflowService>,
}

impl DynamicWorkflowGetOp {
    pub fn new(service: Arc<DynamicWorkflowService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "dynamic_workflow.get".to_owned(),
                description: "Read a saved dynamic workflow document".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "dynamic_workflow_id": { "type": "string" }
                    },
                    "required": ["dynamic_workflow_id"],
                    "additionalProperties": false,
                    "examples": [{ "dynamic_workflow_id": "dwf-1" }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "name": { "type": "string" },
                        "version": { "type": "integer", "minimum": 1 },
                        "workflow": { "type": "object" }
                    },
                    "required": ["id", "name", "version", "workflow"],
                    "additionalProperties": false,
                    "examples": [{
                        "id": "dwf-1",
                        "name": "Normalize",
                        "version": 1,
                        "workflow": {}
                    }]
                }),
                annotations: OperationAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    deterministic: true,
                    open_world: false,
                },
            },
            service,
        }
    }
}

#[async_trait]
impl KernelOp for DynamicWorkflowGetOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let id = parse_workflow_id(&call)?;
        let document = self
            .service
            .get(&call.project_id, &id)
            .await?
            .ok_or_else(|| {
                dynamic_workflow_caller_error(
                    "UNKNOWN_DYNAMIC_WORKFLOW",
                    format!("dynamic workflow `{id}` not found"),
                )
            })?;
        Ok(json!({
            "id": document.id.as_str(),
            "name": document.name,
            "version": document.version,
            "workflow": document.workflow,
        }))
    }
}

/// `dynamic_workflow.list`
pub struct DynamicWorkflowListOp {
    spec: OperationSpec,
    service: Arc<DynamicWorkflowService>,
}

impl DynamicWorkflowListOp {
    pub fn new(service: Arc<DynamicWorkflowService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "dynamic_workflow.list".to_owned(),
                description: "List the project's dynamic workflow documents".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "workflows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "string" },
                                    "name": { "type": "string" },
                                    "version": { "type": "integer", "minimum": 1 },
                                    "step_count": { "type": "integer", "minimum": 0 }
                                },
                                "required": ["id", "name", "version", "step_count"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["workflows"],
                    "additionalProperties": false,
                    "examples": [{ "workflows": [] }]
                }),
                annotations: OperationAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    deterministic: true,
                    open_world: false,
                },
            },
            service,
        }
    }
}

#[async_trait]
impl KernelOp for DynamicWorkflowListOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let summaries = self.service.list(&call.project_id).await?;
        let workflows: Vec<Value> = summaries
            .iter()
            .map(|summary| {
                json!({
                    "id": summary.id.as_str(),
                    "name": summary.name.as_str(),
                    "version": summary.version,
                    "step_count": summary.step_count,
                })
            })
            .collect();
        Ok(json!({ "workflows": workflows }))
    }
}

/// `dynamic_workflow.run`
pub struct DynamicWorkflowRunOp {
    spec: OperationSpec,
    service: Arc<DynamicWorkflowService>,
}

impl DynamicWorkflowRunOp {
    pub fn new(service: Arc<DynamicWorkflowService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "dynamic_workflow.run".to_owned(),
                description: "Run a dynamic workflow over collection slots or parameter groups"
                    .to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "dynamic_workflow_id": { "type": "string" },
                        "target": { "type": "object" },
                        "params": { "type": "object" }
                    },
                    "required": ["dynamic_workflow_id", "target"],
                    "additionalProperties": false,
                    "examples": [{
                        "dynamic_workflow_id": "dwf-1",
                        "target": { "collection_id": "col-1" },
                        "params": {}
                    }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "run_id": { "type": "string" }
                    },
                    "required": ["run_id"],
                    "additionalProperties": false,
                    "examples": [{ "run_id": "run-1" }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    deterministic: false,
                    open_world: false,
                },
            },
            service,
        }
    }
}

#[async_trait]
impl KernelOp for DynamicWorkflowRunOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    fn changes_documents(&self) -> bool {
        true
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let id = parse_workflow_id(&call)?;
        let target = call
            .caller_input
            .get("target")
            .ok_or_else(|| dynamic_workflow_caller_error("INVALID_INPUT", "missing 'target'"))?;
        let target: RunTarget = serde_json::from_value(target.clone()).map_err(|error| {
            dynamic_workflow_caller_error("INVALID_TARGET", format!("invalid target: {error}"))
        })?;
        let params = call.caller_input.get("params").cloned();

        let run = self
            .service
            .run(&call.project_id, &id, target, params, call.context.clone())
            .await?;
        Ok(json!({ "run_id": run.id.as_str() }))
    }
}

/// `dynamic_workflow.cancel`
pub struct DynamicWorkflowCancelOp {
    spec: OperationSpec,
    service: Arc<DynamicWorkflowService>,
}

impl DynamicWorkflowCancelOp {
    pub fn new(service: Arc<DynamicWorkflowService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "dynamic_workflow.cancel".to_owned(),
                description: "Cancel a dynamic workflow run".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "run_id": { "type": "string" }
                    },
                    "required": ["run_id"],
                    "additionalProperties": false,
                    "examples": [{ "run_id": "run-1" }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "state": { "type": "string" }
                    },
                    "required": ["state"],
                    "additionalProperties": false,
                    "examples": [{ "state": "cancelled" }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: true,
                    deterministic: true,
                    open_world: false,
                },
            },
            service,
        }
    }
}

#[async_trait]
impl KernelOp for DynamicWorkflowCancelOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    fn changes_documents(&self) -> bool {
        true
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let run_id = parse_run_id(&call)?;
        let state = self.service.cancel(&call.project_id, &run_id).await?;
        Ok(json!({ "state": state.as_str() }))
    }
}

/// `dynamic_workflow.resume`
pub struct DynamicWorkflowResumeOp {
    spec: OperationSpec,
    service: Arc<DynamicWorkflowService>,
}

impl DynamicWorkflowResumeOp {
    pub fn new(service: Arc<DynamicWorkflowService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "dynamic_workflow.resume".to_owned(),
                description: "Resume an interrupted dynamic workflow run in the background; \
                              progress arrives as run events"
                    .to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "run_id": { "type": "string" }
                    },
                    "required": ["run_id"],
                    "additionalProperties": false,
                    "examples": [{ "run_id": "run-1" }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "state": { "type": "string" }
                    },
                    "required": ["state"],
                    "additionalProperties": false,
                    "examples": [{ "state": "running" }]
                }),
                annotations: OperationAnnotations {
                    read_only: false,
                    destructive: false,
                    idempotent: false,
                    deterministic: false,
                    open_world: false,
                },
            },
            service,
        }
    }
}

#[async_trait]
impl KernelOp for DynamicWorkflowResumeOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    fn changes_documents(&self) -> bool {
        true
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let run_id = parse_run_id(&call)?;
        let state = self.service.resume(&call.project_id, &run_id).await?;
        Ok(json!({ "state": state.as_str() }))
    }
}

/// `dynamic_workflow.get_run`
pub struct DynamicWorkflowGetRunOp {
    spec: OperationSpec,
    service: Arc<DynamicWorkflowService>,
}

impl DynamicWorkflowGetRunOp {
    pub fn new(service: Arc<DynamicWorkflowService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "dynamic_workflow.get_run".to_owned(),
                description: "Read one dynamic workflow run document".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "run_id": { "type": "string" }
                    },
                    "required": ["run_id"],
                    "additionalProperties": false,
                    "examples": [{ "run_id": "run-1" }]
                }),
                output_schema: json!({ "type": "object" }),
                annotations: OperationAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    deterministic: true,
                    open_world: false,
                },
            },
            service,
        }
    }
}

#[async_trait]
impl KernelOp for DynamicWorkflowGetRunOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let run_id = parse_run_id(&call)?;
        let run = self.service.get_run(&call.project_id, &run_id).await?;
        serde_json::to_value(run).map_err(|error| {
            dynamic_workflow_caller_error(
                "INVALID_RUN",
                format!("failed to serialize run document: {error}"),
            )
        })
    }
}

/// `dynamic_workflow.list_runs`
pub struct DynamicWorkflowListRunsOp {
    spec: OperationSpec,
    service: Arc<DynamicWorkflowService>,
}

impl DynamicWorkflowListRunsOp {
    pub fn new(service: Arc<DynamicWorkflowService>) -> Self {
        Self {
            spec: OperationSpec {
                name: "dynamic_workflow.list_runs".to_owned(),
                description: "List dynamic workflow runs, newest first".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "collection_id": { "type": "string" },
                        "limit": { "type": "integer", "minimum": 1 }
                    },
                    "additionalProperties": false,
                    "examples": [{ "collection_id": "col-1", "limit": 1 }]
                }),
                output_schema: json!({
                    "type": "object",
                    "properties": {
                        "runs": { "type": "array" }
                    },
                    "required": ["runs"],
                    "additionalProperties": false,
                    "examples": [{ "runs": [] }]
                }),
                annotations: OperationAnnotations {
                    read_only: true,
                    destructive: false,
                    idempotent: true,
                    deterministic: true,
                    open_world: false,
                },
            },
            service,
        }
    }
}

#[async_trait]
impl KernelOp for DynamicWorkflowListRunsOp {
    fn spec(&self) -> &OperationSpec {
        &self.spec
    }

    async fn invoke(&self, call: KernelCall) -> Result<Value, Diagnostic> {
        let collection_id = match call.caller_input.get("collection_id") {
            Some(Value::String(raw)) => Some(CollectionId::try_new(raw).map_err(|reason| {
                dynamic_workflow_caller_error("INVALID_COLLECTION_ID", reason)
            })?),
            Some(Value::Null) | None => None,
            Some(_) => {
                return Err(dynamic_workflow_caller_error(
                    "INVALID_INPUT",
                    "'collection_id' must be a string",
                ));
            }
        };
        // The input schema already bounds `limit` to a positive integer.
        let limit = call
            .caller_input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|limit| usize::try_from(limit).unwrap_or(usize::MAX));
        let runs = self
            .service
            .list_runs(&call.project_id, collection_id.as_ref(), limit)
            .await?;
        let runs: Vec<Value> = runs
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<_, _>>()
            .map_err(|error| {
                dynamic_workflow_caller_error(
                    "INVALID_RUN",
                    format!("failed to serialize run document: {error}"),
                )
            })?;
        Ok(json!({ "runs": runs }))
    }
}

/// Register all dynamic workflow operations.
pub fn register_dynamic_workflow_ops(
    op_service: &OperationService,
    service: Arc<DynamicWorkflowService>,
) {
    op_service.register_kernel_op(Arc::new(DynamicWorkflowCreateOp::new(Arc::clone(&service))));
    op_service.register_kernel_op(Arc::new(DynamicWorkflowGetOp::new(Arc::clone(&service))));
    op_service.register_kernel_op(Arc::new(DynamicWorkflowListOp::new(Arc::clone(&service))));
    op_service.register_kernel_op(Arc::new(DynamicWorkflowRunOp::new(Arc::clone(&service))));
    op_service.register_kernel_op(Arc::new(DynamicWorkflowCancelOp::new(Arc::clone(&service))));
    op_service.register_kernel_op(Arc::new(DynamicWorkflowResumeOp::new(Arc::clone(&service))));
    op_service.register_kernel_op(Arc::new(DynamicWorkflowGetRunOp::new(Arc::clone(&service))));
    op_service.register_kernel_op(Arc::new(DynamicWorkflowListRunsOp::new(service)));
}
