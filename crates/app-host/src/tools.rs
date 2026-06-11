use std::sync::Arc;

use async_trait::async_trait;

use reimagine_agent::{
    AgentMode, AgentTool, AgentToolRegistry, ToolContext, ToolError, ToolErrorCode, ToolInput,
    ToolName, ToolResult, ToolSpec,
};
use reimagine_agent::{ToolPermission, ToolRiskLevel};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResult,
    CommandResultStatus,
};
use reimagine_core::model::{ProposalId, WorkflowId};

use crate::AppHostError;
use crate::policy::WorkflowCommandPolicy;
use crate::proposal::{ProposalReceipt, ProposalStatus};
use crate::services::WorkspaceServices;

// ------------------------------------------------------------------
// Tool registration entrypoint
// ------------------------------------------------------------------

/// Register all V1 app-host agent tools into `registry`.
///
/// Tools capture `Arc<WorkspaceServices>` directly and verify the
/// incoming `ToolContext.workspace_scope` matches before doing work.
pub fn register_app_tools(registry: &mut AgentToolRegistry, services: Arc<WorkspaceServices>) {
    registry
        .register_arc(Arc::new(WorkflowGetTool::new(Arc::clone(&services))))
        .expect("duplicate tool registration in app-host built-ins");
    registry
        .register_arc(Arc::new(WorkflowPreviewCommandsTool::new(Arc::clone(
            &services,
        ))))
        .expect("duplicate tool registration in app-host built-ins");
    registry
        .register_arc(Arc::new(WorkflowProposeCommandsTool::new(Arc::clone(
            &services,
        ))))
        .expect("duplicate tool registration in app-host built-ins");
    registry
        .register_arc(Arc::new(WorkflowApplyCommandsTool::new(Arc::clone(
            &services,
        ))))
        .expect("duplicate tool registration in app-host built-ins");
    registry
        .register_arc(Arc::new(ModelListTool::new(Arc::clone(&services))))
        .expect("duplicate tool registration in app-host built-ins");
    registry
        .register_arc(Arc::new(ModelResolveRefTool::new(Arc::clone(&services))))
        .expect("duplicate tool registration in app-host built-ins");
    registry
        .register_arc(Arc::new(DiagnosticsForWorkflowTool::new(Arc::clone(
            &services,
        ))))
        .expect("duplicate tool registration in app-host built-ins");
}

// ------------------------------------------------------------------
// Workspace-scope guard
// ------------------------------------------------------------------

fn verify_workspace_scope(
    services: &WorkspaceServices,
    ctx: &ToolContext,
    tool_name: &str,
) -> ToolResult<()> {
    if services.workspace_scope() != ctx.workspace_scope() {
        return Err(ToolError::new(
            ToolErrorCode::WorkspaceMismatch,
            format!(
                "tool `{tool_name}` was invoked with workspace `{}` but is bound to `{}`",
                ctx.workspace_scope().as_str(),
                services.workspace_scope().as_str(),
            ),
        )
        .with_tool(ToolName::new(tool_name)));
    }
    Ok(())
}

// ------------------------------------------------------------------
// workflow.get
// ------------------------------------------------------------------

struct WorkflowGetTool {
    services: Arc<WorkspaceServices>,
}

impl WorkflowGetTool {
    fn new(services: Arc<WorkspaceServices>) -> Self {
        Self { services }
    }
}

#[async_trait]
impl AgentTool for WorkflowGetTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new("workflow.get"),
            "Get the current snapshot of a workflow.",
            [AgentMode::Agent, AgentMode::Build],
            ToolPermission::new("workflow.read"),
            ToolRiskLevel::Read,
        )
    }

    async fn invoke(&self, ctx: &ToolContext, input: ToolInput) -> ToolResult {
        verify_workspace_scope(&self.services, ctx, "workflow.get")?;
        let req: WorkflowGetInput = serde_json::from_value(input).map_err(|e| {
            ToolError::new(ToolErrorCode::InvalidInput, format!("invalid input: {e}"))
                .with_tool(ToolName::new("workflow.get"))
        })?;
        let workflow = self
            .services
            .workflow_service()
            .snapshot(&req.workflow_id)
            .map_err(|e| tool_error_from_app_host(e, "workflow.get"))?;
        let output = WorkflowGetOutput {
            workflow,
            effective: false,
        };
        serialize_output(output, "workflow.get")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkflowGetInput {
    workflow_id: WorkflowId,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkflowGetOutput {
    workflow: reimagine_core::workflow::Workflow,
    effective: bool,
}

// ------------------------------------------------------------------
// workflow.preview_commands
// ------------------------------------------------------------------

struct WorkflowPreviewCommandsTool {
    services: Arc<WorkspaceServices>,
}

impl WorkflowPreviewCommandsTool {
    fn new(services: Arc<WorkspaceServices>) -> Self {
        Self { services }
    }
}

#[async_trait]
impl AgentTool for WorkflowPreviewCommandsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new("workflow.preview_commands"),
            "Preview a command batch against a workflow without mutating it.",
            [AgentMode::Agent, AgentMode::Build],
            ToolPermission::new("workflow.write"),
            ToolRiskLevel::Read,
        )
    }

    async fn invoke(&self, ctx: &ToolContext, input: ToolInput) -> ToolResult {
        verify_workspace_scope(&self.services, ctx, "workflow.preview_commands")?;
        let req: WorkflowPreviewInput = serde_json::from_value(input).map_err(|e| {
            ToolError::new(ToolErrorCode::InvalidInput, format!("invalid input: {e}"))
                .with_tool(ToolName::new("workflow.preview_commands"))
        })?;
        let result = self
            .services
            .workflow_service()
            .preview_batch(
                &req.workflow_id,
                self.services.node_catalog().as_ref(),
                req.batch,
            )
            .map_err(|e| tool_error_from_app_host(e, "workflow.preview_commands"))?;
        let output = WorkflowPreviewOutput {
            result,
            effective: false,
        };
        serialize_output(output, "workflow.preview_commands")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkflowPreviewInput {
    workflow_id: WorkflowId,
    batch: CommandBatch,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkflowPreviewOutput {
    result: CommandResult,
    effective: bool,
}

// ------------------------------------------------------------------
// workflow.propose_commands
// ------------------------------------------------------------------

struct WorkflowProposeCommandsTool {
    services: Arc<WorkspaceServices>,
}

impl WorkflowProposeCommandsTool {
    fn new(services: Arc<WorkspaceServices>) -> Self {
        Self { services }
    }
}

#[async_trait]
impl AgentTool for WorkflowProposeCommandsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new("workflow.propose_commands"),
            "Preview commands and store a pending proposal without mutating the workflow.",
            [AgentMode::Agent, AgentMode::Build],
            ToolPermission::new("workflow.write"),
            ToolRiskLevel::Editor,
        )
    }

    async fn invoke(&self, ctx: &ToolContext, input: ToolInput) -> ToolResult {
        verify_workspace_scope(&self.services, ctx, "workflow.propose_commands")?;
        let req: WorkflowProposeInput = serde_json::from_value(input).map_err(|e| {
            ToolError::new(ToolErrorCode::InvalidInput, format!("invalid input: {e}"))
                .with_tool(ToolName::new("workflow.propose_commands"))
        })?;

        let preview = self
            .services
            .workflow_service()
            .preview_batch(
                &req.workflow_id,
                self.services.node_catalog().as_ref(),
                req.batch.clone(),
            )
            .map_err(|e| tool_error_from_app_host(e, "workflow.propose_commands"))?;

        if matches!(preview.status(), CommandResultStatus::Rejected) {
            let receipt = ProposalReceipt::new(
                req.proposal_id,
                req.workflow_id.clone(),
                req.batch.base_version(),
                preview,
            )
            .with_status(ProposalStatus::Rejected);
            return serialize_output(receipt, "workflow.propose_commands");
        }

        let base_version = req.batch.base_version();
        let proposal = crate::proposal::WorkflowProposal::new(
            req.proposal_id.clone(),
            req.workflow_id.clone(),
            base_version,
            ctx.agent_session_id().clone(),
            req.batch,
            preview.clone(),
            req.created_at,
        );

        self.services
            .workflow_service()
            .store_proposal(proposal)
            .map_err(|e| tool_error_from_app_host(e, "workflow.propose_commands"))?;

        let receipt = ProposalReceipt::new(req.proposal_id, req.workflow_id, base_version, preview);
        serialize_output(receipt, "workflow.propose_commands")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkflowProposeInput {
    workflow_id: WorkflowId,
    proposal_id: ProposalId,
    batch: CommandBatch,
    created_at: String,
}

// ------------------------------------------------------------------
// workflow.apply_commands
// ------------------------------------------------------------------

struct WorkflowApplyCommandsTool {
    services: Arc<WorkspaceServices>,
}

impl WorkflowApplyCommandsTool {
    fn new(services: Arc<WorkspaceServices>) -> Self {
        Self { services }
    }
}

#[async_trait]
impl AgentTool for WorkflowApplyCommandsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new("workflow.apply_commands"),
            "Apply a command batch to a workflow. In Agent mode, low-risk editor-only batches may be auto-applied after a successful preview.",
            [AgentMode::Agent, AgentMode::Build],
            ToolPermission::new("workflow.write"),
            ToolRiskLevel::Editor,
        )
    }

    async fn invoke(&self, ctx: &ToolContext, input: ToolInput) -> ToolResult {
        verify_workspace_scope(&self.services, ctx, "workflow.apply_commands")?;
        let req: WorkflowApplyInput = serde_json::from_value(input).map_err(|e| {
            ToolError::new(ToolErrorCode::InvalidInput, format!("invalid input: {e}"))
                .with_tool(ToolName::new("workflow.apply_commands"))
        })?;

        // Build mode does not mutate workflow through agent tools.
        if ctx.mode() == AgentMode::Build {
            let output = WorkflowApplyOutput {
                result: None,
                effective: false,
                diagnostics: vec![build_mode_diagnostic()],
            };
            return serialize_output(output, "workflow.apply_commands");
        }

        // Reconstruct batch to enforce Agent actor kind and correct provenance.
        let mut batch = CommandBatch::new(
            req.batch.id().clone(),
            CommandActor::new(CommandActorKind::Agent).with_id(ctx.agent_session_id().as_str()),
            req.batch.base_version(),
            CommandProvenance::Direct,
            req.batch.created_at().clone(),
            req.batch.commands().to_vec(),
        );
        if let Some(cid) = req.batch.correlation_id() {
            batch = batch.with_correlation_id(cid.clone());
        }

        let policy = WorkflowCommandPolicy::new();

        // Policy check: actor kind gate.
        if !policy.allowed_actor_kind(batch.actor().kind()) {
            let output = WorkflowApplyOutput {
                result: None,
                effective: false,
                diagnostics: vec![policy_rejection_diagnostic(
                    "command batch actor kind is not permitted for auto-apply",
                )],
            };
            return serialize_output(output, "workflow.apply_commands");
        }

        // Policy check: only editor-only commands may be auto-applied.
        if !policy.allows_auto_apply(batch.commands()) {
            let output = WorkflowApplyOutput {
                result: None,
                effective: false,
                diagnostics: vec![policy_rejection_diagnostic(
                    "command batch contains non-editor commands; auto-apply blocked",
                )],
            };
            return serialize_output(output, "workflow.apply_commands");
        }

        // Preview first.
        let preview = self
            .services
            .workflow_service()
            .preview_batch(
                &req.workflow_id,
                self.services.node_catalog().as_ref(),
                batch.clone(),
            )
            .map_err(|e| tool_error_from_app_host(e, "workflow.apply_commands"))?;

        if matches!(preview.status(), CommandResultStatus::Rejected) {
            let output = WorkflowApplyOutput {
                result: Some(preview),
                effective: false,
                diagnostics: vec![policy_rejection_diagnostic("preview rejected the batch")],
            };
            return serialize_output(output, "workflow.apply_commands");
        }

        // Apply.
        let result = self
            .services
            .workflow_service()
            .apply_batch(
                &req.workflow_id,
                self.services.node_catalog().as_ref(),
                batch,
            )
            .map_err(|e| tool_error_from_app_host(e, "workflow.apply_commands"))?;

        let effective = matches!(result.status(), CommandResultStatus::Applied);
        let output = WorkflowApplyOutput {
            result: Some(result),
            effective,
            diagnostics: Vec::new(),
        };
        serialize_output(output, "workflow.apply_commands")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkflowApplyInput {
    workflow_id: WorkflowId,
    batch: CommandBatch,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkflowApplyOutput {
    result: Option<CommandResult>,
    effective: bool,
    diagnostics: Vec<reimagine_core::diagnostic::Diagnostic>,
}

// ------------------------------------------------------------------
// model.list
// ------------------------------------------------------------------

struct ModelListTool {
    services: Arc<WorkspaceServices>,
}

impl ModelListTool {
    fn new(services: Arc<WorkspaceServices>) -> Self {
        Self { services }
    }
}

#[async_trait]
impl AgentTool for ModelListTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new("model.list"),
            "List available models in the workspace.",
            [AgentMode::Agent, AgentMode::Build],
            ToolPermission::new("model.read"),
            ToolRiskLevel::Read,
        )
    }

    async fn invoke(&self, ctx: &ToolContext, _input: ToolInput) -> ToolResult {
        verify_workspace_scope(&self.services, ctx, "model.list")?;
        let models = self
            .services
            .model_service()
            .list_models()
            .await
            .map_err(|e| tool_error_from_app_host(e, "model.list"))?;
        let output = ModelListOutput {
            models,
            effective: false,
        };
        serialize_output(output, "model.list")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelListOutput {
    models: Vec<reimagine_model_manager::ModelDescriptor>,
    effective: bool,
}

// ------------------------------------------------------------------
// model.resolve_ref
// ------------------------------------------------------------------

struct ModelResolveRefTool {
    services: Arc<WorkspaceServices>,
}

impl ModelResolveRefTool {
    fn new(services: Arc<WorkspaceServices>) -> Self {
        Self { services }
    }
}

#[async_trait]
impl AgentTool for ModelResolveRefTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new("model.resolve_ref"),
            "Resolve a model reference to readiness or descriptor information.",
            [AgentMode::Agent, AgentMode::Build],
            ToolPermission::new("model.read"),
            ToolRiskLevel::Read,
        )
    }

    async fn invoke(&self, ctx: &ToolContext, input: ToolInput) -> ToolResult {
        verify_workspace_scope(&self.services, ctx, "model.resolve_ref")?;
        let req: ModelResolveInput = serde_json::from_value(input).map_err(|e| {
            ToolError::new(ToolErrorCode::InvalidInput, format!("invalid input: {e}"))
                .with_tool(ToolName::new("model.resolve_ref"))
        })?;
        let resolution = self
            .services
            .model_service()
            .resolve_readiness(&req.model_ref)
            .await
            .map_err(|e| tool_error_from_app_host(e, "model.resolve_ref"))?;
        let diagnostics = resolution.report().diagnostics().to_vec();
        let info = resolution.into_value();
        let output = ModelResolveOutput {
            resolved: info.is_some(),
            model_id: info.as_ref().map(|i| i.id().as_str().to_owned()),
            model_series: info.as_ref().map(|i| i.model_series().as_str().to_owned()),
            variant: info.as_ref().map(|i| i.variant().as_str().to_owned()),
            roles: info.as_ref().map(|i| i.roles().to_vec()),
            format: info.as_ref().map(|i| i.format()),
            source_available: info.as_ref().map(|i| i.source_available()),
            diagnostics,
            effective: false,
        };
        serialize_output(output, "model.resolve_ref")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelResolveInput {
    model_ref: reimagine_core::model::ModelRef,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelResolveOutput {
    resolved: bool,
    model_id: Option<String>,
    model_series: Option<String>,
    variant: Option<String>,
    roles: Option<Vec<reimagine_core::model::ModelRole>>,
    format: Option<reimagine_model_manager::ModelFormat>,
    source_available: Option<bool>,
    diagnostics: Vec<reimagine_core::diagnostic::Diagnostic>,
    effective: bool,
}

// ------------------------------------------------------------------
// diagnostics.for_workflow
// ------------------------------------------------------------------

struct DiagnosticsForWorkflowTool {
    services: Arc<WorkspaceServices>,
}

impl DiagnosticsForWorkflowTool {
    fn new(services: Arc<WorkspaceServices>) -> Self {
        Self { services }
    }
}

#[async_trait]
impl AgentTool for DiagnosticsForWorkflowTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new("diagnostics.for_workflow"),
            "Return immediate diagnostics for the current workflow and session.",
            [AgentMode::Agent, AgentMode::Build],
            ToolPermission::new("workflow.read"),
            ToolRiskLevel::Read,
        )
    }

    async fn invoke(&self, ctx: &ToolContext, input: ToolInput) -> ToolResult {
        verify_workspace_scope(&self.services, ctx, "diagnostics.for_workflow")?;
        let req: DiagnosticsInput = serde_json::from_value(input).map_err(|e| {
            ToolError::new(ToolErrorCode::InvalidInput, format!("invalid input: {e}"))
                .with_tool(ToolName::new("diagnostics.for_workflow"))
        })?;

        let mut diagnostics = Vec::new();

        // Workflow structural diagnostics (via readiness snapshot)
        let workflow = self
            .services
            .workflow_service()
            .snapshot(&req.workflow_id)
            .map_err(|e| tool_error_from_app_host(e, "diagnostics.for_workflow"))?;

        let provider = self
            .services
            .model_service()
            .build_readiness_snapshot(&workflow)
            .await
            .map_err(|e| tool_error_from_app_host(e, "diagnostics.for_workflow"))?;

        let plan_result = reimagine_core::readiness::build_execution_plan(
            &workflow,
            self.services.node_catalog().as_ref(),
            reimagine_core::execution_plan::RunTargetSelection::AllDefaultTargets,
            Some(&provider),
        );

        diagnostics.extend(plan_result.report().diagnostics().iter().cloned());

        let output = DiagnosticsOutput {
            diagnostics,
            effective: false,
        };
        serialize_output(output, "diagnostics.for_workflow")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DiagnosticsInput {
    workflow_id: WorkflowId,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DiagnosticsOutput {
    diagnostics: Vec<reimagine_core::diagnostic::Diagnostic>,
    effective: bool,
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

fn serialize_output<T: serde::Serialize>(output: T, tool_name: &str) -> ToolResult {
    serde_json::to_value(output).map_err(|e| {
        ToolError::new(
            ToolErrorCode::ExecutionFailed,
            format!("serialization failed: {e}"),
        )
        .with_tool(ToolName::new(tool_name))
    })
}

fn tool_error_from_app_host(error: AppHostError, tool_name: &str) -> ToolError {
    ToolError::new(ToolErrorCode::ExecutionFailed, error.to_string())
        .with_tool(ToolName::new(tool_name))
}

fn build_mode_diagnostic() -> reimagine_core::diagnostic::Diagnostic {
    use reimagine_core::diagnostic::{
        Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
        DiagnosticTargetDomain,
    };
    use reimagine_core::model::DiagnosticId;

    Diagnostic::new(
        DiagnosticId::new("agent:apply:build-mode"),
        DiagnosticCode::new("AGENT/BUILD_MODE_NO_AUTO_APPLY"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("agent"),
        "Build mode does not allow auto-applying commands through agent tools",
        DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool"))
            .with_id("workflow.apply_commands"),
    )
}

fn policy_rejection_diagnostic(
    message: impl Into<String>,
) -> reimagine_core::diagnostic::Diagnostic {
    use reimagine_core::diagnostic::{
        Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
        DiagnosticTargetDomain,
    };
    use reimagine_core::model::DiagnosticId;

    Diagnostic::new(
        DiagnosticId::new("agent:apply:policy-rejected"),
        DiagnosticCode::new("AGENT/POLICY_REJECTED"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("agent"),
        message,
        DiagnosticTarget::new(DiagnosticTargetDomain::new("agent.tool"))
            .with_id("workflow.apply_commands"),
    )
}
