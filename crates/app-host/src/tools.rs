use std::sync::Arc;

mod workspace_tool;

use reimagine_agent::ToolRiskLevel;
use reimagine_agent::{
    AgentMode, AgentToolRegistry, ToolContext, ToolError, ToolErrorCode, ToolName, ToolResult,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResult,
    CommandResultStatus,
};
use reimagine_core::model::{ProposalId, WorkflowId};

use crate::AppHostError;
use crate::policy::WorkflowCommandPolicy;
use crate::proposal::{ProposalReceipt, ProposalStatus};
use crate::services::WorkspaceServices;
use workspace_tool::{WorkspaceToolSpec, register_workspace_tool};

// ------------------------------------------------------------------
// Tool registration entrypoint
// ------------------------------------------------------------------

/// Register all V1 app-host agent tools into `registry`.
///
/// Tools capture `Arc<WorkspaceServices>` directly and verify the
/// incoming `ToolContext.workspace_scope` matches before doing work.
pub fn register_app_tools(registry: &mut AgentToolRegistry, services: Arc<WorkspaceServices>) {
    register_workspace_tool(
        registry,
        Arc::clone(&services),
        WorkspaceToolSpec::new(
            "workflow.get",
            "Get the current snapshot of a workflow.",
            &[AgentMode::Agent, AgentMode::Build],
            "workflow.read",
            ToolRiskLevel::Read,
        ),
        workflow_get,
    );
    register_workspace_tool(
        registry,
        Arc::clone(&services),
        WorkspaceToolSpec::new(
            "workflow.preview_commands",
            "Preview a command batch against a workflow without mutating it.",
            &[AgentMode::Agent, AgentMode::Build],
            "workflow.write",
            ToolRiskLevel::Read,
        ),
        workflow_preview_commands,
    );
    register_workspace_tool(
        registry,
        Arc::clone(&services),
        WorkspaceToolSpec::new(
            "workflow.propose_commands",
            "Preview commands and store a pending proposal without mutating the workflow.",
            &[AgentMode::Agent, AgentMode::Build],
            "workflow.write",
            ToolRiskLevel::Editor,
        ),
        workflow_propose_commands,
    );
    register_workspace_tool(
        registry,
        Arc::clone(&services),
        WorkspaceToolSpec::new(
            "workflow.apply_commands",
            "Apply a command batch to a workflow. In Agent mode, low-risk editor-only batches may be auto-applied after a successful preview.",
            &[AgentMode::Agent, AgentMode::Build],
            "workflow.write",
            ToolRiskLevel::Editor,
        ),
        workflow_apply_commands,
    );
    register_workspace_tool(
        registry,
        Arc::clone(&services),
        WorkspaceToolSpec::new(
            "model.list",
            "List available models in the workspace.",
            &[AgentMode::Agent, AgentMode::Build],
            "model.read",
            ToolRiskLevel::Read,
        ),
        model_list,
    );
    register_workspace_tool(
        registry,
        Arc::clone(&services),
        WorkspaceToolSpec::new(
            "model.resolve_ref",
            "Resolve a model reference to readiness or descriptor information.",
            &[AgentMode::Agent, AgentMode::Build],
            "model.read",
            ToolRiskLevel::Read,
        ),
        model_resolve_ref,
    );
    register_workspace_tool(
        registry,
        Arc::clone(&services),
        WorkspaceToolSpec::new(
            "diagnostics.for_workflow",
            "Return immediate diagnostics for the current workflow and session.",
            &[AgentMode::Agent, AgentMode::Build],
            "workflow.read",
            ToolRiskLevel::Read,
        ),
        diagnostics_for_workflow,
    );
}

// ------------------------------------------------------------------
// workflow.get
// ------------------------------------------------------------------

async fn workflow_get(
    services: Arc<WorkspaceServices>,
    _ctx: ToolContext,
    req: WorkflowGetInput,
) -> ToolResult<WorkflowGetOutput> {
    let workflow = services
        .workflow_service()
        .snapshot(&req.workflow_id)
        .map_err(|e| tool_error_from_app_host(e, "workflow.get"))?;
    Ok(WorkflowGetOutput {
        workflow,
        effective: false,
    })
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

async fn workflow_preview_commands(
    services: Arc<WorkspaceServices>,
    _ctx: ToolContext,
    req: WorkflowPreviewInput,
) -> ToolResult<WorkflowPreviewOutput> {
    let result = services
        .workflow_service()
        .preview_batch(
            &req.workflow_id,
            services.node_catalog().as_ref(),
            req.batch,
        )
        .map_err(|e| tool_error_from_app_host(e, "workflow.preview_commands"))?;
    Ok(WorkflowPreviewOutput {
        result,
        effective: false,
    })
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

async fn workflow_propose_commands(
    services: Arc<WorkspaceServices>,
    ctx: ToolContext,
    req: WorkflowProposeInput,
) -> ToolResult<ProposalReceipt> {
    let preview = services
        .workflow_service()
        .preview_batch(
            &req.workflow_id,
            services.node_catalog().as_ref(),
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
        return Ok(receipt);
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

    services
        .workflow_service()
        .store_proposal(proposal)
        .map_err(|e| tool_error_from_app_host(e, "workflow.propose_commands"))?;

    Ok(ProposalReceipt::new(
        req.proposal_id,
        req.workflow_id,
        base_version,
        preview,
    ))
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

async fn workflow_apply_commands(
    services: Arc<WorkspaceServices>,
    ctx: ToolContext,
    req: WorkflowApplyInput,
) -> ToolResult<WorkflowApplyOutput> {
    // Build mode does not mutate workflow through agent tools.
    if ctx.mode() == AgentMode::Build {
        return Ok(WorkflowApplyOutput {
            result: None,
            effective: false,
            diagnostics: vec![build_mode_diagnostic()],
        });
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
        return Ok(WorkflowApplyOutput {
            result: None,
            effective: false,
            diagnostics: vec![policy_rejection_diagnostic(
                "command batch actor kind is not permitted for auto-apply",
            )],
        });
    }

    // Policy check: only editor-only commands may be auto-applied.
    if !policy.allows_auto_apply(batch.commands()) {
        return Ok(WorkflowApplyOutput {
            result: None,
            effective: false,
            diagnostics: vec![policy_rejection_diagnostic(
                "command batch contains non-editor commands; auto-apply blocked",
            )],
        });
    }

    // Preview first.
    let preview = services
        .workflow_service()
        .preview_batch(
            &req.workflow_id,
            services.node_catalog().as_ref(),
            batch.clone(),
        )
        .map_err(|e| tool_error_from_app_host(e, "workflow.apply_commands"))?;

    if matches!(preview.status(), CommandResultStatus::Rejected) {
        return Ok(WorkflowApplyOutput {
            result: Some(preview),
            effective: false,
            diagnostics: vec![policy_rejection_diagnostic("preview rejected the batch")],
        });
    }

    // Apply.
    let result = services
        .workflow_service()
        .apply_batch(&req.workflow_id, services.node_catalog().as_ref(), batch)
        .map_err(|e| tool_error_from_app_host(e, "workflow.apply_commands"))?;

    let effective = matches!(result.status(), CommandResultStatus::Applied);
    Ok(WorkflowApplyOutput {
        result: Some(result),
        effective,
        diagnostics: Vec::new(),
    })
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

async fn model_list(
    services: Arc<WorkspaceServices>,
    _ctx: ToolContext,
    _input: serde_json::Value,
) -> ToolResult<ModelListOutput> {
    let models = services
        .model_service()
        .list_models()
        .await
        .map_err(|e| tool_error_from_app_host(e, "model.list"))?;
    Ok(ModelListOutput {
        models,
        effective: false,
    })
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelListOutput {
    models: Vec<reimagine_model_manager::ModelDescriptor>,
    effective: bool,
}

// ------------------------------------------------------------------
// model.resolve_ref
// ------------------------------------------------------------------

async fn model_resolve_ref(
    services: Arc<WorkspaceServices>,
    _ctx: ToolContext,
    req: ModelResolveInput,
) -> ToolResult<ModelResolveOutput> {
    let resolution = services
        .model_service()
        .resolve_readiness(&req.model_ref)
        .await
        .map_err(|e| tool_error_from_app_host(e, "model.resolve_ref"))?;
    let diagnostics = resolution.report().diagnostics().to_vec();
    let info = resolution.into_value();
    Ok(ModelResolveOutput {
        resolved: info.is_some(),
        model_id: info.as_ref().map(|i| i.id().as_str().to_owned()),
        model_series: info.as_ref().map(|i| i.model_series().as_str().to_owned()),
        variant: info.as_ref().map(|i| i.variant().as_str().to_owned()),
        roles: info.as_ref().map(|i| i.roles().to_vec()),
        format: info.as_ref().map(|i| i.format()),
        source_available: info.as_ref().map(|i| i.source_available()),
        diagnostics,
        effective: false,
    })
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

async fn diagnostics_for_workflow(
    services: Arc<WorkspaceServices>,
    _ctx: ToolContext,
    req: DiagnosticsInput,
) -> ToolResult<DiagnosticsOutput> {
    let mut diagnostics = Vec::new();

    // Workspace bootstrap invariant: surface catalog/executor
    // alignment problems so users and Agent tools can see them
    // without having to call `run_workflow` first.
    let alignment = services.runtime_service().registry();
    let alignment_report = services.node_catalog().check_alignment(alignment);
    diagnostics.extend(alignment_report.diagnostics());

    // Workflow structural diagnostics (via readiness snapshot)
    let workflow = services
        .workflow_service()
        .snapshot(&req.workflow_id)
        .map_err(|e| tool_error_from_app_host(e, "diagnostics.for_workflow"))?;

    let provider = services
        .model_service()
        .build_readiness_snapshot(&workflow)
        .await
        .map_err(|e| tool_error_from_app_host(e, "diagnostics.for_workflow"))?;

    let plan_result = reimagine_core::readiness::build_execution_plan(
        &workflow,
        services.node_catalog().as_ref(),
        reimagine_core::execution_plan::RunTargetSelection::AllDefaultTargets,
        Some(&provider),
    );

    diagnostics.extend(plan_result.report().diagnostics().iter().cloned());

    Ok(DiagnosticsOutput {
        diagnostics,
        effective: false,
    })
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
