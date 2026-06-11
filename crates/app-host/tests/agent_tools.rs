//! Integration tests for V1 Agent tools and proposal policy.
//!
//! Covers:
//!   * tool registration via WorkspaceHost
//!   * workspace-scope mismatch rejection
//!   * preview without mutation
//!   * proposal storage without mutation
//!   * WorkflowCommandPolicy auto-apply gating
//!   * agent-mode apply_commands for editor-only batches
//!   * build-mode rejection

use std::sync::Arc;

use reimagine_agent::{
    AgentMode, AgentSessionId, PermissionSet, ToolContext, ToolPermission, WorkspaceScope,
};
use reimagine_app_host::{ProposalReceipt, ProposalStatus, WorkflowCommandPolicy, WorkspaceHost};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResultStatus,
    WorkflowCommand,
};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{CommandBatchId, NodeId, NodeTypeId, SlotId, WorkflowVersion};
use reimagine_core::workflow::{Workflow, WorkflowMetadata};
use reimagine_nodes::BUILTIN_STRING;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-agent-tools-{prefix}-{nonce}"))
}

fn build_host(scope: &str) -> WorkspaceHost {
    WorkspaceHost::with_defaults(WorkspaceScope::new(scope), temp_dir(scope))
}

fn agent_ctx(workspace_scope: WorkspaceScope, mode: AgentMode) -> ToolContext {
    ToolContext::new(workspace_scope, AgentSessionId::new("sess-1"), mode).with_permissions(
        PermissionSet::from_iter([
            ToolPermission::new("workflow.read"),
            ToolPermission::new("workflow.write"),
            ToolPermission::new("model.read"),
        ]),
    )
}

fn build_workflow(workflow_id: &str) -> Workflow {
    Workflow::new(workflow_id, WorkflowVersion::new(0))
}

fn add_node_batch(base_version: WorkflowVersion, label: &str) -> CommandBatch {
    CommandBatch::new(
        CommandBatchId::new(format!("batch-{label}")),
        CommandActor::new(CommandActorKind::Agent).with_id("sess-1"),
        base_version,
        CommandProvenance::Direct,
        Timestamp::new("2026-06-10T00:00:00Z"),
        vec![WorkflowCommand::AddNode {
            node_id: NodeId::new(label),
            type_id: NodeTypeId::new(BUILTIN_STRING),
            label: None,
            params: [(
                SlotId::new("value"),
                reimagine_core::model::ParamValue::String("hello".into()),
            )]
            .into(),
            position: None,
        }],
    )
}

fn metadata_batch(base_version: WorkflowVersion, name: &str) -> CommandBatch {
    CommandBatch::new(
        CommandBatchId::new(format!("batch-metadata-{name}")),
        CommandActor::new(CommandActorKind::Agent).with_id("sess-1"),
        base_version,
        CommandProvenance::Direct,
        Timestamp::new("2026-06-10T00:00:00Z"),
        vec![WorkflowCommand::SetWorkflowMetadata {
            metadata: WorkflowMetadata::new().with_name(name),
        }],
    )
}

#[test]
fn workspace_host_registers_all_v1_tools() {
    let host = build_host("ws-tools");
    let registry = host.agent_service().registry();

    let names: Vec<String> = registry
        .tool_names()
        .into_iter()
        .map(|n| n.as_str().to_owned())
        .collect();
    assert!(names.contains(&"workflow.get".to_owned()));
    assert!(names.contains(&"workflow.preview_commands".to_owned()));
    assert!(names.contains(&"workflow.propose_commands".to_owned()));
    assert!(names.contains(&"workflow.apply_commands".to_owned()));
    assert!(names.contains(&"model.list".to_owned()));
    assert!(names.contains(&"model.resolve_ref".to_owned()));
    assert!(names.contains(&"diagnostics.for_workflow".to_owned()));
    assert_eq!(names.len(), 7);
}

#[tokio::test]
async fn tool_rejects_workspace_mismatch() {
    let host = build_host("ws-mismatch");
    let registry = Arc::clone(host.agent_service().registry());

    let wrong_ctx = agent_ctx(WorkspaceScope::new("other-ws"), AgentMode::Agent);
    let input = serde_json::json!({"workflow_id": "wf-1" });

    let result = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.get"),
            &wrong_ctx,
            input,
        )
        .await;

    let err = result.expect_err("should fail on workspace mismatch");
    let msg = format!("{err}");
    assert!(
        msg.contains("WORKSPACE_MISMATCH"),
        "expected workspace mismatch error, got: {msg}"
    );
}

#[tokio::test]
async fn preview_commands_does_not_mutate_workflow() {
    let host = build_host("ws-preview");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-preview");
    let workflow_id = service.register_workflow(workflow);

    let batch = add_node_batch(WorkflowVersion::new(0), "node-preview");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "batch": batch,
    });

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.preview_commands"),
            &ctx,
            input,
        )
        .await
        .expect("preview should succeed");

    let preview_out: serde_json::Value = output;
    assert_eq!(preview_out["effective"], false);
    assert_eq!(preview_out["result"]["status"], "Applied");

    // Workflow version must remain unchanged after preview.
    let snapshot = service.snapshot(&workflow_id).unwrap();
    assert_eq!(snapshot.version(), WorkflowVersion::new(0));
    assert!(snapshot.nodes().is_empty());
}

#[tokio::test]
async fn propose_commands_stores_pending_proposal_without_mutation() {
    let host = build_host("ws-propose");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-propose");
    let workflow_id = service.register_workflow(workflow);

    let batch = add_node_batch(WorkflowVersion::new(0), "node-propose");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "proposal_id": "prop-1",
        "batch": batch,
        "created_at": "2026-06-10T12:00:00Z",
    });

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Build);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.propose_commands"),
            &ctx,
            input,
        )
        .await
        .expect("propose should succeed");

    let receipt: ProposalReceipt = serde_json::from_value(output).expect("valid receipt");
    assert_eq!(receipt.proposal_id().as_str(), "prop-1");
    assert_eq!(receipt.status(), ProposalStatus::Pending);
    assert_eq!(receipt.effective(), false);
    assert_eq!(
        receipt.preview_result().status(),
        CommandResultStatus::Applied
    );

    // Workflow must not be mutated.
    let snapshot = service.snapshot(&workflow_id).unwrap();
    assert_eq!(snapshot.version(), WorkflowVersion::new(0));
    assert!(snapshot.nodes().is_empty());

    // Proposal must be stored.
    let proposal = service
        .get_pending_proposal(&workflow_id)
        .expect("pending proposal should exist");
    assert_eq!(proposal.proposal_id().as_str(), "prop-1");
    assert_eq!(proposal.status(), ProposalStatus::Pending);
}

#[tokio::test]
async fn propose_commands_replaces_previous_pending_proposal() {
    let host = build_host("ws-supersede");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-supersede");
    let workflow_id = service.register_workflow(workflow);

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Build);
    let registry = Arc::clone(host.agent_service().registry());

    for (prop_id, node_id) in [("prop-1", "node-a"), ("prop-2", "node-b")] {
        let batch = add_node_batch(WorkflowVersion::new(0), node_id);
        let input = serde_json::json!({
            "workflow_id": workflow_id.as_str(),
            "proposal_id": prop_id,
            "batch": batch,
            "created_at": "2026-06-10T12:00:00Z",
        });
        let _ = registry
            .invoke(
                &reimagine_agent::ToolName::new("workflow.propose_commands"),
                &ctx,
                input,
            )
            .await
            .expect("propose should succeed");
    }

    let proposals = service.list_proposals();
    assert_eq!(proposals.len(), 1);
    let pending = service.get_pending_proposal(&workflow_id).unwrap();
    assert_eq!(pending.proposal_id().as_str(), "prop-2");
}

#[tokio::test]
async fn apply_commands_in_build_mode_is_not_effective() {
    let host = build_host("ws-build-apply");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-build-apply");
    let workflow_id = service.register_workflow(workflow);

    let batch = add_node_batch(WorkflowVersion::new(0), "node-build");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "batch": batch,
    });

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Build);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.apply_commands"),
            &ctx,
            input,
        )
        .await
        .expect("invoke should succeed");

    let out: serde_json::Value = output;
    assert_eq!(out["effective"], false);
    assert!(out["diagnostics"].as_array().unwrap().len() > 0);

    let snapshot = service.snapshot(&workflow_id).unwrap();
    assert_eq!(snapshot.version(), WorkflowVersion::new(0));
}

#[tokio::test]
async fn apply_commands_in_agent_mode_applies_editor_only_batch() {
    let host = build_host("ws-agent-apply");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-agent-apply");
    let workflow_id = service.register_workflow(workflow);

    let batch = metadata_batch(WorkflowVersion::new(0), "agent metadata");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "batch": batch,
    });

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.apply_commands"),
            &ctx,
            input,
        )
        .await
        .expect("invoke should succeed");

    let out: serde_json::Value = output;
    assert_eq!(out["effective"], true);
    assert_eq!(out["result"]["status"], "Applied");

    let snapshot = service.snapshot(&workflow_id).unwrap();
    assert_eq!(snapshot.version(), WorkflowVersion::new(1));
    assert_eq!(snapshot.metadata().name(), Some("agent metadata"));
}

#[tokio::test]
async fn apply_commands_in_agent_mode_rejects_graph_semantic_batch() {
    let host = build_host("ws-agent-apply-policy");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-agent-apply-policy");
    let workflow_id = service.register_workflow(workflow);

    let batch = add_node_batch(WorkflowVersion::new(0), "node-agent-rejected");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "batch": batch,
    });

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.apply_commands"),
            &ctx,
            input,
        )
        .await
        .expect("invoke should succeed");

    let out: serde_json::Value = output;
    assert_eq!(out["effective"], false);
    assert!(out["diagnostics"].as_array().unwrap().len() > 0);

    let snapshot = service.snapshot(&workflow_id).unwrap();
    assert_eq!(snapshot.version(), WorkflowVersion::new(0));
    assert!(snapshot.nodes().is_empty());
}

#[tokio::test]
async fn apply_commands_returns_effective_false_when_preview_rejects() {
    let host = build_host("ws-agent-reject");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-agent-reject");
    let workflow_id = service.register_workflow(workflow);

    // Batch with wrong base version should be rejected by preview.
    let batch = metadata_batch(WorkflowVersion::new(99), "wrong-version");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "batch": batch,
    });

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.apply_commands"),
            &ctx,
            input,
        )
        .await
        .expect("invoke should succeed");

    let out: serde_json::Value = output;
    assert_eq!(out["effective"], false);
    assert_eq!(out["result"]["status"], "Rejected");

    let snapshot = service.snapshot(&workflow_id).unwrap();
    assert_eq!(snapshot.version(), WorkflowVersion::new(0));
}

#[test]
fn workflow_command_policy_allows_editor_only() {
    let policy = WorkflowCommandPolicy::new();
    let commands = vec![
        WorkflowCommand::MoveNode {
            node_id: NodeId::new("n1"),
            position: reimagine_core::workflow::Position::new(10.0, 20.0),
        },
        WorkflowCommand::SetWorkflowMetadata {
            metadata: WorkflowMetadata::new().with_name("metadata"),
        },
    ];
    assert!(policy.allows_auto_apply(&commands));
}

#[test]
fn workflow_command_policy_rejects_graph_semantic_commands() {
    let policy = WorkflowCommandPolicy::new();
    let commands = vec![WorkflowCommand::AddNode {
        node_id: NodeId::new("n1"),
        type_id: NodeTypeId::new("t1"),
        label: None,
        params: Default::default(),
        position: None,
    }];
    assert!(!policy.allows_auto_apply(&commands));
}

#[test]
fn workflow_command_policy_allows_empty_batch() {
    let policy = WorkflowCommandPolicy::new();
    assert!(policy.allows_auto_apply(&[]));
}

#[tokio::test]
async fn workflow_get_returns_snapshot() {
    let host = build_host("ws-get");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-get");
    let workflow_id = service.register_workflow(workflow);

    let input = serde_json::json!({ "workflow_id": workflow_id.as_str() });
    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(&reimagine_agent::ToolName::new("workflow.get"), &ctx, input)
        .await
        .expect("get should succeed");

    let out: serde_json::Value = output;
    assert_eq!(out["effective"], false);
    assert_eq!(out["workflow"]["id"], "wf-get");
}

#[tokio::test]
async fn approved_proposal_apply_through_host_api() {
    let host = build_host("ws-approve");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-approve");
    let workflow_id = service.register_workflow(workflow);

    // Create a proposal via tool.
    let batch = add_node_batch(WorkflowVersion::new(0), "node-approve");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "proposal_id": "prop-approve",
        "batch": batch.clone(),
        "created_at": "2026-06-10T12:00:00Z",
    });
    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Build);
    let registry = Arc::clone(host.agent_service().registry());
    let _receipt = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.propose_commands"),
            &ctx,
            input,
        )
        .await
        .expect("propose should succeed");

    // Apply the pending proposal through WorkflowService host API.
    let result = service
        .apply_pending_proposal(
            &workflow_id,
            host.node_catalog().as_ref(),
            Some(reimagine_core::command::CommandActor::new(
                CommandActorKind::Human,
            )),
        )
        .expect("apply_pending_proposal should succeed");
    assert_eq!(result.status(), CommandResultStatus::Applied);

    // Proposal should be removed after application.
    assert!(service.get_pending_proposal(&workflow_id).is_none());

    let snapshot = service.snapshot(&workflow_id).unwrap();
    assert_eq!(snapshot.version(), WorkflowVersion::new(1));
    assert_eq!(snapshot.nodes().len(), 1);
}

#[tokio::test]
async fn rejected_proposal_apply_keeps_pending_proposal() {
    let host = build_host("ws-approve-reject");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-approve-reject");
    let workflow_id = service.register_workflow(workflow);

    let proposed_batch = add_node_batch(WorkflowVersion::new(0), "node-proposed");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "proposal_id": "prop-stale",
        "batch": proposed_batch,
        "created_at": "2026-06-10T12:00:00Z",
    });
    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Build);
    let registry = Arc::clone(host.agent_service().registry());
    let _receipt = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.propose_commands"),
            &ctx,
            input,
        )
        .await
        .expect("propose should succeed");

    let direct_batch = add_node_batch(WorkflowVersion::new(0), "node-direct");
    let direct_result = service
        .apply_batch(&workflow_id, host.node_catalog().as_ref(), direct_batch)
        .expect("direct apply should succeed");
    assert_eq!(direct_result.status(), CommandResultStatus::Applied);

    let stale_result = service
        .apply_pending_proposal(
            &workflow_id,
            host.node_catalog().as_ref(),
            Some(reimagine_core::command::CommandActor::new(
                CommandActorKind::Human,
            )),
        )
        .expect("stale proposal apply should return a rejected result");

    assert_eq!(stale_result.status(), CommandResultStatus::Rejected);
    assert!(service.get_pending_proposal(&workflow_id).is_some());
}

#[tokio::test]
async fn preview_commands_rejects_workspace_mismatch() {
    let host = build_host("ws-mismatch-preview");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-mismatch-preview");
    let workflow_id = service.register_workflow(workflow);

    let batch = add_node_batch(WorkflowVersion::new(0), "node-mismatch");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "batch": batch,
    });

    let wrong_ctx = agent_ctx(WorkspaceScope::new("other-ws"), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let result = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.preview_commands"),
            &wrong_ctx,
            input,
        )
        .await;

    let err = result.expect_err("should fail on workspace mismatch");
    let msg = format!("{err}");
    assert!(
        msg.contains("WORKSPACE_MISMATCH"),
        "expected workspace mismatch error, got: {msg}"
    );
}

#[tokio::test]
async fn propose_commands_rejects_workspace_mismatch() {
    let host = build_host("ws-mismatch-propose");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-mismatch-propose");
    let workflow_id = service.register_workflow(workflow);

    let batch = add_node_batch(WorkflowVersion::new(0), "node-mismatch");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "proposal_id": "prop-mismatch",
        "batch": batch,
        "created_at": "2026-06-10T12:00:00Z",
    });

    let wrong_ctx = agent_ctx(WorkspaceScope::new("other-ws"), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let result = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.propose_commands"),
            &wrong_ctx,
            input,
        )
        .await;

    let err = result.expect_err("should fail on workspace mismatch");
    let msg = format!("{err}");
    assert!(
        msg.contains("WORKSPACE_MISMATCH"),
        "expected workspace mismatch error, got: {msg}"
    );
}

#[tokio::test]
async fn apply_commands_rejects_workspace_mismatch() {
    let host = build_host("ws-mismatch-apply");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-mismatch-apply");
    let workflow_id = service.register_workflow(workflow);

    let batch = add_node_batch(WorkflowVersion::new(0), "node-mismatch");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "batch": batch,
    });

    let wrong_ctx = agent_ctx(WorkspaceScope::new("other-ws"), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let result = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.apply_commands"),
            &wrong_ctx,
            input,
        )
        .await;

    let err = result.expect_err("should fail on workspace mismatch");
    let msg = format!("{err}");
    assert!(
        msg.contains("WORKSPACE_MISMATCH"),
        "expected workspace mismatch error, got: {msg}"
    );
}

#[tokio::test]
async fn propose_commands_returns_rejected_when_preview_fails() {
    let host = build_host("ws-propose-reject");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-propose-reject");
    let workflow_id = service.register_workflow(workflow);

    // Batch with wrong base version should be rejected by preview.
    let batch = add_node_batch(WorkflowVersion::new(99), "node-wrong");
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "proposal_id": "prop-reject",
        "batch": batch,
        "created_at": "2026-06-10T12:00:00Z",
    });

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.propose_commands"),
            &ctx,
            input,
        )
        .await
        .expect("propose should succeed even when preview rejects");

    let receipt: ProposalReceipt = serde_json::from_value(output).expect("valid receipt");
    assert_eq!(receipt.proposal_id().as_str(), "prop-reject");
    assert_eq!(receipt.status(), ProposalStatus::Rejected);
    assert_eq!(receipt.effective(), false);
    assert_eq!(
        receipt.preview_result().status(),
        CommandResultStatus::Rejected
    );

    // No pending proposal should be stored.
    assert!(service.get_pending_proposal(&workflow_id).is_none());
}

#[tokio::test]
async fn apply_commands_ignores_input_actor_kind() {
    let host = build_host("ws-agent-kind");
    let service = host.workflow_service();
    let workflow = build_workflow("wf-agent-kind");
    let workflow_id = service.register_workflow(workflow);

    // Input batch claims Human actor kind.
    let batch = CommandBatch::new(
        CommandBatchId::new("batch-human"),
        CommandActor::new(CommandActorKind::Human).with_id("sess-1"),
        WorkflowVersion::new(0),
        CommandProvenance::Direct,
        Timestamp::new("2026-06-10T00:00:00Z"),
        vec![WorkflowCommand::SetWorkflowMetadata {
            metadata: WorkflowMetadata::new().with_name("actor rewritten"),
        }],
    );
    let input = serde_json::json!({
        "workflow_id": workflow_id.as_str(),
        "batch": batch,
    });

    let ctx = agent_ctx(host.workspace_scope().clone(), AgentMode::Agent);
    let registry = Arc::clone(host.agent_service().registry());
    let output = registry
        .invoke(
            &reimagine_agent::ToolName::new("workflow.apply_commands"),
            &ctx,
            input,
        )
        .await
        .expect("invoke should succeed");

    let out: serde_json::Value = output;
    assert_eq!(out["effective"], true);
    assert_eq!(out["result"]["status"], "Applied");
}

#[test]
fn workflow_command_policy_allows_only_agent_actor_kind() {
    let policy = WorkflowCommandPolicy::new();
    assert!(policy.allowed_actor_kind(CommandActorKind::Agent));
    assert!(!policy.allowed_actor_kind(CommandActorKind::Human));
    assert!(!policy.allowed_actor_kind(CommandActorKind::System));
    assert!(!policy.allowed_actor_kind(CommandActorKind::Importer));
}
