//! Workflow proposals through `WorkflowService` directly: storage, approval
//! (the path behind the `approve_proposal` IPC command) and the auto-apply
//! policy. These do not depend on any agent integration; the proposer is
//! recorded as a plain `CommandActor`.

use reimagine_app_host::{
    AppHostError, ProposalStatus, WorkflowCommandPolicy, WorkflowProposal, WorkspaceHost,
    WorkspaceScope,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandStatus, WorkflowCommand,
};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    CommandBatchId, DocumentVersion, NodeId, NodeTypeId, ParamValue, ProposalId, SlotId, WorkflowId,
};
use reimagine_core::workflow::{Position, Workflow, WorkflowMetadata};
use reimagine_nodes::BUILTIN_STRING;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-proposals-{prefix}-{nonce}"))
}

fn build_host(scope: &str) -> WorkspaceHost {
    WorkspaceHost::with_defaults(WorkspaceScope::new(scope), temp_dir(scope))
}

fn proposer() -> CommandActor {
    CommandActor::new(CommandActorKind::Agent).with_id("session-1")
}

fn add_node_batch(base_version: DocumentVersion, node: &str) -> CommandBatch<WorkflowCommand> {
    CommandBatch::new(
        CommandBatchId::new(format!("batch-{node}")),
        proposer(),
        base_version,
        CommandProvenance::Direct,
        Timestamp::new("2026-06-10T00:00:00Z"),
        vec![WorkflowCommand::AddNode {
            node_id: NodeId::new(node),
            type_id: NodeTypeId::new(BUILTIN_STRING),
            label: None,
            params: [(SlotId::new("value"), ParamValue::String("hello".into()))].into(),
            position: None,
        }],
    )
}

/// Register an empty workflow and store a pending proposal that adds `node`.
fn propose(host: &WorkspaceHost, workflow: &str, proposal: &str, node: &str) -> WorkflowId {
    let service = host.workflow_service();
    let workflow_id = service.register_workflow(Workflow::new(workflow, DocumentVersion::new(0)));
    store(host, &workflow_id, proposal, node);
    workflow_id
}

fn store(host: &WorkspaceHost, workflow_id: &WorkflowId, proposal: &str, node: &str) {
    let service = host.workflow_service();
    let base_version = service.snapshot(workflow_id).expect("snapshot").version();
    let batch = add_node_batch(base_version, node);
    let preview = service
        .preview_batch(workflow_id, host.node_catalog().as_ref(), batch.clone())
        .expect("preview");
    assert_eq!(preview.status(), CommandStatus::Applied);
    service
        .store_proposal(WorkflowProposal::new(
            ProposalId::new(proposal),
            workflow_id.clone(),
            base_version,
            proposer(),
            batch,
            preview,
            "2026-06-10T12:00:00Z",
        ))
        .expect("store proposal");
}

#[test]
fn stored_proposal_is_pending_listed_and_removable_without_mutation() {
    let host = build_host("ws-store");
    let service = host.workflow_service();
    let workflow_id = propose(&host, "wf-store", "prop-1", "node-1");

    let pending = service
        .get_pending_proposal(&workflow_id)
        .expect("pending proposal");
    assert_eq!(pending.proposal_id().as_str(), "prop-1");
    assert_eq!(pending.status(), ProposalStatus::Pending);
    assert_eq!(pending.proposed_by(), &proposer());
    assert_eq!(service.list_proposals().len(), 1);

    // Storing a proposal never touches the workflow.
    let snapshot = service.snapshot(&workflow_id).expect("snapshot");
    assert_eq!(snapshot.version(), DocumentVersion::new(0));
    assert!(snapshot.nodes().is_empty());

    let removed = service.remove_proposal(&workflow_id).expect("removed");
    assert_eq!(removed.proposal_id().as_str(), "prop-1");
    assert!(service.get_pending_proposal(&workflow_id).is_none());
    assert!(service.list_proposals().is_empty());
}

#[test]
fn a_new_proposal_replaces_the_pending_one_for_the_same_workflow() {
    let host = build_host("ws-replace");
    let service = host.workflow_service();
    let workflow_id = propose(&host, "wf-replace", "prop-1", "node-1");
    store(&host, &workflow_id, "prop-2", "node-2");

    assert_eq!(service.list_proposals().len(), 1);
    let pending = service.get_pending_proposal(&workflow_id).expect("pending");
    assert_eq!(pending.proposal_id().as_str(), "prop-2");
}

#[tokio::test]
async fn approval_applies_persists_and_clears_the_proposal() {
    let host = build_host("ws-approve");
    let service = host.workflow_service();
    let workflow_id = propose(&host, "wf-approve", "prop-approve", "node-approve");

    let result = service
        .apply_pending_proposal_persisted(
            &workflow_id,
            host.node_catalog().as_ref(),
            Some(CommandActor::new(CommandActorKind::Human)),
        )
        .await
        .expect("approve");
    assert_eq!(result.status(), CommandStatus::Applied);
    assert!(service.get_pending_proposal(&workflow_id).is_none());

    let snapshot = service.snapshot(&workflow_id).expect("snapshot");
    assert_eq!(snapshot.version(), DocumentVersion::new(1));
    assert_eq!(snapshot.nodes().len(), 1);

    // The accepted workflow is on disk: a fresh load sees the new node.
    let path = service.path_for_workflow_id(&workflow_id).expect("path");
    assert!(path.is_file(), "approved workflow persisted at {path:?}");
    service.clear();
    service.load_workflow(&workflow_id).await.expect("reload");
    let reloaded = service.snapshot(&workflow_id).expect("reloaded snapshot");
    assert_eq!(reloaded.version(), DocumentVersion::new(1));
    assert_eq!(reloaded.nodes().len(), 1);
}

#[tokio::test]
async fn stale_approval_is_rejected_and_keeps_the_proposal() {
    let host = build_host("ws-stale");
    let service = host.workflow_service();
    let workflow_id = propose(&host, "wf-stale", "prop-stale", "node-proposed");

    // A direct edit moves the workflow past the proposal's base version.
    let direct = service
        .apply_batch(
            &workflow_id,
            host.node_catalog().as_ref(),
            add_node_batch(DocumentVersion::new(0), "node-direct"),
        )
        .expect("direct apply");
    assert_eq!(direct.status(), CommandStatus::Applied);

    let result = service
        .apply_pending_proposal_persisted(&workflow_id, host.node_catalog().as_ref(), None)
        .await
        .expect("stale approval returns a result");
    assert_eq!(result.status(), CommandStatus::Rejected);
    assert!(service.get_pending_proposal(&workflow_id).is_some());
    assert_eq!(
        service
            .snapshot(&workflow_id)
            .expect("snapshot")
            .nodes()
            .len(),
        1,
        "only the direct edit is applied"
    );
}

#[tokio::test]
async fn approval_without_a_pending_proposal_fails() {
    let host = build_host("ws-none");
    let service = host.workflow_service();
    let workflow_id = service.register_workflow(Workflow::new("wf-none", DocumentVersion::new(0)));

    let error = service
        .apply_pending_proposal_persisted(&workflow_id, host.node_catalog().as_ref(), None)
        .await
        .expect_err("no pending proposal");
    assert!(matches!(error, AppHostError::NoPendingProposal { .. }));
}

#[test]
fn policy_auto_applies_only_editor_commands() {
    let policy = WorkflowCommandPolicy::new();
    assert!(policy.allows_auto_apply(&[
        WorkflowCommand::MoveNode {
            node_id: NodeId::new("n1"),
            position: Position::new(10.0, 20.0),
        },
        WorkflowCommand::SetWorkflowMetadata {
            metadata: WorkflowMetadata::new().with_name("metadata"),
        },
    ]));
    assert!(policy.allows_auto_apply(&[]));
    assert!(!policy.allows_auto_apply(&[WorkflowCommand::AddNode {
        node_id: NodeId::new("n1"),
        type_id: NodeTypeId::new("t1"),
        label: None,
        params: Default::default(),
        position: None,
    }]));
}

#[test]
fn policy_auto_applies_only_for_agent_actors() {
    let policy = WorkflowCommandPolicy::new();
    assert!(policy.allowed_actor_kind(CommandActorKind::Agent));
    assert!(!policy.allowed_actor_kind(CommandActorKind::Human));
    assert!(!policy.allowed_actor_kind(CommandActorKind::System));
    assert!(!policy.allowed_actor_kind(CommandActorKind::Importer));
}
