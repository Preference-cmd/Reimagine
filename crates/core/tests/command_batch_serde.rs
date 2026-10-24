use reimagine_core::command::{CommandActorKind, CommandBatch, CommandProvenance, WorkflowCommand};
use reimagine_core::model::ProposalId;

#[test]
fn command_batch_serde_roundtrip_with_correlation_id() {
    let json_literal = serde_json::json!({
        "id": "batch-1",
        "actor": {
            "kind": "Human",
            "id": "user-1",
            "label": "Alice"
        },
        "base_version": 7,
        "provenance": {
            "type": "direct"
        },
        "created_at": "2026-06-09T10:00:00Z",
        "correlation_id": "corr-preview",
        "commands": [
            {
                "type": "remove_node",
                "node_id": "node_source"
            }
        ]
    });

    let batch: CommandBatch<WorkflowCommand> =
        serde_json::from_value(json_literal.clone()).expect("deserialization succeeds");

    assert_eq!(batch.id().as_str(), "batch-1");
    assert_eq!(batch.context().actor().kind(), CommandActorKind::Human);
    assert_eq!(batch.context().actor().id(), Some("user-1"));
    assert_eq!(batch.context().actor().label(), Some("Alice"));
    assert_eq!(batch.base_version().get(), 7);
    assert_eq!(batch.context().provenance(), &CommandProvenance::Direct);
    assert_eq!(batch.created_at().as_str(), "2026-06-09T10:00:00Z");
    assert_eq!(
        batch.context().correlation_id().map(|c| c.as_str()),
        Some("corr-preview")
    );
    assert_eq!(batch.commands().len(), 1);

    let serialized = serde_json::to_value(&batch).expect("serialization succeeds");
    assert_eq!(serialized, json_literal);
}

#[test]
fn command_batch_serde_roundtrip_without_correlation_id() {
    let json_literal = serde_json::json!({
        "id": "batch-2",
        "actor": {
            "kind": "Agent",
            "id": null,
            "label": "planner"
        },
        "base_version": 2,
        "provenance": {
            "type": "agent_proposal",
            "proposal_id": "proposal-1",
            "approved_by": null
        },
        "created_at": "2026-06-09T11:00:00Z",
        "correlation_id": null,
        "commands": [
            {
                "type": "remove_node",
                "node_id": "node_consumer"
            }
        ]
    });

    let batch: CommandBatch<WorkflowCommand> =
        serde_json::from_value(json_literal.clone()).expect("deserialization succeeds");

    assert_eq!(batch.id().as_str(), "batch-2");
    assert_eq!(batch.context().actor().kind(), CommandActorKind::Agent);
    assert_eq!(batch.context().actor().id(), None);
    assert_eq!(batch.context().actor().label(), Some("planner"));
    assert_eq!(batch.base_version().get(), 2);
    assert_eq!(
        batch.context().provenance(),
        &CommandProvenance::AgentProposal {
            proposal_id: ProposalId::new("proposal-1"),
            approved_by: None,
        }
    );
    assert_eq!(batch.created_at().as_str(), "2026-06-09T11:00:00Z");
    assert!(batch.context().correlation_id().is_none());
    assert_eq!(batch.commands().len(), 1);

    let serialized = serde_json::to_value(&batch).expect("serialization succeeds");
    assert_eq!(serialized, json_literal);
}

#[test]
fn command_batch_deserialization_accepts_omitted_correlation_id() {
    let json_literal = serde_json::json!({
        "id": "batch-3",
        "actor": {
            "kind": "Human",
            "id": null,
            "label": null
        },
        "base_version": 0,
        "provenance": {
            "type": "direct"
        },
        "created_at": "2026-01-01T00:00:00Z",
        "commands": []
    });

    let batch: CommandBatch<WorkflowCommand> = serde_json::from_value(json_literal)
        .expect("deserialization without correlation_id field succeeds");
    assert!(batch.context().correlation_id().is_none());
}
