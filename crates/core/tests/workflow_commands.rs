use std::collections::{BTreeMap, HashMap};

use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandResultStatus,
    WorkflowChange, WorkflowCommand,
};
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    CommandBatchId, EdgeId, NodeCatalog, NodeDef, NodeId, NodeTypeId, ParamValue, SlotId, SlotKind,
    WorkflowVersion,
};
use reimagine_core::session::WorkflowSession;
use reimagine_core::workflow::{
    Endpoint, Position, Workflow, WorkflowEdge, WorkflowLayout, WorkflowMetadata, WorkflowNode,
};

#[test]
fn preview_batch_returns_changes_without_mutating_session_or_history() {
    let catalog = test_catalog();
    let workflow = empty_workflow("workflow_preview", 7);
    let batch = CommandBatch::new(
        CommandBatchId::new("batch-preview"),
        CommandActor::new(CommandActorKind::Human)
            .with_id("user-1")
            .with_label("Alice"),
        WorkflowVersion::new(7),
        CommandProvenance::Direct,
        Timestamp::new("2026-06-09T10:00:00Z"),
        vec![
            WorkflowCommand::AddNode {
                node_id: NodeId::new("node_source"),
                type_id: NodeTypeId::new("builtin.source"),
                label: Some("Source".to_owned()),
                params: BTreeMap::new(),
                position: Some(Position::new(10.0, 20.0)),
            },
            WorkflowCommand::AddNode {
                node_id: NodeId::new("node_consumer"),
                type_id: NodeTypeId::new("builtin.consumer"),
                label: None,
                params: BTreeMap::from([(SlotId::new("count"), ParamValue::Integer(3))]),
                position: Some(Position::new(110.0, 20.0)),
            },
            WorkflowCommand::Connect {
                edge_id: EdgeId::new("edge-1"),
                from: Endpoint::node_slot(NodeId::new("node_source"), SlotId::new("value")),
                to: Endpoint::node_slot(NodeId::new("node_consumer"), SlotId::new("dynamic_value")),
            },
        ],
    )
    .with_correlation_id(CorrelationId::new("corr-preview"));

    let session = WorkflowSession::new(workflow.clone());

    let result = session.preview_batch(&catalog, batch);

    assert_eq!(result.status(), CommandResultStatus::Applied);
    assert_eq!(result.workflow_version(), WorkflowVersion::new(8));
    assert_eq!(result.history_entry_id(), None);
    assert!(result.diagnostics().is_empty());
    assert_eq!(session.workflow(), &workflow);
    assert_eq!(session.version(), WorkflowVersion::new(7));
    assert!(session.history().entries().is_empty());
    assert_eq!(session.history().cursor(), 0);

    assert_eq!(
        result.changes(),
        &[
            WorkflowChange::NodeAdded {
                node: WorkflowNode::new(
                    NodeId::new("node_source"),
                    NodeTypeId::new("builtin.source"),
                )
                .with_label("Source"),
            },
            WorkflowChange::NodeMoved {
                node_id: NodeId::new("node_source"),
                before: None,
                after: Some(Position::new(10.0, 20.0)),
            },
            WorkflowChange::NodeAdded {
                node: WorkflowNode::new(
                    NodeId::new("node_consumer"),
                    NodeTypeId::new("builtin.consumer"),
                )
                .with_param(SlotId::new("count"), ParamValue::Integer(3)),
            },
            WorkflowChange::NodeMoved {
                node_id: NodeId::new("node_consumer"),
                before: None,
                after: Some(Position::new(110.0, 20.0)),
            },
            WorkflowChange::EdgeAdded {
                edge: WorkflowEdge::new(
                    EdgeId::new("edge-1"),
                    Endpoint::node_slot(NodeId::new("node_source"), SlotId::new("value")),
                    Endpoint::node_slot(NodeId::new("node_consumer"), SlotId::new("dynamic_value")),
                ),
            },
            WorkflowChange::VersionAdvanced {
                before: WorkflowVersion::new(7),
                after: WorkflowVersion::new(8),
            },
        ]
    );
}

#[test]
fn apply_batch_commits_changes_advances_version_and_records_history() {
    let catalog = test_catalog();
    let workflow = workflow_with_nodes();
    let original_layout = workflow.layout().clone();
    let mut session = WorkflowSession::new(workflow);

    let metadata = WorkflowMetadata::new()
        .with_name("Prompt Flow")
        .with_description("demo")
        .with_created_by("tester");

    let batch = CommandBatch::new(
        CommandBatchId::new("batch-apply"),
        CommandActor::new(CommandActorKind::Agent).with_label("planner"),
        WorkflowVersion::new(2),
        CommandProvenance::AgentProposal {
            proposal_id: reimagine_core::model::ProposalId::new("proposal-1"),
            approved_by: Some(
                CommandActor::new(CommandActorKind::Human)
                    .with_id("user-2")
                    .with_label("Reviewer"),
            ),
        },
        Timestamp::new("2026-06-09T11:00:00Z"),
        vec![
            WorkflowCommand::SetParam {
                node_id: NodeId::new("node_consumer"),
                slot_id: SlotId::new("count"),
                value: ParamValue::Integer(9),
            },
            WorkflowCommand::MoveNode {
                node_id: NodeId::new("node_consumer"),
                position: Position::new(300.0, 120.0),
            },
            WorkflowCommand::SetNodeLabel {
                node_id: NodeId::new("node_consumer"),
                label: Some("Sink".to_owned()),
            },
            WorkflowCommand::SetWorkflowMetadata {
                metadata: metadata.clone(),
            },
        ],
    );

    let result = session.apply_batch(&catalog, batch.clone());

    assert_eq!(result.status(), CommandResultStatus::Applied);
    assert_eq!(result.workflow_version(), WorkflowVersion::new(3));
    assert_eq!(session.version(), WorkflowVersion::new(3));
    assert!(result.diagnostics().is_empty());
    assert_eq!(
        result.history_entry_id(),
        Some(&reimagine_core::model::HistoryEntryId::new(
            "history:batch-apply"
        ))
    );
    assert_eq!(session.history().entries().len(), 1);
    assert_eq!(session.history().cursor(), 1);

    let node = session
        .workflow()
        .nodes()
        .iter()
        .find(|node| node.id() == &NodeId::new("node_consumer"))
        .expect("consumer node present");
    assert_eq!(node.label(), Some("Sink"));
    assert_eq!(
        node.params().get(&SlotId::new("count")),
        Some(&ParamValue::Integer(9))
    );
    assert_eq!(session.workflow().metadata(), &metadata);
    assert_eq!(
        session
            .workflow()
            .layout()
            .nodes()
            .get(&NodeId::new("node_consumer")),
        Some(&Position::new(300.0, 120.0))
    );

    assert_eq!(
        result.changes(),
        &[
            WorkflowChange::ParamSet {
                node_id: NodeId::new("node_consumer"),
                slot_id: SlotId::new("count"),
                before: Some(ParamValue::Integer(5)),
                after: ParamValue::Integer(9),
            },
            WorkflowChange::NodeMoved {
                node_id: NodeId::new("node_consumer"),
                before: Some(Position::new(180.0, 40.0)),
                after: Some(Position::new(300.0, 120.0)),
            },
            WorkflowChange::NodeLabelSet {
                node_id: NodeId::new("node_consumer"),
                before: Some("Consumer".to_owned()),
                after: Some("Sink".to_owned()),
            },
            WorkflowChange::WorkflowMetadataSet {
                before: WorkflowMetadata::new(),
                after: metadata.clone(),
            },
            WorkflowChange::VersionAdvanced {
                before: WorkflowVersion::new(2),
                after: WorkflowVersion::new(3),
            },
        ]
    );

    let entry = &session.history().entries()[0];
    assert_eq!(entry.command_batch(), &batch);
    assert_eq!(entry.before().layout(), &original_layout);
    assert_eq!(entry.after(), session.workflow());
    assert_eq!(entry.forward_changes(), result.changes());
    assert_eq!(entry.created_at(), &Timestamp::new("2026-06-09T11:00:00Z"));
}

#[test]
fn apply_batch_is_atomic_when_any_command_is_invalid() {
    let catalog = test_catalog();
    let workflow = workflow_with_nodes();
    let snapshot = workflow.clone();
    let mut session = WorkflowSession::new(workflow);

    let result = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-invalid"),
            CommandActor::new(CommandActorKind::Human),
            WorkflowVersion::new(2),
            CommandProvenance::Direct,
            Timestamp::new("2026-06-09T12:00:00Z"),
            vec![
                WorkflowCommand::SetParam {
                    node_id: NodeId::new("node_consumer"),
                    slot_id: SlotId::new("count"),
                    value: ParamValue::Integer(8),
                },
                WorkflowCommand::SetParam {
                    node_id: NodeId::new("node_consumer"),
                    slot_id: SlotId::new("dynamic_value"),
                    value: ParamValue::String("should fail".to_owned()),
                },
            ],
        ),
    );

    assert_eq!(result.status(), CommandResultStatus::Rejected);
    assert!(result.changes().is_empty());
    assert!(result.history_entry_id().is_none());
    assert_eq!(session.workflow(), &snapshot);
    assert_eq!(session.version(), WorkflowVersion::new(2));
    assert!(session.history().entries().is_empty());
    assert!(
        result
            .diagnostics()
            .iter()
            .any(|diag| diag.code().as_str() == "CORE/WORKFLOW_PARAM_ON_DYNAMIC_SLOT")
    );
}

#[test]
fn add_node_duplicate_then_remove_node_self_heal_rejects_without_mutation() {
    let catalog = test_catalog();
    let workflow = workflow_with_nodes();
    let snapshot = workflow.clone();
    let mut session = WorkflowSession::new(workflow);

    let result = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-dup-node"),
            CommandActor::new(CommandActorKind::Human),
            WorkflowVersion::new(2),
            CommandProvenance::Direct,
            Timestamp::new("2026-06-09T12:15:00Z"),
            vec![
                WorkflowCommand::AddNode {
                    node_id: NodeId::new("node_consumer"),
                    type_id: NodeTypeId::new("builtin.consumer"),
                    label: Some("Duplicate".to_owned()),
                    params: BTreeMap::new(),
                    position: Some(Position::new(999.0, 999.0)),
                },
                WorkflowCommand::RemoveNode {
                    node_id: NodeId::new("node_consumer"),
                },
            ],
        ),
    );

    assert_eq!(result.status(), CommandResultStatus::Rejected);
    assert!(result.changes().is_empty());
    assert_eq!(session.workflow(), &snapshot);
    assert!(session.history().entries().is_empty());
    assert!(
        result
            .diagnostics()
            .iter()
            .any(|diag| diag.code().as_str() == "CORE/WORKFLOW_NODE_DUPLICATE")
    );
}

#[test]
fn connect_duplicate_then_disconnect_self_heal_rejects_without_mutation() {
    let catalog = test_catalog();
    let workflow = workflow_with_static_edge_override();
    let snapshot = workflow.clone();
    let mut session = WorkflowSession::new(workflow);

    let result = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-dup-edge"),
            CommandActor::new(CommandActorKind::Human),
            WorkflowVersion::new(6),
            CommandProvenance::Direct,
            Timestamp::new("2026-06-09T12:20:00Z"),
            vec![
                WorkflowCommand::Connect {
                    edge_id: EdgeId::new("edge-count"),
                    from: Endpoint::node_slot(NodeId::new("node_int_source"), SlotId::new("value")),
                    to: Endpoint::node_slot(NodeId::new("node_consumer"), SlotId::new("count")),
                },
                WorkflowCommand::Disconnect {
                    edge_id: EdgeId::new("edge-count"),
                },
            ],
        ),
    );

    assert_eq!(result.status(), CommandResultStatus::Rejected);
    assert!(result.changes().is_empty());
    assert_eq!(session.workflow(), &snapshot);
    assert!(session.history().entries().is_empty());
    assert!(
        result
            .diagnostics()
            .iter()
            .any(|diag| diag.code().as_str() == "CORE/WORKFLOW_EDGE_DUPLICATE")
    );
}

#[test]
fn apply_batch_rejects_version_conflicts_without_mutation() {
    let catalog = test_catalog();
    let workflow = workflow_with_nodes();
    let snapshot = workflow.clone();
    let mut session = WorkflowSession::new(workflow);

    let result = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-conflict"),
            CommandActor::new(CommandActorKind::Human),
            WorkflowVersion::new(1),
            CommandProvenance::Direct,
            Timestamp::new("2026-06-09T13:00:00Z"),
            vec![WorkflowCommand::SetNodeLabel {
                node_id: NodeId::new("node_consumer"),
                label: Some("Mismatch".to_owned()),
            }],
        ),
    );

    assert_eq!(result.status(), CommandResultStatus::Rejected);
    assert!(result.changes().is_empty());
    assert_eq!(session.workflow(), &snapshot);
    assert_eq!(session.version(), WorkflowVersion::new(2));
    assert!(session.history().entries().is_empty());
    assert!(
        result
            .diagnostics()
            .iter()
            .any(|diag| diag.code().as_str() == "CORE/WORKFLOW_VERSION_CONFLICT")
    );
}

#[test]
fn apply_batch_returns_noop_without_history_when_nothing_changes() {
    let catalog = test_catalog();
    let mut session = WorkflowSession::new(workflow_with_nodes());

    let result = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-noop"),
            CommandActor::new(CommandActorKind::System),
            WorkflowVersion::new(2),
            CommandProvenance::Migration {
                from_schema_version: "reimagine.workflow.v1".to_owned(),
            },
            Timestamp::new("2026-06-09T14:00:00Z"),
            vec![
                WorkflowCommand::MoveNode {
                    node_id: NodeId::new("node_source"),
                    position: Position::new(20.0, 40.0),
                },
                WorkflowCommand::SetNodeLabel {
                    node_id: NodeId::new("node_consumer"),
                    label: Some("Consumer".to_owned()),
                },
            ],
        ),
    );

    assert_eq!(result.status(), CommandResultStatus::NoOp);
    assert!(result.changes().is_empty());
    assert!(result.diagnostics().is_empty());
    assert_eq!(result.workflow_version(), WorkflowVersion::new(2));
    assert!(result.history_entry_id().is_none());
    assert!(session.history().entries().is_empty());
    assert_eq!(session.version(), WorkflowVersion::new(2));
}

#[test]
fn move_node_can_add_a_missing_layout_entry() {
    let catalog = test_catalog();
    let mut session = WorkflowSession::new(workflow_without_consumer_layout());

    let result = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-move-without-layout"),
            CommandActor::new(CommandActorKind::Human),
            WorkflowVersion::new(2),
            CommandProvenance::Direct,
            Timestamp::new("2026-06-09T14:30:00Z"),
            vec![WorkflowCommand::MoveNode {
                node_id: NodeId::new("node_consumer"),
                position: Position::new(222.0, 88.0),
            }],
        ),
    );

    assert_eq!(result.status(), CommandResultStatus::Applied);
    assert_eq!(
        session
            .workflow()
            .layout()
            .nodes()
            .get(&NodeId::new("node_consumer")),
        Some(&Position::new(222.0, 88.0))
    );
    assert_eq!(
        result.changes(),
        &[
            WorkflowChange::NodeMoved {
                node_id: NodeId::new("node_consumer"),
                before: None,
                after: Some(Position::new(222.0, 88.0)),
            },
            WorkflowChange::VersionAdvanced {
                before: WorkflowVersion::new(2),
                after: WorkflowVersion::new(3),
            },
        ]
    );
}

#[test]
fn set_param_updates_static_slot_fallback_even_when_an_edge_currently_overrides_it() {
    let catalog = test_catalog();
    let mut session = WorkflowSession::new(workflow_with_static_edge_override());

    let result = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-static-fallback"),
            CommandActor::new(CommandActorKind::Agent),
            WorkflowVersion::new(6),
            CommandProvenance::Direct,
            Timestamp::new("2026-06-09T14:45:00Z"),
            vec![WorkflowCommand::SetParam {
                node_id: NodeId::new("node_consumer"),
                slot_id: SlotId::new("count"),
                value: ParamValue::Integer(12),
            }],
        ),
    );

    assert_eq!(result.status(), CommandResultStatus::Applied);
    assert_eq!(result.diagnostics(), &[]);
    assert_eq!(session.workflow().edges().len(), 1);
    assert_eq!(
        session
            .workflow()
            .nodes()
            .iter()
            .find(|node| node.id() == &NodeId::new("node_consumer"))
            .and_then(|node| node.params().get(&SlotId::new("count"))),
        Some(&ParamValue::Integer(12))
    );
    assert_eq!(
        result.changes(),
        &[
            WorkflowChange::ParamSet {
                node_id: NodeId::new("node_consumer"),
                slot_id: SlotId::new("count"),
                before: Some(ParamValue::Integer(4)),
                after: ParamValue::Integer(12),
            },
            WorkflowChange::VersionAdvanced {
                before: WorkflowVersion::new(6),
                after: WorkflowVersion::new(7),
            },
        ]
    );
}

#[test]
fn apply_batch_supports_remove_disconnect_apply_layout_and_remove_param() {
    let catalog = test_catalog();
    let workflow = workflow_with_connected_fallback();
    let mut session = WorkflowSession::new(workflow);

    let new_layout = WorkflowLayout::new()
        .with_node(NodeId::new("node_source"), Position::new(8.0, 9.0))
        .with_node(NodeId::new("node_consumer"), Position::new(20.0, 30.0));

    let result = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-layout"),
            CommandActor::new(CommandActorKind::Importer).with_label("loader"),
            WorkflowVersion::new(4),
            CommandProvenance::Import {
                format: "json".to_owned(),
                source: Some("fixtures/connected.json".to_owned()),
            },
            Timestamp::new("2026-06-09T15:00:00Z"),
            vec![
                WorkflowCommand::Disconnect {
                    edge_id: EdgeId::new("edge-live"),
                },
                WorkflowCommand::RemoveParam {
                    node_id: NodeId::new("node_consumer"),
                    slot_id: SlotId::new("count"),
                },
                WorkflowCommand::ApplyLayout {
                    layout: new_layout.clone(),
                },
                WorkflowCommand::RemoveNode {
                    node_id: NodeId::new("node_source"),
                },
            ],
        ),
    );

    assert_eq!(result.status(), CommandResultStatus::Applied);
    assert_eq!(result.workflow_version(), WorkflowVersion::new(5));
    assert_eq!(session.workflow().nodes().len(), 1);
    assert_eq!(session.workflow().edges().len(), 0);
    assert_eq!(
        session.workflow().layout(),
        &WorkflowLayout::new().with_node(NodeId::new("node_consumer"), Position::new(20.0, 30.0))
    );
    assert_eq!(
        result.changes(),
        &[
            WorkflowChange::EdgeRemoved {
                edge: WorkflowEdge::new(
                    EdgeId::new("edge-live"),
                    Endpoint::node_slot(NodeId::new("node_source"), SlotId::new("value")),
                    Endpoint::node_slot(NodeId::new("node_consumer"), SlotId::new("dynamic_value")),
                ),
            },
            WorkflowChange::ParamRemoved {
                node_id: NodeId::new("node_consumer"),
                slot_id: SlotId::new("count"),
                before: ParamValue::Integer(4),
            },
            WorkflowChange::LayoutApplied {
                before: WorkflowLayout::new()
                    .with_node(NodeId::new("node_source"), Position::new(0.0, 0.0))
                    .with_node(NodeId::new("node_consumer"), Position::new(20.0, 30.0)),
                after: new_layout,
            },
            WorkflowChange::NodeRemoved {
                node: WorkflowNode::new(
                    NodeId::new("node_source"),
                    NodeTypeId::new("builtin.source"),
                )
                .with_label("Source"),
                removed_edges: vec![],
                removed_layout: Some(Position::new(8.0, 9.0)),
            },
            WorkflowChange::VersionAdvanced {
                before: WorkflowVersion::new(4),
                after: WorkflowVersion::new(5),
            },
        ]
    );
}

#[test]
fn undo_and_redo_restore_snapshots_and_advance_version() {
    let catalog = test_catalog();
    let mut session = WorkflowSession::new(workflow_with_nodes());

    let applied = session.apply_batch(
        &catalog,
        CommandBatch::new(
            CommandBatchId::new("batch-undo"),
            CommandActor::new(CommandActorKind::Human),
            WorkflowVersion::new(2),
            CommandProvenance::Direct,
            Timestamp::new("2026-06-09T16:00:00Z"),
            vec![
                WorkflowCommand::SetParam {
                    node_id: NodeId::new("node_consumer"),
                    slot_id: SlotId::new("count"),
                    value: ParamValue::Integer(10),
                },
                WorkflowCommand::MoveNode {
                    node_id: NodeId::new("node_consumer"),
                    position: Position::new(250.0, 90.0),
                },
            ],
        ),
    );
    assert_eq!(applied.status(), CommandResultStatus::Applied);
    let after_apply = session.workflow().clone();

    let undo = session.undo().expect("undo result");
    assert_eq!(undo.status(), CommandResultStatus::Applied);
    assert_eq!(undo.workflow_version(), WorkflowVersion::new(4));
    assert_eq!(session.version(), WorkflowVersion::new(4));
    assert_eq!(session.history().cursor(), 0);
    assert_eq!(
        session
            .workflow()
            .nodes()
            .iter()
            .find(|node| node.id() == &NodeId::new("node_consumer"))
            .and_then(|node| node.params().get(&SlotId::new("count"))),
        Some(&ParamValue::Integer(5))
    );
    assert_eq!(
        undo.changes(),
        &[
            WorkflowChange::NodeMoved {
                node_id: NodeId::new("node_consumer"),
                before: Some(Position::new(250.0, 90.0)),
                after: Some(Position::new(180.0, 40.0)),
            },
            WorkflowChange::ParamSet {
                node_id: NodeId::new("node_consumer"),
                slot_id: SlotId::new("count"),
                before: Some(ParamValue::Integer(10)),
                after: ParamValue::Integer(5),
            },
            WorkflowChange::VersionAdvanced {
                before: WorkflowVersion::new(3),
                after: WorkflowVersion::new(4),
            },
        ]
    );

    let redo = session.redo().expect("redo result");
    assert_eq!(redo.status(), CommandResultStatus::Applied);
    assert_eq!(redo.workflow_version(), WorkflowVersion::new(5));
    assert_eq!(session.version(), WorkflowVersion::new(5));
    assert_eq!(session.history().cursor(), 1);
    assert_eq!(session.workflow().id(), after_apply.id());
    assert_eq!(session.workflow().metadata(), after_apply.metadata());
    assert_eq!(session.workflow().nodes(), after_apply.nodes());
    assert_eq!(session.workflow().edges(), after_apply.edges());
    assert_eq!(session.workflow().layout(), after_apply.layout());
    assert_eq!(
        redo.changes(),
        &[
            WorkflowChange::ParamSet {
                node_id: NodeId::new("node_consumer"),
                slot_id: SlotId::new("count"),
                before: Some(ParamValue::Integer(5)),
                after: ParamValue::Integer(10),
            },
            WorkflowChange::NodeMoved {
                node_id: NodeId::new("node_consumer"),
                before: Some(Position::new(180.0, 40.0)),
                after: Some(Position::new(250.0, 90.0)),
            },
            WorkflowChange::VersionAdvanced {
                before: WorkflowVersion::new(4),
                after: WorkflowVersion::new(5),
            },
        ]
    );
}

#[derive(Clone)]
struct TestCatalog {
    defs: HashMap<NodeTypeId, NodeDef>,
}

impl TestCatalog {
    fn new(defs: Vec<NodeDef>) -> Self {
        Self {
            defs: defs
                .into_iter()
                .map(|def| (def.type_id().clone(), def))
                .collect(),
        }
    }
}

impl NodeCatalog for TestCatalog {
    fn get(&self, type_id: &NodeTypeId) -> Option<&NodeDef> {
        self.defs.get(type_id)
    }
}

fn test_catalog() -> TestCatalog {
    use reimagine_core::model::{InputSlotDef, OutputSlotDef};

    TestCatalog::new(vec![
        NodeDef::new("builtin.source", "Source", "test")
            .with_output_slot(OutputSlotDef::new("value", SlotKind::String).required(true)),
        NodeDef::new("builtin.int_source", "Int Source", "test")
            .with_output_slot(OutputSlotDef::new("value", SlotKind::Integer).required(true)),
        NodeDef::new("builtin.consumer", "Consumer", "test")
            .with_input_slot(
                InputSlotDef::new("dynamic_value", SlotKind::String)
                    .dynamic(true)
                    .required(true),
            )
            .with_input_slot(
                InputSlotDef::new("count", SlotKind::Integer)
                    .required(true)
                    .with_default_value(ParamValue::Integer(1)),
            ),
    ])
}

fn empty_workflow(id: &str, version: u64) -> Workflow {
    Workflow::new(id, WorkflowVersion::new(version))
}

fn workflow_with_nodes() -> Workflow {
    Workflow::new("workflow-main", WorkflowVersion::new(2))
        .with_node(
            WorkflowNode::new(
                NodeId::new("node_source"),
                NodeTypeId::new("builtin.source"),
            )
            .with_label("Source"),
        )
        .with_node(
            WorkflowNode::new(
                NodeId::new("node_consumer"),
                NodeTypeId::new("builtin.consumer"),
            )
            .with_label("Consumer")
            .with_param(SlotId::new("count"), ParamValue::Integer(5)),
        )
        .with_layout(
            WorkflowLayout::new()
                .with_node(NodeId::new("node_source"), Position::new(20.0, 40.0))
                .with_node(NodeId::new("node_consumer"), Position::new(180.0, 40.0)),
        )
}

fn workflow_without_consumer_layout() -> Workflow {
    Workflow::new("workflow-no-layout", WorkflowVersion::new(2))
        .with_node(
            WorkflowNode::new(
                NodeId::new("node_source"),
                NodeTypeId::new("builtin.source"),
            )
            .with_label("Source"),
        )
        .with_node(
            WorkflowNode::new(
                NodeId::new("node_consumer"),
                NodeTypeId::new("builtin.consumer"),
            )
            .with_label("Consumer")
            .with_param(SlotId::new("count"), ParamValue::Integer(5)),
        )
        .with_layout(
            WorkflowLayout::new().with_node(NodeId::new("node_source"), Position::new(20.0, 40.0)),
        )
}

fn workflow_with_connected_fallback() -> Workflow {
    Workflow::new("workflow-connected", WorkflowVersion::new(4))
        .with_node(
            WorkflowNode::new(
                NodeId::new("node_source"),
                NodeTypeId::new("builtin.source"),
            )
            .with_label("Source"),
        )
        .with_node(
            WorkflowNode::new(
                NodeId::new("node_consumer"),
                NodeTypeId::new("builtin.consumer"),
            )
            .with_label("Consumer")
            .with_param(SlotId::new("count"), ParamValue::Integer(4)),
        )
        .with_edge(WorkflowEdge::new(
            EdgeId::new("edge-live"),
            Endpoint::node_slot(NodeId::new("node_source"), SlotId::new("value")),
            Endpoint::node_slot(NodeId::new("node_consumer"), SlotId::new("dynamic_value")),
        ))
        .with_layout(
            WorkflowLayout::new()
                .with_node(NodeId::new("node_source"), Position::new(0.0, 0.0))
                .with_node(NodeId::new("node_consumer"), Position::new(20.0, 30.0)),
        )
}

fn workflow_with_static_edge_override() -> Workflow {
    Workflow::new("workflow-static-edge", WorkflowVersion::new(6))
        .with_node(
            WorkflowNode::new(
                NodeId::new("node_int_source"),
                NodeTypeId::new("builtin.int_source"),
            )
            .with_label("Int Source"),
        )
        .with_node(
            WorkflowNode::new(
                NodeId::new("node_consumer"),
                NodeTypeId::new("builtin.consumer"),
            )
            .with_label("Consumer")
            .with_param(SlotId::new("count"), ParamValue::Integer(4)),
        )
        .with_edge(WorkflowEdge::new(
            EdgeId::new("edge-count"),
            Endpoint::node_slot(NodeId::new("node_int_source"), SlotId::new("value")),
            Endpoint::node_slot(NodeId::new("node_consumer"), SlotId::new("count")),
        ))
        .with_layout(
            WorkflowLayout::new()
                .with_node(NodeId::new("node_int_source"), Position::new(10.0, 10.0))
                .with_node(NodeId::new("node_consumer"), Position::new(120.0, 10.0)),
        )
}
