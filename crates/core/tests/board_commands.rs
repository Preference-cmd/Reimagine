use reimagine_core::board::{
    BoardCommand, BoardDocument, BoardItemKind, BoardItemPosition, BoardItemSize,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, CommandStatus,
};
use reimagine_core::document::DocumentSession;
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    BoardId, BoardItemId, CommandBatchId, DocumentVersion, ProjectId, WorkflowId,
};

fn test_board() -> BoardDocument {
    BoardDocument::new(
        BoardId::new("test-board"),
        ProjectId::new("test-project"),
        DocumentVersion::new(0),
    )
}

fn test_batch(
    board_version: DocumentVersion,
    commands: Vec<BoardCommand>,
) -> CommandBatch<BoardCommand> {
    CommandBatch::new(
        CommandBatchId::new("batch-1"),
        CommandActor::new(CommandActorKind::Human),
        board_version,
        CommandProvenance::Direct,
        Timestamp::new("2026-01-01T00:00:00Z"),
        commands,
    )
}

#[test]
fn preview_does_not_modify_state() {
    let session = DocumentSession::new(test_board());
    let batch = test_batch(
        DocumentVersion::new(0),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("item-1"),
            kind: BoardItemKind::Note {
                content: "Test".to_string(),
            },
            position: BoardItemPosition::new(0, 0),
            size: BoardItemSize::new(100, 100),
        }],
    );

    let result = session.preview_batch(&(), batch);
    assert_eq!(result.status(), CommandStatus::Applied);
    assert!(session.document().items().is_empty());
    assert_eq!(session.version(), DocumentVersion::new(0));
}

#[test]
fn apply_add_item() {
    let mut session = DocumentSession::new(test_board());
    let batch = test_batch(
        DocumentVersion::new(0),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("item-1"),
            kind: BoardItemKind::Note {
                content: "Test".to_string(),
            },
            position: BoardItemPosition::new(10, 20),
            size: BoardItemSize::new(100, 100),
        }],
    );

    let result = session.apply_batch(&(), batch);
    assert_eq!(result.status(), CommandStatus::Applied);
    assert_eq!(session.version(), DocumentVersion::new(1));
    assert_eq!(session.document().items().len(), 1);

    let item = session
        .document()
        .item(&BoardItemId::new("item-1"))
        .unwrap();
    assert_eq!(item.position().x, 10);
    assert_eq!(item.position().y, 20);
}

#[test]
fn version_conflict_rejection() {
    let mut session = DocumentSession::new(test_board());
    let batch = test_batch(
        DocumentVersion::new(5), // Wrong version
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("item-1"),
            kind: BoardItemKind::Note {
                content: "Test".to_string(),
            },
            position: BoardItemPosition::new(0, 0),
            size: BoardItemSize::new(100, 100),
        }],
    );

    let result = session.apply_batch(&(), batch);
    assert_eq!(result.status(), CommandStatus::Rejected);
    assert_eq!(result.diagnostics().len(), 1);
    assert_eq!(
        result.diagnostics()[0].code().as_str(),
        "CORE/BOARD_VERSION_CONFLICT"
    );
}

#[test]
fn undo_redo() {
    let mut session = DocumentSession::new(test_board());

    // Apply
    let batch1 = test_batch(
        DocumentVersion::new(0),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("item-1"),
            kind: BoardItemKind::Note {
                content: "Test".to_string(),
            },
            position: BoardItemPosition::new(0, 0),
            size: BoardItemSize::new(100, 100),
        }],
    );
    session.apply_batch(&(), batch1);
    assert_eq!(session.document().items().len(), 1);

    // Undo
    let undo_result = session.undo();
    assert!(undo_result.is_some());
    assert_eq!(session.document().items().len(), 0);

    // Redo
    let redo_result = session.redo();
    assert!(redo_result.is_some());
    assert_eq!(session.document().items().len(), 1);
}

#[test]
fn move_item() {
    let mut session = DocumentSession::new(test_board());

    // Add item
    let batch1 = test_batch(
        DocumentVersion::new(0),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("item-1"),
            kind: BoardItemKind::Note {
                content: "Test".to_string(),
            },
            position: BoardItemPosition::new(0, 0),
            size: BoardItemSize::new(100, 100),
        }],
    );
    session.apply_batch(&(), batch1);

    // Move item
    let batch2 = test_batch(
        DocumentVersion::new(1),
        vec![BoardCommand::MoveItem {
            item_id: BoardItemId::new("item-1"),
            position: BoardItemPosition::new(50, 60),
        }],
    );
    session.apply_batch(&(), batch2);

    let item = session
        .document()
        .item(&BoardItemId::new("item-1"))
        .unwrap();
    assert_eq!(item.position().x, 50);
    assert_eq!(item.position().y, 60);

    // Undo move
    session.undo();
    let item = session
        .document()
        .item(&BoardItemId::new("item-1"))
        .unwrap();
    assert_eq!(item.position().x, 0);
    assert_eq!(item.position().y, 0);
}

#[test]
fn lock_item_prevents_modification() {
    let mut session = DocumentSession::new(test_board());

    // Add item
    let batch1 = test_batch(
        DocumentVersion::new(0),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("item-1"),
            kind: BoardItemKind::Note {
                content: "Test".to_string(),
            },
            position: BoardItemPosition::new(0, 0),
            size: BoardItemSize::new(100, 100),
        }],
    );
    session.apply_batch(&(), batch1);

    // Lock item
    let batch2 = test_batch(
        DocumentVersion::new(1),
        vec![BoardCommand::LockItem {
            item_id: BoardItemId::new("item-1"),
            locked: true,
        }],
    );
    session.apply_batch(&(), batch2);

    // Try to move locked item
    let batch3 = test_batch(
        DocumentVersion::new(2),
        vec![BoardCommand::MoveItem {
            item_id: BoardItemId::new("item-1"),
            position: BoardItemPosition::new(50, 50),
        }],
    );
    let result = session.apply_batch(&(), batch3);
    assert_eq!(result.status(), CommandStatus::Rejected);
    assert_eq!(
        result.diagnostics()[0].code().as_str(),
        "CORE/BOARD_ITEM_LOCKED"
    );
}

#[test]
fn update_note() {
    let mut session = DocumentSession::new(test_board());

    // Add note
    let batch1 = test_batch(
        DocumentVersion::new(0),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("note-1"),
            kind: BoardItemKind::Note {
                content: "Original".to_string(),
            },
            position: BoardItemPosition::new(0, 0),
            size: BoardItemSize::new(100, 100),
        }],
    );
    session.apply_batch(&(), batch1);

    // Update note
    let batch2 = test_batch(
        DocumentVersion::new(1),
        vec![BoardCommand::UpdateNote {
            item_id: BoardItemId::new("note-1"),
            content: "Updated content".to_string(),
        }],
    );
    session.apply_batch(&(), batch2);

    let item = session
        .document()
        .item(&BoardItemId::new("note-1"))
        .unwrap();
    match item.kind() {
        BoardItemKind::Note { content } => assert_eq!(content, "Updated content"),
        _ => panic!("Expected note"),
    }

    // Undo update
    session.undo();
    let item = session
        .document()
        .item(&BoardItemId::new("note-1"))
        .unwrap();
    match item.kind() {
        BoardItemKind::Note { content } => assert_eq!(content, "Original"),
        _ => panic!("Expected note"),
    }
}

#[test]
fn update_non_note_fails() {
    let mut session = DocumentSession::new(test_board());

    // Add workflow ref
    let batch1 = test_batch(
        DocumentVersion::new(0),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("wf-1"),
            kind: BoardItemKind::WorkflowRef {
                workflow_id: WorkflowId::new("wf-test"),
            },
            position: BoardItemPosition::new(0, 0),
            size: BoardItemSize::new(100, 100),
        }],
    );
    session.apply_batch(&(), batch1);

    // Try to update as note
    let batch2 = test_batch(
        DocumentVersion::new(1),
        vec![BoardCommand::UpdateNote {
            item_id: BoardItemId::new("wf-1"),
            content: "Should fail".to_string(),
        }],
    );
    let result = session.apply_batch(&(), batch2);
    assert_eq!(result.status(), CommandStatus::Rejected);
    assert_eq!(
        result.diagnostics()[0].code().as_str(),
        "CORE/BOARD_ITEM_NOT_NOTE"
    );
}

#[test]
fn unified_document_session_works_for_both_board_and_workflow() {
    use reimagine_core::command::WorkflowCommand;
    use reimagine_core::model::{NodeCatalog, NodeDef, NodeId, NodeTypeId};
    use reimagine_core::workflow::Workflow;
    use std::collections::{BTreeMap, HashMap};

    struct TestCatalog(HashMap<NodeTypeId, NodeDef>);
    impl NodeCatalog for TestCatalog {
        fn get(&self, type_id: &NodeTypeId) -> Option<&NodeDef> {
            self.0.get(type_id)
        }
    }

    // 1. Board document in DocumentSession
    let mut board_session = DocumentSession::new(test_board());
    let board_batch = test_batch(
        DocumentVersion::new(0),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("item-unified"),
            kind: BoardItemKind::Note {
                content: "Unified".to_string(),
            },
            position: BoardItemPosition::new(5, 5),
            size: BoardItemSize::new(50, 50),
        }],
    );
    let board_res = board_session.apply_batch(&(), board_batch);
    assert_eq!(board_res.status(), CommandStatus::Applied);
    assert_eq!(board_session.version(), DocumentVersion::new(1));
    assert_eq!(board_session.document().items().len(), 1);

    // 2. Workflow document in DocumentSession
    let mut catalog_map = HashMap::new();
    catalog_map.insert(
        NodeTypeId::new("test.node"),
        NodeDef::new("test.node", "Test Node", "test"),
    );
    let catalog = TestCatalog(catalog_map);

    let workflow = Workflow::new("wf-unified", DocumentVersion::new(0));
    let mut wf_session = DocumentSession::new(workflow);
    let wf_batch = CommandBatch::new(
        CommandBatchId::new("wf-batch-1"),
        CommandActor::new(CommandActorKind::Human),
        DocumentVersion::new(0),
        CommandProvenance::Direct,
        Timestamp::new("2026-01-01T00:00:00Z"),
        vec![WorkflowCommand::AddNode {
            node_id: NodeId::new("node-1"),
            type_id: NodeTypeId::new("test.node"),
            label: Some("Test".to_string()),
            params: BTreeMap::new(),
            position: None,
        }],
    );
    let wf_res = wf_session.apply_batch(&catalog, wf_batch);
    assert_eq!(wf_res.status(), CommandStatus::Applied);
    assert_eq!(wf_session.version(), DocumentVersion::new(1));
    assert_eq!(wf_session.document().nodes().len(), 1);
}
