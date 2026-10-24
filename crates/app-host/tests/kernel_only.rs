#![cfg(not(feature = "aigc"))]

use reimagine_app_host::{RunWorkflowRequest, WorkspaceHostBuilder, WorkspaceScope};
use reimagine_core::board::{BoardCommand, BoardItemKind, BoardItemPosition, BoardItemSize};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, OperationContext,
};
use reimagine_core::event::Timestamp;
use reimagine_core::health::Health;
use reimagine_core::model::{
    BoardItemId, CommandBatchId, DocumentVersion, NodeId, NodeTypeId, ProjectId, WorkflowId,
};
use reimagine_core::project::ProjectMetadata;
use reimagine_core::readiness::RunTargetSelection;
use reimagine_core::workflow::{Workflow, WorkflowNode};
use reimagine_runtime::{RunInputs, RuntimeOptions};

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-kernel-{prefix}-{nonce}"))
}

#[tokio::test]
async fn kernel_only_workspace_reports_not_installed_and_rejects_workflow_run() {
    let base = temp_dir("kernel-run");
    let host = WorkspaceHostBuilder::new(WorkspaceScope::new("test-kernel"), &base)
        .build()
        .await
        .expect("build kernel workspace host");

    // 1. Verify Health is Unavailable with AIGC/NOT_INSTALLED
    assert!(!host.aigc_health().is_available());
    let diagnostic = match host.aigc_health() {
        Health::Unavailable(diag) => diag,
        Health::Available => panic!("expected aigc to be unavailable in kernel-only build"),
    };
    assert_eq!(diagnostic.code().as_str(), "AIGC/NOT_INSTALLED");
    assert_eq!(diagnostic.message(), "AIGC is not installed");
    assert!(!diagnostic.message().is_empty());

    // 2. Create project
    let project_id = ProjectId::new("proj-kernel");
    let metadata = ProjectMetadata::new(
        "Kernel Project",
        "Test project in kernel-only build",
        Timestamp::new("2026-06-12T00:00:00Z"),
        Timestamp::new("2026-06-12T00:00:00Z"),
    );
    host.project_service()
        .create_project(project_id.clone(), metadata)
        .await
        .expect("create project");

    // 3. Create board item
    let batch = CommandBatch::new(
        CommandBatchId::new("batch-1"),
        OperationContext::new(
            CommandActor::new(CommandActorKind::Human),
            CommandProvenance::Direct,
        ),
        DocumentVersion::new(0),
        Timestamp::new("2026-06-12T00:00:00Z"),
        vec![BoardCommand::AddItem {
            item_id: BoardItemId::new("item-1"),
            kind: BoardItemKind::WorkflowRef {
                workflow_id: WorkflowId::new("wf-kernel"),
            },
            position: BoardItemPosition::new(100, 200),
            size: BoardItemSize::new(300, 150),
        }],
    );
    host.board_service()
        .apply_batch(&project_id, batch)
        .await
        .expect("apply board command");

    // 4. Create workflow with at least one node (non-empty)
    let workflow_id = WorkflowId::new("wf-kernel");
    let workflow = Workflow::new(workflow_id.clone(), DocumentVersion::new(1)).with_node(
        WorkflowNode::new(NodeId::new("node-1"), NodeTypeId::new("test.node")),
    );
    let wf_service = host.workflow_service_for_project(&project_id);
    wf_service
        .save_external_workflow(workflow)
        .await
        .expect("save workflow");
    wf_service
        .load_workflow(&workflow_id)
        .await
        .expect("load workflow");

    // 5. Run non-empty workflow -> must fail with code "aigc_unavailable"
    let request = RunWorkflowRequest {
        project_id,
        workflow_id,
        target_selection: RunTargetSelection::AllDefaultTargets,
        run_inputs: RunInputs::new(),
        options: RuntimeOptions::default(),
        correlation_id: None,
    };
    let err = host
        .run_workflow(request)
        .await
        .expect_err("non-empty workflow run must fail in kernel-only build");

    assert_eq!(err.code().as_str(), "aigc_unavailable");

    let _ = std::fs::remove_dir_all(&base);
}
