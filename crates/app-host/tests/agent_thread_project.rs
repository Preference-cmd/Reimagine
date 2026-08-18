//! AR-14: AgentThread binding to ProjectId.
//!
//! Verifies project-scoped threads: create_thread binds the session,
//! list_threads filters the registry by project, and workspace tools
//! reject calls from a thread bound to a different project (the
//! workspace service is scoped to the default project).

use std::sync::Arc;

use reimagine_agent_harness::{
    AgentMode, AgentSessionId, PermissionSet, ProviderName, ToolContext, ToolName, ToolPermission,
    WorkspaceScope,
};
use reimagine_app_host::{AgentProviderCatalog, AgentService, WorkspaceHost};
use reimagine_core::model::ProjectId;
use reimagine_core::model::WorkflowVersion;
use reimagine_core::workflow::Workflow;
use serde_json::json;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-ar14-{prefix}-{nonce}"))
}

fn make_service(base: &std::path::Path) -> (AgentService, Arc<WorkspaceHost>) {
    let host = Arc::new(WorkspaceHost::with_defaults(
        WorkspaceScope::new("ws-ar14"),
        base,
    ));
    let svc = AgentService::with_registry_providers_and_sink(
        WorkspaceScope::new("ws-ar14"),
        Arc::clone(host.agent_service().registry()),
        AgentProviderCatalog::new(),
        Arc::new(reimagine_agent_harness::VecAgentEventSink::new()),
    );
    (svc, host)
}

#[test]
fn create_thread_binds_project_and_list_filters() {
    let base = temp_dir("list");
    let (svc, _host) = make_service(&base);

    let proj_a = ProjectId::new("proj-a");
    let proj_b = ProjectId::new("proj-b");
    let ta = svc.create_thread(
        proj_a.clone(),
        AgentSessionId::new("t-a"),
        AgentMode::Agent,
        ProviderName::new("openai"),
        "2026-08-18T00:00:00Z",
    );
    assert_eq!(ta.project_id().unwrap(), &proj_a);

    svc.create_thread(
        proj_b.clone(),
        AgentSessionId::new("t-b"),
        AgentMode::Agent,
        ProviderName::new("openai"),
        "2026-08-18T00:00:00Z",
    );

    let a_threads = svc.list_threads(&proj_a);
    assert_eq!(a_threads.len(), 1);
    assert_eq!(a_threads[0].id().as_str(), "t-a");
    assert_eq!(svc.list_threads(&proj_b).len(), 1);
    assert_eq!(svc.list_threads(&ProjectId::new("other")).len(), 0);
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn workspace_tool_rejects_cross_project_thread() {
    let base = temp_dir("cross");
    let (svc, host) = make_service(&base);
    // Register a workflow in the default-project service so the
    // matching (default) case has a target.
    let wf_id = host
        .workflow_service()
        .register_workflow(Workflow::new("wf-ar14", WorkflowVersion::new(0)));

    // Matching project: passes the ownership check.
    svc.create_thread(
        ProjectId::new("default"),
        AgentSessionId::new("t-ok"),
        AgentMode::Agent,
        ProviderName::new("openai"),
        "2026-08-18T00:00:00Z",
    );

    // Cross-project thread: tool boundary must reject.
    let registry = Arc::clone(svc.registry());
    let ctx = ToolContext::new(
        WorkspaceScope::new("ws-ar14"),
        AgentSessionId::new("t-x"),
        AgentMode::Agent,
    )
    .with_permissions(PermissionSet::from_iter([ToolPermission::new(
        "workflow.read",
    )]))
    .with_project_id(ProjectId::new("proj-x"));
    let input = json!({ "workflow_id": wf_id.as_str() });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(registry.invoke(&ToolName::new("workflow.get"), &ctx, input));
    assert!(
        result.is_err(),
        "cross-project workflow.get must be rejected"
    );
    let _ = std::fs::remove_dir_all(&base);
}
