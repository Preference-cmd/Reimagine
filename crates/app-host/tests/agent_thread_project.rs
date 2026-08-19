//! AR-14: AgentThread binding to ProjectId.
//!
//! Verifies project-scoped threads: create_thread binds the session,
//! list_threads filters the registry by project, and workspace tools
//! reject calls from a thread bound to a different project (the
//! workspace service is scoped to the default project).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use reimagine_agent_harness::{
    AgentLoop, AgentMode, AgentProvider, AgentRequest, AgentResponse, AgentSession, AgentSessionId,
    AgentStream, AgentStreamEvent, AgentTurnId, AgentTurnRequest, AgentTurnStopReason, Message,
    ModelInfo, ModelName, PermissionSet, ProviderError, ProviderName, ToolCall, ToolCallId,
    ToolCallStatus, ToolContext, ToolErrorCode, ToolName, ToolPermission, VecAgentEventSink,
    WorkspaceScope,
};
use reimagine_app_host::{
    AgentProviderCatalog, AgentService, AgentServiceTurnRequest, WorkspaceHost,
};
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

// ------------------------------------------------------------------
// AR-42: agent-loop end-to-end coverage for project binding
// ------------------------------------------------------------------

/// Minimal scripted provider: queues one `AgentResponse` per `complete`
/// (and `stream`) call. Same shape as `agent_service_turn.rs`.
struct ScriptedProvider {
    name: ProviderName,
    responses: Mutex<VecDeque<AgentResponse>>,
    requests: Mutex<Vec<AgentRequest>>,
}

impl ScriptedProvider {
    fn new(name: &str, responses: Vec<AgentResponse>) -> Self {
        Self {
            name: ProviderName::new(name),
            responses: Mutex::new(responses.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<AgentRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl AgentProvider for ScriptedProvider {
    fn name(&self) -> ProviderName {
        self.name.clone()
    }

    async fn complete(&self, request: AgentRequest) -> Result<AgentResponse, ProviderError> {
        self.requests.lock().unwrap().push(request);
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| ProviderError::new("SCRIPT_EXHAUSTED", "scripted provider exhausted"))
    }

    async fn stream(&self, request: AgentRequest) -> Result<Box<dyn AgentStream>, ProviderError> {
        self.requests.lock().unwrap().push(request);
        let response =
            self.responses.lock().unwrap().pop_front().ok_or_else(|| {
                ProviderError::new("SCRIPT_EXHAUSTED", "scripted provider exhausted")
            })?;
        let mut events = VecDeque::new();
        if !response.message().content().is_empty() {
            events.push_back(AgentStreamEvent::ContentDelta(
                response.message().content().to_owned(),
            ));
        }
        for tc in response.message().tool_calls() {
            events.push_back(AgentStreamEvent::ToolCall(tc.clone()));
        }
        events.push_back(AgentStreamEvent::Done { stop_reason: None });
        Ok(Box::new(ScriptedStream { events }))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(Vec::new())
    }
}

struct ScriptedStream {
    events: VecDeque<AgentStreamEvent>,
}

#[async_trait]
impl AgentStream for ScriptedStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        self.events.pop_front()
    }
}

/// End-to-end AR-14/AR-42: a session bound to project X drives
/// `workflow.get` through the real `AgentLoop`. The workspace tool's
/// project-ownership check must reject it as a failed observation —
/// proving the full bound -> propagate -> reject chain, not a direct
/// `registry.invoke` call.
#[tokio::test]
async fn agent_loop_end_to_end_rejects_cross_project_get() {
    let base = temp_dir("e2e");
    let host = Arc::new(WorkspaceHost::with_defaults(
        WorkspaceScope::new("ws-ar14"),
        &base,
    ));
    // Target lives in the default project's workflow service.
    let wf_id = host
        .workflow_service()
        .register_workflow(Workflow::new("wf-e2e", WorkflowVersion::new(0)));

    let provider = Arc::new(ScriptedProvider::new(
        "mock",
        vec![
            AgentResponse::new(Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    ToolCallId::new("c1"),
                    "workflow.get",
                    json!({ "workflow_id": wf_id.as_str() }),
                )],
            )),
            AgentResponse::new(Message::assistant("observed")),
        ],
    ));
    let loop_harness = AgentLoop::new(provider.clone(), Arc::new(VecAgentEventSink::new()));

    // Thread bound to proj-x (NOT the default project the service is
    // scoped to), with the workflow.read permission.
    let session = AgentSession::new(
        AgentSessionId::new("sess-e2e"),
        WorkspaceScope::new("ws-ar14"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        Arc::clone(host.agent_service().registry()),
    )
    .with_permissions(PermissionSet::from_iter([ToolPermission::new(
        "workflow.read",
    )]))
    .with_project_id(ProjectId::new("proj-x"));

    let req = AgentTurnRequest::new(
        session,
        AgentTurnId::new("e2e-cross-project"),
        ModelName::new("test-model"),
        vec![Message::user("hi")],
    );
    let result = loop_harness.run_turn(req, None).await;

    // The turn terminates with the final response; the single tool
    // observation is the rejected workflow.get.
    assert_eq!(result.stop_reason(), AgentTurnStopReason::FinalResponse);
    let calls = result.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].status(), ToolCallStatus::Failed);
    let diag = calls[0].diagnostic().expect("rejection diagnostic");
    assert_eq!(
        diag.code(),
        ToolErrorCode::WorkspaceMismatch,
        "cross-project workflow.get must fail with WorkspaceMismatch"
    );
    assert_eq!(
        provider.requests().len(),
        2,
        "one tool round + one final round"
    );
    let _ = std::fs::remove_dir_all(&base);
}

/// AR-14/AR-42: two threads bound to the SAME project can run turns in
/// parallel without a per-session collision (each session owns its own
/// turn permit).
#[tokio::test]
async fn same_project_threads_run_parallel_turns() {
    let base = temp_dir("parallel");
    let (svc, _host) = make_service(&base);
    svc.providers().register(Arc::new(ScriptedProvider::new(
        "mock",
        vec![
            AgentResponse::new(Message::assistant("a")),
            AgentResponse::new(Message::assistant("b")),
        ],
    )));
    svc.create_thread(
        ProjectId::new("proj-p"),
        AgentSessionId::new("t-a"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-08-18T00:00:00Z",
    );
    svc.create_thread(
        ProjectId::new("proj-p"),
        AgentSessionId::new("t-b"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-08-18T00:00:00Z",
    );

    let (a, b) = tokio::join!(
        svc.run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("t-a"),
            AgentTurnId::new("turn-a"),
            ModelName::new("test-model"),
            "hello",
        )),
        svc.run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("t-b"),
            AgentTurnId::new("turn-b"),
            ModelName::new("test-model"),
            "world",
        )),
    );
    assert_eq!(a.unwrap().stop_reason(), AgentTurnStopReason::FinalResponse);
    assert_eq!(b.unwrap().stop_reason(), AgentTurnStopReason::FinalResponse);
    assert_eq!(svc.list_threads(&ProjectId::new("proj-p")).len(), 2);
    let _ = std::fs::remove_dir_all(&base);
}
