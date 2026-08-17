use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_agent_harness::{
    AgentEvent, AgentMode, AgentProvider, AgentRequest, AgentResponse, AgentSessionId, AgentStream,
    AgentStreamEvent, AgentTool, AgentToolRegistry, AgentTurnId, AgentTurnStopReason, Message,
    ModelInfo, ModelName, PermissionSet, ProviderError, ProviderName, ToolCall, ToolCallId,
    ToolContext, ToolName, ToolPermission, ToolResult, ToolRiskLevel, ToolSpec, VecAgentEventSink,
    WorkspaceScope,
};
use reimagine_app_host::{
    AgentProviderCatalog, AgentService, AgentServiceTurnRequest, AppHostError, WorkspaceHost,
};
use reimagine_core::command::{
    CommandActor, CommandActorKind, CommandBatch, CommandProvenance, WorkflowCommand,
};
use reimagine_core::event::Timestamp;
use reimagine_core::model::{
    CommandBatchId, NodeId, NodeTypeId, ProposalId, SlotId, WorkflowVersion,
};
use reimagine_core::workflow::Workflow;
use reimagine_nodes::BUILTIN_STRING;
use serde_json::json;
use tokio::sync::{Barrier, Notify};

struct ScriptedProvider {
    name: ProviderName,
    responses: Mutex<VecDeque<Result<AgentResponse, ProviderError>>>,
    requests: Mutex<Vec<AgentRequest>>,
}

impl ScriptedProvider {
    fn new(name: &str, responses: Vec<AgentResponse>) -> Self {
        Self {
            name: ProviderName::new(name),
            responses: Mutex::new(responses.into_iter().map(Ok).collect()),
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
            .unwrap_or_else(|| {
                Err(ProviderError::new(
                    "SCRIPT_EXHAUSTED",
                    "scripted provider exhausted",
                ))
            })
    }

    async fn stream(&self, request: AgentRequest) -> Result<Box<dyn AgentStream>, ProviderError> {
        self.requests.lock().unwrap().push(request);
        let response = self.responses.lock().unwrap().pop_front().ok_or_else(|| {
            ProviderError::new("SCRIPT_EXHAUSTED", "scripted provider exhausted")
        })??;
        Ok(Box::new(ScriptedStream::from_response(response)))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(Vec::new())
    }
}

struct ScriptedStream {
    events: VecDeque<AgentStreamEvent>,
}

impl ScriptedStream {
    fn from_response(response: AgentResponse) -> Self {
        let mut events = VecDeque::new();
        let message = response.message();
        let content = message.content();
        if !content.is_empty() {
            events.push_back(AgentStreamEvent::ContentDelta(content.to_string()));
        }
        for tool_call in message.tool_calls() {
            events.push_back(AgentStreamEvent::ToolCall(tool_call.clone()));
        }
        events.push_back(AgentStreamEvent::Done { stop_reason: None });
        Self { events }
    }
}

#[async_trait]
impl AgentStream for ScriptedStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        self.events.pop_front()
    }
}

/// Stream that blocks until `release` fires, then yields one content
/// delta and a clean `Done`. Lets tests hold a turn open deterministically.
struct WaitForNotifyStream {
    release: Arc<Notify>,
    queued: Mutex<VecDeque<AgentStreamEvent>>,
}

impl WaitForNotifyStream {
    fn new(release: Arc<Notify>) -> Self {
        Self {
            release,
            queued: Mutex::new(VecDeque::new()),
        }
    }
}

#[async_trait]
impl AgentStream for WaitForNotifyStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        let empty = self.queued.lock().unwrap().is_empty();
        if empty {
            let _ = tokio::time::timeout(Duration::from_secs(10), self.release.notified()).await;
            *self.queued.lock().unwrap() = VecDeque::from([
                AgentStreamEvent::ContentDelta("released".into()),
                AgentStreamEvent::Done {
                    stop_reason: Some("stop".into()),
                },
            ]);
        }
        self.queued.lock().unwrap().pop_front()
    }
}

/// Stream that immediately reports a normal final answer.
struct DoneStream {
    queued: Mutex<VecDeque<AgentStreamEvent>>,
}

impl DoneStream {
    fn new() -> Self {
        Self {
            queued: Mutex::new(VecDeque::from([
                AgentStreamEvent::ContentDelta("done".into()),
                AgentStreamEvent::Done {
                    stop_reason: Some("stop".into()),
                },
            ])),
        }
    }
}

#[async_trait]
impl AgentStream for DoneStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        self.queued.lock().unwrap().pop_front()
    }
}

/// Stream whose `next_event` never resolves; cancellation must drop it.
struct NeverStream;

#[async_trait]
impl AgentStream for NeverStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        std::future::pending::<()>().await;
        unreachable!("never stream is dropped by cancellation")
    }
}

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            ToolName::new("echo"),
            "echo",
            [AgentMode::Agent],
            ToolPermission::new("workflow.read"),
            ToolRiskLevel::Read,
        )
    }

    async fn invoke(&self, _ctx: &ToolContext, _input: serde_json::Value) -> ToolResult {
        Ok(json!({"ok": true, "effective": false}))
    }
}

#[tokio::test]
async fn run_turn_returns_final_response_without_tool_calls() {
    let host = build_host("ws-agent-final");
    let provider = Arc::new(ScriptedProvider::new(
        "mock",
        vec![AgentResponse::new(Message::assistant("done"))],
    ));
    host.agent_service().providers().register(provider);
    host.agent_service().create_session_with_permissions(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
        workflow_permissions(),
    );

    let result = host
        .agent_service()
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            "hello",
        ))
        .await
        .expect("turn should run");

    assert!(result.is_completed());
    assert_eq!(result.final_response().unwrap().content(), "done");
    assert!(result.tool_calls().is_empty());
}

#[tokio::test]
async fn run_turn_rejects_missing_session() {
    let host = build_host("ws-agent-missing-session");
    let err = host
        .agent_service()
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("missing"),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            "hello",
        ))
        .await
        .expect_err("missing session should fail");

    assert!(matches!(err, AppHostError::UnknownAgentSession { .. }));
}

#[tokio::test]
async fn run_turn_rejects_missing_provider_before_loop_starts() {
    let host = build_host("ws-agent-missing-provider");
    host.agent_service().create_session_with_permissions(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("ghost"),
        "2026-06-12T00:00:00Z",
        workflow_permissions(),
    );

    let err = host
        .agent_service()
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            "hello",
        ))
        .await
        .expect_err("missing provider should fail");

    assert!(matches!(err, AppHostError::UnknownAgentProvider { .. }));
}

#[tokio::test]
async fn run_turn_preserves_session_history_between_turns() {
    let host = build_host("ws-agent-history");
    let provider = Arc::new(ScriptedProvider::new(
        "mock",
        vec![
            AgentResponse::new(Message::assistant("first")),
            AgentResponse::new(Message::assistant("second")),
        ],
    ));
    host.agent_service().providers().register(provider.clone());
    host.agent_service().create_session_with_permissions(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
        workflow_permissions(),
    );

    let service = host.agent_service();
    service
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            "hello",
        ))
        .await
        .unwrap();
    service
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-2"),
            ModelName::new("test-model"),
            "again",
        ))
        .await
        .unwrap();

    let session = service.get_session(&AgentSessionId::new("sess-1")).unwrap();
    // Embedded turns run through the per-session ContextManager, which
    // owns the rolling history; the session's own vector stays empty
    // (the old V1 seed path only applies when no context is supplied).
    assert!(
        session.history().is_empty(),
        "ContextManager owns history, not AgentSession"
    );

    let requests = provider.requests();
    let second_request_roles: Vec<&str> = requests[1].messages().iter().map(|m| m.role()).collect();
    // ContextManager seeds its own system slot ahead of the windowed
    // history: system, turn-1 user + assistant, turn-2 user.
    assert_eq!(
        second_request_roles,
        vec!["system", "user", "assistant", "user"]
    );
}

#[tokio::test]
async fn run_turn_routes_tool_call_through_app_host_registry() {
    let host = build_host("ws-agent-tool");
    let workflow_id = host
        .workflow_service()
        .register_workflow(Workflow::new("wf-tool", WorkflowVersion::new(0)));
    let provider = Arc::new(ScriptedProvider::new(
        "mock",
        vec![
            AgentResponse::new(Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    ToolCallId::new("call-1"),
                    "workflow.get",
                    json!({ "workflow_id": workflow_id.as_str() }),
                )],
            )),
            AgentResponse::new(Message::assistant("observed")),
        ],
    ));
    host.agent_service().providers().register(provider);
    host.agent_service().create_session_with_permissions(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
        workflow_permissions(),
    );

    let result = host
        .agent_service()
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            "inspect workflow",
        ))
        .await
        .unwrap();

    assert!(result.is_completed());
    assert_eq!(result.tool_calls().len(), 1);
    assert_eq!(result.tool_calls()[0].tool_name().as_str(), "workflow.get");
    assert_eq!(result.tool_calls()[0].effective(), Some(false));
}

#[tokio::test]
async fn run_turn_emits_agent_events_through_injected_sink() {
    let mut registry = AgentToolRegistry::new();
    registry.register(EchoTool).unwrap();
    let registry = Arc::new(registry);
    let provider = Arc::new(ScriptedProvider::new(
        "mock",
        vec![
            AgentResponse::new(Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(ToolCallId::new("call-1"), "echo", json!({}))],
            )),
            AgentResponse::new(Message::assistant("done")),
        ],
    ));
    let providers = AgentProviderCatalog::with_provider(provider);
    let sink = Arc::new(VecAgentEventSink::new());
    let service = AgentService::with_registry_providers_and_sink(
        WorkspaceScope::new("ws-events"),
        registry,
        providers,
        sink.clone(),
    );
    service.create_session_with_permissions(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
        PermissionSet::from_iter([ToolPermission::new("workflow.read")]),
    );

    service
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            "echo please",
        ))
        .await
        .unwrap();

    let events = sink.events();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolInvoked { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCompleted { .. }))
    );
}

#[tokio::test]
async fn build_mode_proposal_turn_returns_effective_false_without_mutation() {
    let host = build_host("ws-agent-proposal");
    let workflow_id = host
        .workflow_service()
        .register_workflow(Workflow::new("wf-proposal", WorkflowVersion::new(0)));
    let batch = add_node_batch(WorkflowVersion::new(0), "node-proposal");
    let provider = Arc::new(ScriptedProvider::new(
        "mock",
        vec![
            AgentResponse::new(Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    ToolCallId::new("call-1"),
                    "workflow.propose_commands",
                    json!({
                        "workflow_id": workflow_id.as_str(),
                        "proposal_id": ProposalId::new("proposal-1"),
                        "batch": batch,
                        "created_at": "2026-06-12T00:00:00Z",
                    }),
                )],
            )),
            AgentResponse::new(Message::assistant("proposal ready")),
        ],
    ));
    host.agent_service().providers().register(provider);
    host.agent_service().create_session_with_permissions(
        AgentSessionId::new("sess-1"),
        AgentMode::Build,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
        workflow_permissions(),
    );

    let result = host
        .agent_service()
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            "prepare proposal",
        ))
        .await
        .unwrap();

    assert!(result.is_completed());
    assert_eq!(result.tool_calls().len(), 1);
    assert_eq!(result.tool_calls()[0].effective(), Some(false));
    assert_eq!(
        host.workflow_service()
            .snapshot(&workflow_id)
            .unwrap()
            .version(),
        WorkflowVersion::new(0)
    );
    assert!(
        host.workflow_service()
            .get_pending_proposal(&workflow_id)
            .is_some()
    );
}

#[test]
fn provider_catalog_registers_and_resolves_providers() {
    let catalog = AgentProviderCatalog::new();
    let provider = Arc::new(ScriptedProvider::new(
        "mock",
        vec![AgentResponse::new(Message::assistant("unused"))],
    ));
    catalog.register(provider);
    assert!(catalog.contains(&ProviderName::new("mock")));
    assert_eq!(catalog.provider_names()[0].as_str(), "mock");
}

#[test]
fn workspace_host_agent_service_starts_with_empty_provider_catalog() {
    let host = build_host("ws-empty-providers");
    assert!(host.agent_service().providers().is_empty());
}

#[test]
fn explicit_session_permissions_are_attached() {
    let host = build_host("ws-permissions");
    let session = host.agent_service().create_session_with_permissions(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
        workflow_permissions(),
    );
    assert!(
        session
            .permissions()
            .contains(&ToolPermission::new("workflow.write"))
    );
}

#[tokio::test]
async fn concurrent_turn_on_same_session_is_rejected() {
    struct HangingProvider {
        name: ProviderName,
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    #[async_trait]
    impl AgentProvider for HangingProvider {
        fn name(&self) -> ProviderName {
            self.name.clone()
        }

        async fn complete(&self, _request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            Err(ProviderError::new("stream_only", "streaming provider"))
        }

        async fn stream(
            &self,
            _request: AgentRequest,
        ) -> Result<Box<dyn AgentStream>, ProviderError> {
            self.started.notify_one();
            Ok(Box::new(WaitForNotifyStream::new(Arc::clone(
                &self.release,
            ))))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
    }

    let host = build_host("ws-agent-same-session-lock");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    host.agent_service()
        .providers()
        .register(Arc::new(HangingProvider {
            name: ProviderName::new("mock"),
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }));
    host.agent_service().create_session(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
    );

    let service = host.agent_service().clone();
    let first = tokio::spawn({
        let service = Arc::clone(&service);
        async move {
            service
                .run_turn(AgentServiceTurnRequest::from_user_text(
                    AgentSessionId::new("sess-1"),
                    AgentTurnId::new("turn-1"),
                    ModelName::new("test-model"),
                    "first",
                ))
                .await
        }
    });

    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("first turn reaches the provider");

    let second = service
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-2"),
            ModelName::new("test-model"),
            "second",
        ))
        .await;
    assert!(
        matches!(second, Err(AppHostError::AgentTurnInProgress { .. })),
        "expected AgentTurnInProgress, got: {second:?}"
    );

    release.notify_one();
    let first_result = tokio::time::timeout(Duration::from_secs(10), first)
        .await
        .expect("first turn finishes")
        .expect("first turn task joins")
        .expect("first turn succeeds");
    assert!(first_result.is_completed());
}

#[tokio::test]
async fn different_sessions_run_turns_in_parallel() {
    struct BarrierProvider {
        name: ProviderName,
        barrier: Arc<Barrier>,
    }

    #[async_trait]
    impl AgentProvider for BarrierProvider {
        fn name(&self) -> ProviderName {
            self.name.clone()
        }

        async fn complete(&self, _request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            Err(ProviderError::new("stream_only", "streaming provider"))
        }

        async fn stream(
            &self,
            _request: AgentRequest,
        ) -> Result<Box<dyn AgentStream>, ProviderError> {
            match tokio::time::timeout(Duration::from_secs(5), self.barrier.wait()).await {
                Ok(_) => Ok(Box::new(DoneStream::new())),
                Err(_) => Err(ProviderError::new(
                    "BARRIER_TIMEOUT",
                    "second session never started its turn",
                )),
            }
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
    }

    let host = build_host("ws-agent-parallel-sessions");
    let barrier = Arc::new(Barrier::new(2));
    host.agent_service()
        .providers()
        .register(Arc::new(BarrierProvider {
            name: ProviderName::new("mock"),
            barrier: Arc::clone(&barrier),
        }));
    host.agent_service().create_session(
        AgentSessionId::new("sess-a"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
    );
    host.agent_service().create_session(
        AgentSessionId::new("sess-b"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
    );

    let service = host.agent_service().clone();
    let turn_a = {
        let service = Arc::clone(&service);
        tokio::spawn(async move {
            service
                .run_turn(AgentServiceTurnRequest::from_user_text(
                    AgentSessionId::new("sess-a"),
                    AgentTurnId::new("turn-a"),
                    ModelName::new("test-model"),
                    "a",
                ))
                .await
        })
    };
    let turn_b = {
        let service = Arc::clone(&service);
        tokio::spawn(async move {
            service
                .run_turn(AgentServiceTurnRequest::from_user_text(
                    AgentSessionId::new("sess-b"),
                    AgentTurnId::new("turn-b"),
                    ModelName::new("test-model"),
                    "b",
                ))
                .await
        })
    };

    let (a, b) = tokio::time::timeout(Duration::from_secs(15), async {
        let a = turn_a
            .await
            .expect("turn-a joins")
            .expect("turn-a succeeds");
        let b = turn_b
            .await
            .expect("turn-b joins")
            .expect("turn-b succeeds");
        (a, b)
    })
    .await
    .expect("both parallel turns finish");

    assert!(a.is_completed());
    assert!(b.is_completed());
}

#[tokio::test]
async fn cancel_turn_aborts_in_flight_turn() {
    struct NeverProvider {
        name: ProviderName,
        started: Arc<Notify>,
    }

    #[async_trait]
    impl AgentProvider for NeverProvider {
        fn name(&self) -> ProviderName {
            self.name.clone()
        }

        async fn complete(&self, _request: AgentRequest) -> Result<AgentResponse, ProviderError> {
            Err(ProviderError::new("stream_only", "streaming provider"))
        }

        async fn stream(
            &self,
            _request: AgentRequest,
        ) -> Result<Box<dyn AgentStream>, ProviderError> {
            self.started.notify_one();
            Ok(Box::new(NeverStream))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
    }

    let host = build_host("ws-agent-cancel");
    let started = Arc::new(Notify::new());
    host.agent_service()
        .providers()
        .register(Arc::new(NeverProvider {
            name: ProviderName::new("mock"),
            started: Arc::clone(&started),
        }));
    host.agent_service().create_session(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
    );

    let service = host.agent_service().clone();
    let turn = tokio::spawn({
        let service = Arc::clone(&service);
        async move {
            service
                .run_turn(AgentServiceTurnRequest::from_user_text(
                    AgentSessionId::new("sess-1"),
                    AgentTurnId::new("turn-1"),
                    ModelName::new("test-model"),
                    "never finishes",
                ))
                .await
        }
    });

    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("turn reaches the provider");
    service
        .cancel_turn(&AgentSessionId::new("sess-1"))
        .expect("active turn cancels");

    let result = tokio::time::timeout(Duration::from_secs(10), turn)
        .await
        .expect("cancelled turn finishes")
        .expect("turn task joins")
        .expect("cancelled turn returns a result");
    assert_eq!(result.stop_reason(), AgentTurnStopReason::Cancelled);
    assert!(!result.is_completed());

    assert!(matches!(
        service.cancel_turn(&AgentSessionId::new("sess-1")),
        Err(AppHostError::NoActiveAgentTurn { .. })
    ));
    assert!(matches!(
        service.cancel_turn(&AgentSessionId::new("missing")),
        Err(AppHostError::UnknownAgentSession { .. })
    ));
}

#[tokio::test]
async fn turn_initializes_and_persists_per_session_context() {
    let host = build_host("ws-agent-context-persist");
    host.agent_service()
        .providers()
        .register(Arc::new(ScriptedProvider::new(
            "mock",
            vec![AgentResponse::new(Message::assistant("persisted context"))],
        )));
    host.agent_service().create_session(
        AgentSessionId::new("sess-1"),
        AgentMode::Agent,
        ProviderName::new("mock"),
        "2026-06-12T00:00:00Z",
    );

    host.agent_service()
        .run_turn(AgentServiceTurnRequest::from_user_text(
            AgentSessionId::new("sess-1"),
            AgentTurnId::new("turn-1"),
            ModelName::new("test-model"),
            "hello",
        ))
        .await
        .expect("turn succeeds");

    let context_file = host.base_path().join("agent-sessions").join("sess-1.json");
    let contents = std::fs::read_to_string(&context_file).expect("per-session context file exists");
    let parsed: serde_json::Value =
        serde_json::from_str(&contents).expect("persisted context is JSON");
    assert_eq!(parsed["session_id"], "sess-1");
    assert!(parsed["history"].is_array());
}

fn build_host(scope: &str) -> WorkspaceHost {
    WorkspaceHost::with_defaults(WorkspaceScope::new(scope), temp_dir(scope))
}

fn workflow_permissions() -> PermissionSet {
    PermissionSet::from_iter([
        ToolPermission::new("workflow.read"),
        ToolPermission::new("workflow.write"),
        ToolPermission::new("model.read"),
    ])
}

fn add_node_batch(base_version: WorkflowVersion, label: &str) -> CommandBatch {
    CommandBatch::new(
        CommandBatchId::new(format!("batch-{label}")),
        CommandActor::new(CommandActorKind::Agent).with_id("sess-1"),
        base_version,
        CommandProvenance::Direct,
        Timestamp::new("2026-06-12T00:00:00Z"),
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

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-app-host-agent-turn-{prefix}-{nonce}"))
}
