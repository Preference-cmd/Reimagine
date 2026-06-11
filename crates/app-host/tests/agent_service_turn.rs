use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use reimagine_agent::{
    AgentEvent, AgentMode, AgentProvider, AgentRequest, AgentResponse, AgentSessionId, AgentStream,
    AgentStreamEvent, AgentTool, AgentToolRegistry, AgentTurnId, Message, ModelInfo, ModelName,
    PermissionSet, ProviderError, ProviderName, ToolCall, ToolCallId, ToolContext, ToolName,
    ToolPermission, ToolResult, ToolRiskLevel, ToolSpec, VecAgentEventSink, WorkspaceScope,
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

    async fn stream(&self, _request: AgentRequest) -> Result<Box<dyn AgentStream>, ProviderError> {
        Ok(Box::new(EmptyStream))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        Ok(Vec::new())
    }
}

struct EmptyStream;

#[async_trait]
impl AgentStream for EmptyStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        None
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
    let history = session.history();
    let roles: Vec<&str> = history.iter().map(|m| m.role()).collect();
    assert_eq!(roles, vec!["user", "assistant", "user", "assistant"]);

    let requests = provider.requests();
    let second_request_roles: Vec<&str> = requests[1].messages().iter().map(|m| m.role()).collect();
    assert_eq!(second_request_roles, vec!["user", "assistant", "user"]);
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
