//! Integration tests for the daemon server core (AS-08): handshake,
//! session lifecycle, turn streaming, cancellation, and the turn lock.

use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use reimagine_agent::{
    AgentProvider, AgentRequest, AgentResponse, AgentStream, AgentStreamEvent, Message, ModelInfo,
    ProviderError, ProviderName, ToolCall, ToolCallId,
};
use reimagine_agent_daemon::{
    protocol::{
        METHOD_AGENT_CONTENT_DELTA, METHOD_AGENT_TOOL_COMPLETED, METHOD_AGENT_TOOL_FAILED,
        METHOD_AGENT_TOOL_INVOKED, METHOD_AGENT_TURN_COMPLETED, METHOD_INITIALIZE,
        METHOD_PROVIDERS_LIST, METHOD_SESSION_CREATE, METHOD_SESSION_GET, METHOD_SESSION_LIST,
        METHOD_TURN_CANCEL, METHOD_TURN_RUN, METHOD_TURN_STEER,
    },
    server::{AgentDaemon, SharedWriter},
    transport::StdioTransport,
};
use serde_json::{Value, json};

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

fn request(method: &str, id: u64, params: Value) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string()
}

async fn drive(daemon: &mut AgentDaemon, requests: &[(&str, u64, Value)]) -> Arc<Mutex<Vec<u8>>> {
    let output = Arc::new(Mutex::new(Vec::new()));
    let mut input = requests
        .iter()
        .map(|(method, id, params)| request(method, *id, params.clone()))
        .collect::<Vec<String>>()
        .join("\n");
    input.push('\n');
    let transport = StdioTransport::new(Cursor::new(input), SharedWriter::new(Arc::clone(&output)));
    daemon
        .serve(transport)
        .await
        .expect("serve should return at end of input");
    output
}

fn find_response(output: &Mutex<Vec<u8>>, id: u64) -> Value {
    let text = String::from_utf8(output.lock().unwrap().clone()).expect("output is utf8");
    for line in text.lines() {
        let value: Value = serde_json::from_str(line).expect("output line is json");
        if value.get("id").and_then(Value::as_u64) == Some(id) {
            return value;
        }
    }
    panic!("no response with id {id} in output:\n{text}");
}

async fn wait_for_notification(output: &Mutex<Vec<u8>>, method: &str) -> Value {
    for _ in 0..500 {
        let text = String::from_utf8(output.lock().unwrap().clone()).expect("output is utf8");
        for line in text.lines() {
            let value: Value = serde_json::from_str(line).expect("output line is json");
            if value.get("method").and_then(Value::as_str) == Some(method) {
                return value;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("no notification for {method} within deadline");
}

async fn wait_for_any_notification(output: &Mutex<Vec<u8>>, methods: &[&str]) -> Value {
    for _ in 0..500 {
        let text = String::from_utf8(output.lock().unwrap().clone()).expect("output is utf8");
        for line in text.lines() {
            let value: Value = serde_json::from_str(line).expect("output line is json");
            if let Some(method) = value.get("method").and_then(Value::as_str)
                && methods.contains(&method)
            {
                return value;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("no notification in {methods:?} within deadline");
}

async fn create_session(daemon: &mut AgentDaemon) -> (String, Arc<Mutex<Vec<u8>>>) {
    let output = drive(
        daemon,
        &[
            (
                METHOD_INITIALIZE,
                1,
                json!({"client_info": {"name": "test-client", "version": "0.1.0"}, "capabilities": {}}),
            ),
            (METHOD_SESSION_CREATE, 2, json!({"mode": "agent", "provider": "mock"})),
        ],
    )
    .await;
    let created = find_response(&output, 2);
    let session_id = created["result"]["session_id"]
        .as_str()
        .expect("session id")
        .to_string();
    (session_id, output)
}

#[tokio::test]
async fn initialize_create_turn_streams_and_completes() {
    let dir = temp_dir("full-cycle");
    let mut daemon = AgentDaemon::new(&dir).await.expect("daemon initializes");
    daemon
        .workspace()
        .agent_service()
        .providers()
        .register(Arc::new(ScriptedProvider::new(
            "mock",
            vec![AgentResponse::new(Message::assistant("hello from mock"))],
        )));

    let output_a = drive(
        &mut daemon,
        &[
            (
                METHOD_INITIALIZE,
                1,
                json!({"client_info": {"name": "test-client", "version": "0.1.0"}, "capabilities": {}}),
            ),
            (METHOD_SESSION_CREATE, 2, json!({"mode": "agent", "provider": "mock"})),
        ],
    )
    .await;
    assert_eq!(
        daemon.client_info().expect("client info stored").name,
        "test-client"
    );
    let init = find_response(&output_a, 1);
    assert_eq!(
        init["result"]["server_info"]["name"],
        "reimagine-agent-daemon"
    );
    let created = find_response(&output_a, 2);
    assert_eq!(created["result"]["mode"], "agent");
    assert_eq!(created["result"]["provider"], "mock");
    let session_id = created["result"]["session_id"]
        .as_str()
        .expect("session id")
        .to_string();

    let output_b = drive(
        &mut daemon,
        &[
            (METHOD_PROVIDERS_LIST, 3, json!({})),
            (METHOD_SESSION_GET, 4, json!({"session_id": session_id})),
            (METHOD_SESSION_LIST, 5, json!({})),
            (
                METHOD_TURN_RUN,
                6,
                json!({"session_id": session_id, "turn_id": "turn-1", "model": "test-model", "input": {"text": "hello"}}),
            ),
            (
                METHOD_TURN_STEER,
                7,
                json!({"session_id": session_id, "turn_id": "turn-1", "input": {"text": "go faster"}}),
            ),
            ("no.such.method", 8, json!({})),
        ],
    )
    .await;

    let providers = find_response(&output_b, 3);
    assert_eq!(providers["result"]["providers"][0]["id"], "mock");
    let session = find_response(&output_b, 4);
    assert_eq!(session["result"]["session_id"], session_id);
    assert_eq!(session["result"]["mode"], "agent");
    let list = find_response(&output_b, 5);
    assert_eq!(list["result"]["sessions"][0]["session_id"], session_id);
    let accepted = find_response(&output_b, 6);
    assert_eq!(accepted["result"]["status"], "accepted");
    assert_eq!(accepted["result"]["turn_id"], "turn-1");
    let steer = find_response(&output_b, 7);
    assert_eq!(steer["error"]["code"].as_i64(), Some(-32604));
    let unknown = find_response(&output_b, 8);
    assert_eq!(unknown["error"]["code"].as_i64(), Some(-32601));

    let delta = wait_for_notification(&output_b, METHOD_AGENT_CONTENT_DELTA).await;
    assert_eq!(delta["params"]["session_id"], session_id);
    assert_eq!(delta["params"]["turn_id"], "turn-1");
    assert_eq!(delta["params"]["text"], "hello from mock");

    let completed = wait_for_notification(&output_b, METHOD_AGENT_TURN_COMPLETED).await;
    assert_eq!(completed["params"]["session_id"], session_id);
    assert_eq!(completed["params"]["turn_id"], "turn-1");
    assert_eq!(completed["params"]["result"]["status"], "completed");
    assert_eq!(
        completed["params"]["result"]["final_response"],
        "hello from mock"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn turn_with_tool_call_streams_tool_events() {
    let dir = temp_dir("tool-events");
    let mut daemon = AgentDaemon::new(&dir).await.expect("daemon initializes");
    daemon
        .workspace()
        .agent_service()
        .providers()
        .register(Arc::new(ScriptedProvider::new(
            "mock",
            vec![
                AgentResponse::new(Message::assistant_with_tool_calls(
                    "",
                    vec![ToolCall::new(
                        ToolCallId::new("call-1"),
                        "model.list",
                        json!({}),
                    )],
                )),
                AgentResponse::new(Message::assistant("tool roundtrip done")),
            ],
        )));

    let (session_id, _) = create_session(&mut daemon).await;

    let output = drive(
        &mut daemon,
        &[(
            METHOD_TURN_RUN,
            3,
            json!({"session_id": session_id, "turn_id": "turn-1", "model": "test-model", "input": {"text": "list models"}}),
        )],
    )
    .await;

    let invoked = wait_for_notification(&output, METHOD_AGENT_TOOL_INVOKED).await;
    assert_eq!(invoked["params"]["tool"], "model.list");
    assert_eq!(invoked["params"]["tool_call_id"], "call-1");
    let done = wait_for_any_notification(
        &output,
        &[METHOD_AGENT_TOOL_COMPLETED, METHOD_AGENT_TOOL_FAILED],
    )
    .await;
    assert_eq!(done["params"]["tool"], "model.list");
    assert_eq!(done["params"]["tool_call_id"], "call-1");

    let completed = wait_for_notification(&output, METHOD_AGENT_TURN_COMPLETED).await;
    assert_eq!(completed["params"]["result"]["status"], "completed");
    assert_eq!(
        completed["params"]["result"]["final_response"],
        "tool roundtrip done"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn turn_lock_rejects_concurrent_turn_on_same_session() {
    let dir = temp_dir("turn-lock");
    let mut daemon = AgentDaemon::new(&dir).await.expect("daemon initializes");
    daemon
        .workspace()
        .agent_service()
        .providers()
        .register(Arc::new(ScriptedProvider::new(
            "mock",
            vec![AgentResponse::new(Message::assistant("first turn done"))],
        )));

    let (session_id, _) = create_session(&mut daemon).await;

    let output = drive(
        &mut daemon,
        &[
            (
                METHOD_TURN_RUN,
                3,
                json!({"session_id": session_id, "turn_id": "turn-1", "model": "test-model", "input": {"text": "first"}}),
            ),
            (
                METHOD_TURN_RUN,
                4,
                json!({"session_id": session_id, "turn_id": "turn-2", "model": "test-model", "input": {"text": "second"}}),
            ),
        ],
    )
    .await;

    let first = find_response(&output, 3);
    assert_eq!(first["result"]["status"], "accepted");
    let second = find_response(&output, 4);
    assert_eq!(second["error"]["code"].as_i64(), Some(-32602));
    assert_eq!(
        second["error"]["data"]["reason"],
        "turn already in progress"
    );

    let completed = wait_for_notification(&output, METHOD_AGENT_TURN_COMPLETED).await;
    assert_eq!(completed["params"]["turn_id"], "turn-1");
    assert_eq!(completed["params"]["result"]["status"], "completed");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn persist_all_writes_every_session_context() {
    let dir = temp_dir("persist-all");
    let mut daemon = AgentDaemon::new(&dir).await.expect("daemon initializes");
    daemon
        .workspace()
        .agent_service()
        .providers()
        .register(Arc::new(ScriptedProvider::new("mock", vec![])));

    let (session_a, _) = create_session(&mut daemon).await;
    let (session_b, _) = create_session(&mut daemon).await;

    let persisted = daemon.persist_all().await;
    assert_eq!(persisted, 2, "both sessions persist");

    let session_dir = dir.join("agent-sessions");
    assert!(
        session_dir.join(format!("{session_a}.json")).exists(),
        "session a context file exists"
    );
    assert!(
        session_dir.join(format!("{session_b}.json")).exists(),
        "session b context file exists"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn turn_cancel_stops_accepted_turn() {
    let dir = temp_dir("turn-cancel");
    let mut daemon = AgentDaemon::new(&dir).await.expect("daemon initializes");
    daemon
        .workspace()
        .agent_service()
        .providers()
        .register(Arc::new(ScriptedProvider::new(
            "mock",
            vec![AgentResponse::new(Message::assistant("never meant to run"))],
        )));

    let (session_id, _) = create_session(&mut daemon).await;

    let output = drive(
        &mut daemon,
        &[
            (
                METHOD_TURN_RUN,
                3,
                json!({"session_id": session_id, "turn_id": "turn-1", "model": "test-model", "input": {"text": "hello"}}),
            ),
            (
                METHOD_TURN_CANCEL,
                4,
                json!({"session_id": session_id, "turn_id": "turn-1"}),
            ),
        ],
    )
    .await;

    let accepted = find_response(&output, 3);
    assert_eq!(accepted["result"]["status"], "accepted");
    let cancelled = find_response(&output, 4);
    assert_eq!(cancelled["result"]["status"], "cancelled");

    let completed = wait_for_notification(&output, METHOD_AGENT_TURN_COMPLETED).await;
    assert_eq!(completed["params"]["turn_id"], "turn-1");
    assert_eq!(completed["params"]["result"]["status"], "stopped");
    assert_eq!(completed["params"]["result"]["stop_reason"], "cancelled");

    let _ = std::fs::remove_dir_all(&dir);
}

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-agent-daemon-server-{prefix}-{nonce}"))
}
