//! Daemon server core: JSON-RPC dispatch, session management, and turn
//! execution.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use reimagine_agent::{
    AgentEvent, AgentEventSink, AgentLoop, AgentMode, AgentSessionId, AgentTurnId,
    AgentTurnRequest, AgentTurnResult, ContextConfig, ContextManager, Message, ModelName,
    PermissionSet, ProviderName, ToolCallId, ToolPermission, VecAgentEventSink,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::init::{DaemonInitError, DaemonWorkspace};
use crate::protocol::{
    AgentErrorParams, ClientInfo, ContentDeltaParams, InitializeRequest, InitializeResponse,
    JsonRpcError, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, METHOD_AGENT_CONTENT_DELTA,
    METHOD_AGENT_ERROR, METHOD_AGENT_PROPOSAL_READY, METHOD_AGENT_SESSION_STARTED,
    METHOD_AGENT_SESSION_STOPPED, METHOD_AGENT_TOOL_COMPLETED, METHOD_AGENT_TOOL_FAILED,
    METHOD_AGENT_TOOL_INVOKED, METHOD_AGENT_TURN_COMPLETED, METHOD_INITIALIZE, METHOD_INITIALIZED,
    METHOD_PROVIDERS_LIST, METHOD_SESSION_CREATE, METHOD_SESSION_GET, METHOD_SESSION_LIST,
    METHOD_TURN_CANCEL, METHOD_TURN_RUN, METHOD_TURN_STEER, ProposalReadyParams, ProviderInfo,
    ProvidersListResult, ServerCapabilities, ServerInfo, SessionCreateParams, SessionCreateResult,
    SessionGetParams, SessionInfo, SessionListResult, SessionStartedParams, SessionStoppedParams,
    ToolEventParams, TurnCancelParams, TurnCancelResult, TurnCancelStatus, TurnCompletedParams,
    TurnRunParams, TurnRunResult, TurnRunStatus,
};
use crate::transport::StdioTransport;

type DispatchResult = Result<Option<Value>, JsonRpcError>;

/// Writer shared between the serve loop and spawned turn tasks.
///
/// The serve loop reads requests through its exclusive `StdioTransport`
/// reader; turn tasks stream notifications through the same writer
/// behind an `Arc`, so `serve` takes the transport by value and rebuilds
/// it around this newtype.
pub struct SharedWriter<W: Write> {
    inner: Arc<Mutex<W>>,
}

impl<W: Write> SharedWriter<W> {
    pub fn new(inner: Arc<Mutex<W>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> Arc<Mutex<W>> {
        self.inner
    }
}

impl<W: Write> Write for SharedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .flush()
    }
}

/// The daemon's per-session state.
struct SessionState {
    info: SessionInfo,
    context: Arc<AsyncMutex<ContextManager>>,
    turn_lock: Arc<Semaphore>,
}

/// JSON-RPC server core: owns the daemon workspace, live sessions, and
/// the map of in-flight turn cancellation tokens.
pub struct AgentDaemon {
    workspace: DaemonWorkspace,
    sessions: HashMap<AgentSessionId, SessionState>,
    active_turns: Arc<Mutex<HashMap<AgentSessionId, CancellationToken>>>,
    client: Option<ClientInfo>,
}

impl AgentDaemon {
    /// Initialize the daemon workspace rooted at `workspace_dir`.
    ///
    /// Providers must be registered on the workspace's agent service
    /// before `session.create` / `turn.run` succeed.
    pub async fn new(workspace_dir: &Path) -> Result<Self, DaemonInitError> {
        let sink: Arc<dyn AgentEventSink> = Arc::new(VecAgentEventSink::new());
        let workspace = DaemonWorkspace::initialize(workspace_dir, sink).await?;
        Ok(Self {
            workspace,
            sessions: HashMap::new(),
            active_turns: Arc::new(Mutex::new(HashMap::new())),
            client: None,
        })
    }

    pub fn workspace(&self) -> &DaemonWorkspace {
        &self.workspace
    }

    /// Client identity reported by the last `initialize` handshake.
    pub fn client_info(&self) -> Option<&ClientInfo> {
        self.client.as_ref()
    }

    /// Serve requests from `transport` until the reader is exhausted.
    ///
    /// Responses are written synchronously; notifications stream from
    /// spawned turn tasks through the shared writer.
    pub async fn serve<R, W>(
        &mut self,
        transport: StdioTransport<R, SharedWriter<W>>,
    ) -> io::Result<()>
    where
        R: BufRead,
        W: Write + Send + 'static,
    {
        let (reader, shared_writer) = transport.into_parts();
        let writer = shared_writer.into_inner();
        let mut transport = StdioTransport::new(reader, SharedWriter::new(Arc::clone(&writer)));
        while let Some(request) = transport.read_request() {
            self.dispatch(&transport, &writer, request)?;
        }
        Ok(())
    }

    /// Serve on the process stdin/stdout.
    pub async fn serve_stdio(&mut self) -> io::Result<()> {
        let transport = StdioTransport::new(
            io::BufReader::new(io::stdin()),
            SharedWriter::new(Arc::new(Mutex::new(io::stdout()))),
        );
        self.serve(transport).await
    }

    fn dispatch<R, W>(
        &mut self,
        transport: &StdioTransport<R, SharedWriter<W>>,
        writer: &Arc<Mutex<W>>,
        request: JsonRpcRequest<Value>,
    ) -> io::Result<()>
    where
        R: BufRead,
        W: Write + Send + 'static,
    {
        let id = request.id;
        let result = match request.method.as_str() {
            METHOD_INITIALIZED => return Ok(()),
            METHOD_INITIALIZE => self.handle_initialize(request.params),
            METHOD_SESSION_CREATE => self.handle_session_create(request.params),
            METHOD_SESSION_GET => self.handle_session_get(request.params),
            METHOD_SESSION_LIST => self.handle_session_list(),
            METHOD_TURN_RUN => self.handle_turn_run(writer, request.params),
            METHOD_TURN_CANCEL => self.handle_turn_cancel(request.params),
            METHOD_TURN_STEER => Err(JsonRpcError::not_supported()),
            METHOD_PROVIDERS_LIST => self.handle_providers_list(),
            _ => Err(JsonRpcError::method_not_found()),
        };
        match result {
            Ok(Some(result)) => transport.write_response(&JsonRpcResponse::success(id, result)),
            Ok(None) => Ok(()),
            Err(error) => transport.write_response(&JsonRpcResponse::<Value>::error(id, error)),
        }
    }

    fn handle_initialize(&mut self, params: Value) -> DispatchResult {
        let request: InitializeRequest = from_value(params)?;
        self.client = Some(request.client_info);
        Ok(Some(to_value(InitializeResponse {
            server_info: ServerInfo {
                name: "reimagine-agent-daemon".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
            },
            capabilities: ServerCapabilities {},
        })))
    }

    fn handle_session_create(&mut self, params: Value) -> DispatchResult {
        let request: SessionCreateParams = from_value(params)?;
        let mode = match request.mode.as_str() {
            "agent" => AgentMode::Agent,
            "build" => AgentMode::Build,
            _ => return Err(JsonRpcError::invalid_params()),
        };
        let provider = ProviderName::new(request.provider);
        if !self.workspace.providers().contains(&provider) {
            return Err(JsonRpcError::invalid_params().with_data(json!({
                "provider": provider.to_string(),
            })));
        }
        let session_id = AgentSessionId::new(format!("sess-{}", timestamp_nanos()));
        let started_at = timestamp_millis();
        self.workspace
            .agent_service()
            .create_session_with_permissions(
                session_id.clone(),
                mode,
                provider.clone(),
                started_at.clone(),
                v1_permissions(),
            );
        let info = SessionInfo {
            session_id: session_id.to_string(),
            mode: mode.to_string(),
            provider: provider.to_string(),
            created_at: started_at,
        };
        let config = ContextConfig {
            max_tokens: 64_000,
            recent_turns: 20,
            session_dir: self.workspace.host().base_path().join("agent-sessions"),
        };
        let state = SessionState {
            info: info.clone(),
            context: Arc::new(AsyncMutex::new(ContextManager::new(config))),
            turn_lock: Arc::new(Semaphore::new(1)),
        };
        self.sessions.insert(session_id.clone(), state);
        Ok(Some(to_value(SessionCreateResult {
            session_id: info.session_id,
            mode: info.mode,
            provider: info.provider,
            created_at: info.created_at,
        })))
    }

    fn handle_session_get(&self, params: Value) -> DispatchResult {
        let request: SessionGetParams = from_value(params)?;
        let session_id = AgentSessionId::new(request.session_id);
        match self.sessions.get(&session_id) {
            Some(state) => Ok(Some(to_value(state.info.clone()))),
            None => Err(unknown_session(&session_id)),
        }
    }

    fn handle_session_list(&self) -> DispatchResult {
        let mut sessions: Vec<SessionInfo> = self
            .sessions
            .values()
            .map(|state| state.info.clone())
            .collect();
        sessions.sort_by(|a, b| a.session_id.cmp(&b.session_id));
        Ok(Some(to_value(SessionListResult { sessions })))
    }

    fn handle_turn_run<W: Write + Send + 'static>(
        &mut self,
        writer: &Arc<Mutex<W>>,
        params: Value,
    ) -> DispatchResult {
        let request: TurnRunParams = from_value(params)?;
        let session_id = AgentSessionId::new(request.session_id);
        let turn_id = AgentTurnId::new(request.turn_id);
        let model = ModelName::new(request.model);
        let Some(text) = turn_input_text(&request.input) else {
            return Err(JsonRpcError::invalid_params());
        };
        let agent_session = self
            .workspace
            .agent_service()
            .get_session(&session_id)
            .map_err(|_| unknown_session(&session_id))?;
        let provider = self
            .workspace
            .providers()
            .get(agent_session.provider())
            .ok_or_else(|| {
                JsonRpcError::invalid_params().with_data(json!({
                    "provider": agent_session.provider().to_string(),
                }))
            })?;
        let Some(state) = self.sessions.get(&session_id) else {
            return Err(unknown_session(&session_id));
        };
        let permit = state.turn_lock.clone().try_acquire_owned().map_err(|_| {
            JsonRpcError::invalid_params().with_data(json!({
                "session_id": session_id.to_string(),
                "reason": "turn already in progress",
            }))
        })?;
        let cancel_token = CancellationToken::new();
        lock(&self.active_turns).insert(session_id.clone(), cancel_token.clone());
        let context = Arc::clone(&state.context);
        let sink = Arc::new(TurnEventSink {
            writer: Arc::clone(writer),
            session_id: session_id.clone(),
            turn_id: turn_id.clone(),
        });
        let loop_harness = AgentLoop::new(provider, sink);
        let writer = Arc::clone(writer);
        let active_turns = Arc::clone(&self.active_turns);
        let task_session_id = session_id.clone();
        let task_turn_id = turn_id.clone();
        tokio::spawn(async move {
            let turn_request = AgentTurnRequest::new(
                agent_session,
                task_turn_id.clone(),
                model,
                vec![Message::user(text)],
            )
            .with_cancel_token(cancel_token);
            let mut context_guard = context.lock().await;
            let result = loop_harness
                .run_turn_streaming(turn_request, Some(&mut context_guard))
                .await;
            let _ = context_guard.persist(task_session_id.as_str());
            drop(context_guard);
            let notification = JsonRpcNotification::new(
                METHOD_AGENT_TURN_COMPLETED,
                TurnCompletedParams {
                    session_id: task_session_id.to_string(),
                    turn_id: task_turn_id.to_string(),
                    result: turn_result_value(&result),
                },
            );
            let _ = write_line(&writer, &notification);
            lock(&active_turns).remove(&task_session_id);
            drop(permit);
        });
        Ok(Some(to_value(TurnRunResult {
            status: TurnRunStatus::Accepted,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
        })))
    }

    fn handle_turn_cancel(&self, params: Value) -> DispatchResult {
        let request: TurnCancelParams = from_value(params)?;
        let session_id = AgentSessionId::new(request.session_id);
        if !self.sessions.contains_key(&session_id) {
            return Err(unknown_session(&session_id));
        }
        if let Some(cancel_token) = lock(&self.active_turns).get(&session_id) {
            cancel_token.cancel();
        }
        Ok(Some(to_value(TurnCancelResult {
            status: TurnCancelStatus::Cancelled,
        })))
    }

    fn handle_providers_list(&self) -> DispatchResult {
        let providers = self
            .workspace
            .providers()
            .provider_names()
            .into_iter()
            .map(|name| ProviderInfo {
                id: name.to_string(),
                name: name.to_string(),
            })
            .collect();
        Ok(Some(to_value(ProvidersListResult { providers })))
    }
}

/// Event sink that forwards one turn's `AgentEvent`s to the transport
/// as daemon-to-client notifications. Closes over the turn id, which
/// `AgentEvent` does not carry.
struct TurnEventSink<W: Write + Send> {
    writer: Arc<Mutex<W>>,
    session_id: AgentSessionId,
    turn_id: AgentTurnId,
}

impl<W: Write + Send> AgentEventSink for TurnEventSink<W> {
    fn handle(&self, event: &AgentEvent) {
        if let Some((method, params)) = self.notification_for(event) {
            let _ = write_line(&self.writer, &JsonRpcNotification::new(method, params));
        }
    }
}

impl<W: Write + Send> TurnEventSink<W> {
    fn notification_for(&self, event: &AgentEvent) -> Option<(&'static str, Value)> {
        match event {
            AgentEvent::ContentDelta { text, .. } => Some((
                METHOD_AGENT_CONTENT_DELTA,
                to_value(ContentDeltaParams {
                    session_id: self.session_id.to_string(),
                    turn_id: self.turn_id.to_string(),
                    text: text.clone(),
                }),
            )),
            AgentEvent::ToolInvoked { tool, id, .. } => {
                Some((METHOD_AGENT_TOOL_INVOKED, self.tool_params(tool, id, None)))
            }
            AgentEvent::ToolCompleted { tool, id, .. } => Some((
                METHOD_AGENT_TOOL_COMPLETED,
                self.tool_params(tool, id, None),
            )),
            AgentEvent::ToolFailed {
                tool, id, message, ..
            } => Some((
                METHOD_AGENT_TOOL_FAILED,
                self.tool_params(tool, id, Some(message.clone())),
            )),
            AgentEvent::ProviderError { code, message, .. } => Some((
                METHOD_AGENT_ERROR,
                to_value(AgentErrorParams {
                    session_id: self.session_id.to_string(),
                    code: code.clone(),
                    message: message.clone(),
                }),
            )),
            AgentEvent::SessionStarted { provider, mode, .. } => Some((
                METHOD_AGENT_SESSION_STARTED,
                to_value(SessionStartedParams {
                    session_id: self.session_id.to_string(),
                    mode: mode.to_string(),
                    provider: provider.to_string(),
                }),
            )),
            AgentEvent::SessionStopped { reason, .. } => Some((
                METHOD_AGENT_SESSION_STOPPED,
                to_value(SessionStoppedParams {
                    session_id: self.session_id.to_string(),
                    reason: reason.clone(),
                }),
            )),
            AgentEvent::ProposalReady { proposal_id, .. } => Some((
                METHOD_AGENT_PROPOSAL_READY,
                to_value(ProposalReadyParams {
                    session_id: self.session_id.to_string(),
                    proposal_id: proposal_id.clone(),
                }),
            )),
        }
    }

    fn tool_params(
        &self,
        tool: &reimagine_agent::ToolName,
        tool_call_id: &Option<ToolCallId>,
        error: Option<String>,
    ) -> Value {
        to_value(ToolEventParams {
            session_id: self.session_id.to_string(),
            turn_id: self.turn_id.to_string(),
            tool: tool.to_string(),
            tool_call_id: tool_call_id
                .as_ref()
                .map(ToolCallId::as_str)
                .unwrap_or_default()
                .to_string(),
            error,
        })
    }
}

fn v1_permissions() -> PermissionSet {
    PermissionSet::from_iter([
        ToolPermission::new("workflow.read"),
        ToolPermission::new("workflow.write"),
        ToolPermission::new("model.read"),
        ToolPermission::new("model.write"),
    ])
}

fn turn_input_text(input: &Value) -> Option<String> {
    match input {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => map.get("text").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

fn turn_result_value(result: &AgentTurnResult) -> Value {
    json!({
        "status": result.status().as_str(),
        "stop_reason": result.stop_reason().as_str(),
        "final_response": result.final_response().map(Message::content).unwrap_or_default(),
        "tool_calls": result.tool_calls().iter().map(|call| {
            json!({
                "tool": call.tool_name().to_string(),
                "tool_call_id": call.tool_call_id().as_str(),
                "status": call.status().as_str(),
                "effective": call.effective(),
                "output": call.output(),
            })
        }).collect::<Vec<Value>>(),
        "usage": result.usage().map(|usage| json!({
            "input_tokens": usage.input_tokens(),
            "output_tokens": usage.output_tokens(),
        })),
    })
}

fn unknown_session(session_id: &AgentSessionId) -> JsonRpcError {
    JsonRpcError::invalid_params().with_data(json!({
        "session_id": session_id.to_string(),
    }))
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn write_line<W: Write, T: Serialize>(writer: &Mutex<W>, message: &T) -> io::Result<()> {
    let mut writer = lock(writer);
    let line = serde_json::to_string(message).expect("protocol messages always serialize");
    writer.write_all(line.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn from_value<T: DeserializeOwned>(value: Value) -> Result<T, JsonRpcError> {
    serde_json::from_value(value).map_err(|_| JsonRpcError::invalid_params())
}

fn to_value<T: Serialize>(value: T) -> Value {
    serde_json::to_value(value).expect("protocol values always serialize")
}

fn timestamp_nanos() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .to_string()
}

fn timestamp_millis() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .to_string()
}
