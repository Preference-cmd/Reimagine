//! Daemon server core: JSON-RPC dispatch, session management, and turn
//! execution.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use reimagine_agent_harness::{
    AgentEvent, AgentEventSink, AgentLoop, AgentMode, AgentSessionId, AgentTurnId,
    AgentTurnRequest, AgentTurnResult, ContentBlock, ContextConfig, ContextManager, Message,
    ModelName, PermissionSet, ProviderName, ToolCallId, ToolName, ToolPermission, VecAgentEventSink,
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
    METHOD_AGENT_CONTEXT_COMPACTED, METHOD_AGENT_ERROR, METHOD_AGENT_PROPOSAL_READY,
    METHOD_AGENT_REASONING_DELTA,
    METHOD_AGENT_SESSION_STARTED, METHOD_AGENT_SESSION_STOPPED, METHOD_AGENT_TOOL_COMPLETED,
    METHOD_AGENT_TOOL_FAILED, METHOD_AGENT_TOOL_INVOKED, METHOD_AGENT_TURN_COMPLETED,
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_PROVIDERS_LIST, METHOD_SESSION_CREATE,
    METHOD_SESSION_GET, METHOD_SESSION_LIST, METHOD_TURN_CANCEL, METHOD_TURN_RUN, METHOD_TURN_STEER,
    ContextCompactedParams, ProposalReadyParams, ProviderInfo, ProvidersListResult,
    ReasoningDeltaParams,
    ServerCapabilities, ServerInfo, SessionCreateParams, SessionCreateResult, SessionGetParams,
    SessionInfo, SessionListResult, SessionStartedParams, SessionStoppedParams, ToolEventParams,
    TurnCancelParams, TurnCancelResult, TurnCancelStatus, TurnCompletedParams,
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

/// Outcome of a resumed session's tool registry compatibility check.
///
/// A session is always allowed to resume; `compatible` reports whether
/// tool drift was detected so hosts can surface it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRegistryValidation {
    /// `true` when the persisted tool snapshot matches the current
    /// registry. V1 persists no snapshot, so this is always `true`
    /// unless the caller supplies one.
    pub compatible: bool,
    /// Tool names recorded at persist time that are no longer
    /// registered in the workspace.
    pub missing: Vec<String>,
    /// Tool names that are still registered but whose schema changed.
    /// V1 does not persist schemas, so this is always empty; the field
    /// exists for a future persistence format.
    pub schema_changed: Vec<String>,
}

/// Minimal metadata read from a persisted session file at resume time.
/// V1 files written by `ContextManager::persist` carry `created_at`;
/// the history itself is loaded separately via `ContextManager::load`.
#[derive(serde::Deserialize)]
struct PersistedSessionMetadata {
    #[serde(default)]
    created_at: String,
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
    /// before `session.create` / `turn.run` succeed. After workspace
    /// init, sessions persisted by a previous daemon run are resumed
    /// from `{workspace_dir}/agent-sessions` (see
    /// [`Self::resume_persisted_sessions`]).
    pub async fn new(workspace_dir: &Path) -> Result<Self, DaemonInitError> {
        let sink: Arc<dyn AgentEventSink> = Arc::new(VecAgentEventSink::new());
        let workspace = DaemonWorkspace::initialize(workspace_dir, sink).await?;
        let mut daemon = Self {
            workspace,
            sessions: HashMap::new(),
            active_turns: Arc::new(Mutex::new(HashMap::new())),
            client: None,
        };
        daemon.resume_persisted_sessions();
        Ok(daemon)
    }

    pub fn workspace(&self) -> &DaemonWorkspace {
        &self.workspace
    }

    /// Client identity reported by the last `initialize` handshake.
    pub fn client_info(&self) -> Option<&ClientInfo> {
        self.client.as_ref()
    }

    /// Scan the session directory for persisted sessions and load each
    /// into the live session map.
    ///
    /// Called by [`AgentDaemon::new`] after workspace init so a daemon
    /// restart makes prior sessions available through `session.list`
    /// and `turn.run` with their full history. Corrupt or partial
    /// session files are logged and skipped — startup never fails
    /// because of a bad session file. Re-scanning is idempotent:
    /// sessions already in the map are skipped.
    ///
    /// V1 persisted sessions store conversation history only. Mode and
    /// provider are not persisted, so resumed sessions default to
    /// `AgentMode::Agent` and the first provider registered in the
    /// catalog (an empty provider name when the catalog is empty;
    /// `turn.run` then falls back to the current default provider or
    /// fails with an unknown-provider error). Tool registry snapshots
    /// are not persisted either, so the compatibility check runs with
    /// an empty snapshot.
    ///
    /// Returns the number of sessions resumed.
    pub fn resume_persisted_sessions(&mut self) -> usize {
        let config = self.session_config();
        let session_dir = config.session_dir.clone();
        let entries = match std::fs::read_dir(&session_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return 0,
            Err(error) => {
                tracing::warn!(
                    path = %session_dir.display(),
                    %error,
                    "failed to scan session directory"
                );
                return 0;
            }
        };
        let mut resumed = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(OsStr::to_str) != Some("json") {
                continue;
            }
            let Some(session_id) = path.file_stem().and_then(OsStr::to_str) else {
                continue;
            };
            let id = AgentSessionId::new(session_id);
            if self.sessions.contains_key(&id) {
                continue;
            }
            let context = match ContextManager::load(
                session_id,
                ContextConfig::new(config.max_tokens, config.recent_turns, session_dir.clone()),
            ) {
                Ok(context) => context,
                Err(error) => {
                    tracing::warn!(
                        session_id,
                        %error,
                        "failed to load persisted session; skipping"
                    );
                    continue;
                }
            };
            let created_at = persisted_created_at(&path).unwrap_or_default();
            let provider = self.default_resume_provider();
            self.workspace
                .agent_service()
                .create_session_with_permissions(
                    id.clone(),
                    AgentMode::Agent,
                    provider.clone(),
                    created_at.clone(),
                    v1_permissions(),
                );
            self.validate_tool_registry(&id, &[]);
            let info = SessionInfo {
                session_id: id.to_string(),
                mode: AgentMode::Agent.to_string(),
                provider: provider.to_string(),
                created_at,
            };
            let state = SessionState {
                info,
                context: Arc::new(AsyncMutex::new(context)),
                turn_lock: Arc::new(Semaphore::new(1)),
            };
            self.sessions.insert(id, state);
            resumed += 1;
        }
        resumed
    }

    /// Context configuration shared by new and resumed sessions: the
    /// persistence directory under the workspace base path.
    fn session_config(&self) -> ContextConfig {
        ContextConfig::new(64_000, 20, self.workspace.host().base_path().join("agent-sessions"))
    }

    /// Provider bound to resumed sessions. V1 does not persist the
    /// provider, so resumed sessions default to the first registered
    /// provider, or an empty provider name when none is registered —
    /// `turn.run` then falls back to the current default provider or
    /// fails with an unknown-provider error.
    fn default_resume_provider(&self) -> ProviderName {
        self.workspace
            .providers()
            .provider_names()
            .into_iter()
            .next()
            .unwrap_or_else(|| ProviderName::new(""))
    }

    /// Validate the tool registry against the tool names recorded when
    /// a session was persisted.
    ///
    /// V1 does not persist tool information, so the daemon always calls
    /// this with an empty snapshot and it logs a debug note. The seam
    /// exists so a future persistence format can detect tool drift
    /// between daemon restarts. Missing or changed tools are reported
    /// in the result and logged as a warning, but never block
    /// resumption: turns surface unknown-tool failures at runtime.
    pub fn validate_tool_registry(
        &self,
        session_id: &AgentSessionId,
        persisted_tools: &[String],
    ) -> ToolRegistryValidation {
        if persisted_tools.is_empty() {
            tracing::debug!(
                session_id = %session_id,
                "no tool registry snapshot in persisted session; skipping compatibility check"
            );
            return ToolRegistryValidation {
                compatible: true,
                missing: Vec::new(),
                schema_changed: Vec::new(),
            };
        }
        let current = self.workspace.registry().tool_names();
        let missing: Vec<String> = persisted_tools
            .iter()
            .filter(|tool| !current.contains(&ToolName::new(tool.as_str())))
            .cloned()
            .collect();
        if !missing.is_empty() {
            tracing::warn!(
                session_id = %session_id,
                missing = ?missing,
                "persisted session tool registry drift detected; resuming anyway"
            );
        }
        ToolRegistryValidation {
            compatible: missing.is_empty(),
            missing,
            schema_changed: Vec::new(),
        }
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

    /// Persist every live session context to its session file.
    ///
    /// Part of the graceful shutdown path, called after the serve loop
    /// returns at EOF. One session failing to persist does not block the
    /// rest: failures are logged and the remaining sessions still write.
    /// Returns the number of sessions persisted.
    pub async fn persist_all(&self) -> usize {
        let mut persisted = 0;
        for (session_id, state) in &self.sessions {
            let context = state.context.lock().await;
            match context.persist(session_id.as_str()) {
                Ok(()) => persisted += 1,
                Err(error) => tracing::warn!(
                    session = %session_id,
                    %error,
                    "failed to persist session context during shutdown"
                ),
            }
        }
        persisted
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
        let config = self.session_config();
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
        let input_blocks = turn_input_blocks(&request.input)?;
        let agent_session = self
            .workspace
            .agent_service()
            .get_session(&session_id)
            .map_err(|_| unknown_session(&session_id))?;
        let provider = if agent_session.provider().as_str().is_empty() {
            // Resumed V1 sessions do not persist their provider; when
            // none was registered at resume time the session carries an
            // empty provider name, so bind the turn to the current
            // default provider.
            self.workspace
                .providers()
                .provider_names()
                .into_iter()
                .next()
                .and_then(|name| self.workspace.providers().get(&name))
        } else {
            self.workspace.providers().get(agent_session.provider())
        };
        let provider = provider.ok_or_else(|| {
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
                vec![Message::user_with_blocks(input_blocks)],
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
            AgentEvent::ReasoningDelta { text, .. } => Some((
                METHOD_AGENT_REASONING_DELTA,
                to_value(ReasoningDeltaParams {
                    session_id: self.session_id.to_string(),
                    turn_id: self.turn_id.to_string(),
                    text: text.clone(),
                }),
            )),
            AgentEvent::ContextCompacted {
                summary,
                tokens_before,
                tokens_after,
                ..
            } => Some((
                METHOD_AGENT_CONTEXT_COMPACTED,
                to_value(ContextCompactedParams {
                    session_id: self.session_id.to_string(),
                    turn_id: self.turn_id.to_string(),
                    summary: summary.clone(),
                    tokens_before: *tokens_before as u64,
                    tokens_after: *tokens_after as u64,
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
        tool: &reimagine_agent_harness::ToolName,
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

/// Read the `created_at` field out of a persisted session file. The
/// field is opaque (V1 writes unix seconds); an unreadable or partial
/// file yields `None` and the caller falls back to an empty timestamp.
fn persisted_created_at(path: &Path) -> Option<String> {
    let json = std::fs::read(path).ok()?;
    serde_json::from_slice::<PersistedSessionMetadata>(&json)
        .ok()
        .map(|meta| meta.created_at)
}

/// Maximum decoded size of an inline (base64) file block, in base64
/// characters. This is a **character-count** limit, not a decoded-byte
/// check: `10 * 1024 * 1024 * 4 / 3` ≈ 13.3M chars, which by integer
/// truncation admits files up to 10MiB − 1 byte (conservative, safe
/// direction). Base64 validity is the provider's concern downstream.
const MAX_INLINE_FILE_BASE64_CHARS: usize = 10 * 1024 * 1024 * 4 / 3;

/// Maximum number of file blocks accepted in a single turn input.
const MAX_FILE_BLOCKS_PER_TURN: usize = 4;

/// Parse the `turn.run` input into content blocks.
///
/// Three shapes are accepted:
/// - a plain string (`"hi"`),
/// - a compatibility object (`{"text":"hi"}`),
/// - a content-block array (`[{type:"text",text:"hi"},{type:"file",...}]`).
///
/// File blocks are validated: `media_type` must be non-empty, inline
/// base64 must stay within the 10MB (decoded) limit, and at most
/// [`MAX_FILE_BLOCKS_PER_TURN`] file blocks are allowed per turn.
/// Invalid inputs yield an `invalid_params` error.
fn turn_input_blocks(input: &Value) -> Result<Vec<ContentBlock>, JsonRpcError> {
    let blocks = match input {
        Value::String(text) => vec![ContentBlock::Text(text.clone())],
        Value::Object(map) => {
            let text = map
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_input("turn input object must carry a `text` string"))?;
            vec![ContentBlock::Text(text.to_owned())]
        }
        Value::Array(items) => {
            let mut blocks = Vec::with_capacity(items.len());
            for item in items {
                let block = serde_json::from_value::<ContentBlock>(item.clone()).map_err(|_| {
                    invalid_input(
                        "invalid content block: expected `{type:\"text\",text}` or `{type:\"file\",...}`",
                    )
                })?;
                blocks.push(block);
            }
            blocks
        }
        _ => {
            return Err(invalid_input(
                "turn input must be a string, a `{text}` object, or a content-block array",
            ))
        }
    };
    validate_turn_blocks(&blocks)?;
    Ok(blocks)
}

fn validate_turn_blocks(blocks: &[ContentBlock]) -> Result<(), JsonRpcError> {
    let file_count = blocks
        .iter()
        .filter(|block| matches!(block, ContentBlock::File(_)))
        .count();
    if file_count > MAX_FILE_BLOCKS_PER_TURN {
        return Err(invalid_input(format!(
            "at most {MAX_FILE_BLOCKS_PER_TURN} file blocks allowed per turn"
        )));
    }
    for block in blocks {
        if let ContentBlock::File(file) = block {
            if file.media_type().is_empty() {
                return Err(invalid_input("file block `media_type` must not be empty"));
            }
            if let Some(base64) = file.source().base64()
                && base64.len() > MAX_INLINE_FILE_BASE64_CHARS
            {
                return Err(invalid_input(
                    "inline file base64 exceeds the 10MB limit",
                ));
            }
        }
    }
    Ok(())
}

fn invalid_input(reason: impl Into<String>) -> JsonRpcError {
    JsonRpcError::invalid_params().with_data(json!({ "reason": reason.into() }))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(value: Value) -> Result<Vec<ContentBlock>, JsonRpcError> {
        turn_input_blocks(&value)
    }

    fn file_block(media_type: &str, base64: &str) -> Value {
        json!({
            "type": "file",
            "media_type": media_type,
            "source": { "type": "data", "base64": base64 },
        })
    }

    #[test]
    fn string_input_becomes_text_block() {
        assert_eq!(parse(json!("hi")).unwrap(), vec![ContentBlock::Text("hi".into())]);
    }

    #[test]
    fn text_object_is_compat_shape() {
        assert_eq!(
            parse(json!({"text": "hi"})).unwrap(),
            vec![ContentBlock::Text("hi".into())]
        );
    }

    #[test]
    fn text_object_without_text_is_rejected() {
        let err = parse(json!({"other": 1})).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn blocks_array_mixes_text_and_file() {
        let blocks = parse(json!([
            {"type": "text", "text": "describe"},
            {"type": "file", "media_type": "image/png", "source": {"type": "data", "base64": "AAAA"}},
            {"type": "file", "media_type": "image/jpeg", "source": {"type": "url", "url": "refs/pic.jpg"}, "filename": "pic.jpg"},
        ]))
        .unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0], ContentBlock::Text("describe".into()));
        let ContentBlock::File(file) = &blocks[1] else {
            panic!("expected file block");
        };
        assert_eq!(file.media_type(), "image/png");
        assert_eq!(file.source().base64(), Some("AAAA"));
        let ContentBlock::File(file) = &blocks[2] else {
            panic!("expected file block");
        };
        assert_eq!(file.media_type(), "image/jpeg");
        assert_eq!(file.source().url(), Some("refs/pic.jpg"));
        assert_eq!(file.filename(), Some("pic.jpg"));
    }

    #[test]
    fn unknown_block_type_is_rejected() {
        let err = parse(json!([{"type": "video", "url": "clip.mp4"}])).unwrap_err();
        assert_eq!(err.code, -32602);
    }

    #[test]
    fn empty_media_type_is_rejected() {
        let err = parse(json!([file_block("", "AAAA")])).unwrap_err();
        assert_eq!(err.code, -32602);
        assert!(err
            .data
            .as_ref()
            .and_then(|data| data.get("reason"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("media_type"));
    }

    #[test]
    fn oversized_inline_file_is_rejected() {
        let big = "A".repeat(MAX_INLINE_FILE_BASE64_CHARS + 1);
        let err = parse(json!([file_block("image/png", &big)])).unwrap_err();
        assert_eq!(err.code, -32602);
        assert!(err
            .data
            .as_ref()
            .and_then(|data| data.get("reason"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("10MB"));
    }

    #[test]
    fn at_limit_inline_file_is_accepted() {
        let big = "A".repeat(MAX_INLINE_FILE_BASE64_CHARS);
        let blocks = parse(json!([file_block("image/png", &big)])).unwrap();
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn too_many_file_blocks_are_rejected() {
        let blocks: Vec<Value> = (0..=MAX_FILE_BLOCKS_PER_TURN)
            .map(|_| file_block("image/png", "AAAA"))
            .collect();
        let err = parse(Value::Array(blocks)).unwrap_err();
        assert_eq!(err.code, -32602);
        assert!(err
            .data
            .as_ref()
            .and_then(|data| data.get("reason"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("4 file blocks"));
    }

    #[test]
    fn url_source_blocks_count_toward_the_limit() {
        let blocks: Vec<Value> = (0..=MAX_FILE_BLOCKS_PER_TURN)
            .map(|i| {
                json!({
                    "type": "file",
                    "media_type": "image/png",
                    "source": { "type": "url", "url": format!("refs/{i}.png") },
                })
            })
            .collect();
        assert!(parse(Value::Array(blocks)).is_err());
    }

    #[test]
    fn non_text_non_object_input_is_rejected() {
        assert!(parse(json!(42)).is_err());
        assert!(parse(Value::Null).is_err());
        assert!(parse(json!(true)).is_err());
    }
}
