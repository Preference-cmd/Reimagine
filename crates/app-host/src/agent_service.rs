use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use reimagine_agent_harness::{
    AgentEventSink, AgentLoop, AgentMode, AgentSession, AgentSessionId, AgentToolRegistry,
    AgentTurnId, AgentTurnRequest, AgentTurnResult, ContextConfig, ContextManager, Message,
    ModelName, PermissionSet, ProviderName, VecAgentEventSink, WorkspaceScope,
};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::AgentProviderCatalog;
use crate::{AppHostError, AppHostResult};

/// Conservative V1 context budget per embedded agent session (AR-02).
///
/// `ContextConfig` hard-caps the rolling window at this many estimated
/// tokens. Per-model budgets arrive with RF-B1 / AR-17; until then a
/// single conservative default keeps every session bounded.
pub const DEFAULT_CONTEXT_WINDOW_TOKENS: usize = 64_000;

/// Fallback session directory for services built without an explicit
/// workspace layout. Production hosts always pass a real directory
/// (see `WorkspaceHost`); the fallback only serves bare test services.
fn fallback_session_dir() -> PathBuf {
    std::env::temp_dir().join("reimagine-agent-sessions")
}

/// Per-session runtime state for the embedded agent (AR-02).
///
/// Mirrors the daemon's `SessionState`: one `ContextManager` per
/// session, a per-session turn permit, and the cancellation token of
/// the in-flight turn. Sessions are held behind `Arc` so the service
/// map stays cheap to clone.
pub struct SessionRuntime {
    session: AgentSession,
    context: Arc<AsyncMutex<ContextManager>>,
    turn_lock: Arc<Semaphore>,
    active_turn: Mutex<Option<CancellationToken>>,
}

impl std::fmt::Debug for SessionRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionRuntime")
            .field("session", &self.session)
            .field("context", &"Arc<AsyncMutex<ContextManager>>")
            .field("turn_lock", &self.turn_lock)
            .field("active_turn", &self.active_turn)
            .finish()
    }
}
impl SessionRuntime {
    fn new(session: AgentSession, context_dir: PathBuf) -> Self {
        let context = ContextManager::new(ContextConfig::new(
            DEFAULT_CONTEXT_WINDOW_TOKENS,
            context_dir,
        ));
        Self {
            session,
            context: Arc::new(AsyncMutex::new(context)),
            turn_lock: Arc::new(Semaphore::new(1)),
            active_turn: Mutex::new(None),
        }
    }

    pub fn session(&self) -> &AgentSession {
        &self.session
    }

    fn set_active_turn(&self, token: CancellationToken) {
        *self
            .active_turn
            .lock()
            .expect("agent active-turn lock poisoned") = Some(token);
    }

    fn clear_active_turn(&self) {
        *self
            .active_turn
            .lock()
            .expect("agent active-turn lock poisoned") = None;
    }

    fn cancel_active_turn(&self) -> bool {
        match self
            .active_turn
            .lock()
            .expect("agent active-turn lock poisoned")
            .as_ref()
        {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }
}

pub struct AgentService {
    workspace_scope: WorkspaceScope,
    registry: Arc<AgentToolRegistry>,
    providers: AgentProviderCatalog,
    event_sink: Arc<dyn AgentEventSink>,
    sessions: RwLock<BTreeMap<AgentSessionId, Arc<SessionRuntime>>>,
    /// Directory where `ContextManager::persist` writes session
    /// context files (`{session_dir}/{session_id}.json`).
    session_dir: PathBuf,
}

impl std::fmt::Debug for AgentService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let session_count = self
            .sessions
            .read()
            .map(|sessions| sessions.len())
            .unwrap_or_default();
        f.debug_struct("AgentService")
            .field("workspace_scope", &self.workspace_scope)
            .field("registry", &self.registry)
            .field("providers", &self.providers)
            .field("event_sink", &"Arc<dyn AgentEventSink>")
            .field("session_count", &session_count)
            .field("session_dir", &self.session_dir)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct AgentServiceTurnRequest {
    session_id: AgentSessionId,
    turn_id: AgentTurnId,
    model: ModelName,
    input: Vec<Message>,
    max_tool_steps: Option<usize>,
}

impl AgentServiceTurnRequest {
    pub fn new(
        session_id: AgentSessionId,
        turn_id: AgentTurnId,
        model: ModelName,
        input: Vec<Message>,
    ) -> Self {
        Self {
            session_id,
            turn_id,
            model,
            input,
            max_tool_steps: None,
        }
    }

    pub fn from_user_text(
        session_id: AgentSessionId,
        turn_id: AgentTurnId,
        model: ModelName,
        text: impl Into<String>,
    ) -> Self {
        Self::new(session_id, turn_id, model, vec![Message::user(text)])
    }

    pub fn with_max_tool_steps(mut self, max_tool_steps: usize) -> Self {
        self.max_tool_steps = Some(max_tool_steps);
        self
    }

    pub fn session_id(&self) -> &AgentSessionId {
        &self.session_id
    }

    pub fn turn_id(&self) -> &AgentTurnId {
        &self.turn_id
    }

    pub fn model(&self) -> &ModelName {
        &self.model
    }

    pub fn input(&self) -> &[Message] {
        &self.input
    }

    pub fn max_tool_steps(&self) -> Option<usize> {
        self.max_tool_steps
    }
}

impl AgentService {
    pub fn new(workspace_scope: WorkspaceScope) -> Self {
        Self::with_registry(workspace_scope, Arc::new(AgentToolRegistry::new()))
    }

    pub fn with_registry(
        workspace_scope: WorkspaceScope,
        registry: Arc<AgentToolRegistry>,
    ) -> Self {
        Self::with_registry_providers_and_sink(
            workspace_scope,
            registry,
            AgentProviderCatalog::new(),
            Arc::new(VecAgentEventSink::new()),
        )
    }

    pub fn with_registry_and_session_dir(
        workspace_scope: WorkspaceScope,
        registry: Arc<AgentToolRegistry>,
        session_dir: impl Into<PathBuf>,
    ) -> Self {
        Self::with_registry_providers_sink_and_session_dir(
            workspace_scope,
            registry,
            AgentProviderCatalog::new(),
            Arc::new(VecAgentEventSink::new()),
            session_dir,
        )
    }

    pub fn with_registry_providers_and_sink(
        workspace_scope: WorkspaceScope,
        registry: Arc<AgentToolRegistry>,
        providers: AgentProviderCatalog,
        event_sink: Arc<dyn AgentEventSink>,
    ) -> Self {
        Self::with_registry_providers_sink_and_session_dir(
            workspace_scope,
            registry,
            providers,
            event_sink,
            fallback_session_dir(),
        )
    }

    pub fn with_registry_providers_sink_and_session_dir(
        workspace_scope: WorkspaceScope,
        registry: Arc<AgentToolRegistry>,
        providers: AgentProviderCatalog,
        event_sink: Arc<dyn AgentEventSink>,
        session_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            workspace_scope,
            registry,
            providers,
            event_sink,
            sessions: RwLock::new(BTreeMap::new()),
            session_dir: session_dir.into(),
        }
    }

    pub fn workspace_scope(&self) -> &WorkspaceScope {
        &self.workspace_scope
    }

    pub fn registry(&self) -> &Arc<AgentToolRegistry> {
        &self.registry
    }

    pub fn providers(&self) -> &AgentProviderCatalog {
        &self.providers
    }

    pub fn event_sink(&self) -> &Arc<dyn AgentEventSink> {
        &self.event_sink
    }

    pub fn session_dir(&self) -> &std::path::Path {
        &self.session_dir
    }

    pub fn create_session(
        &self,
        id: AgentSessionId,
        mode: AgentMode,
        provider: ProviderName,
        started_at: impl Into<String>,
    ) -> AgentSession {
        self.create_session_with_permissions(id, mode, provider, started_at, PermissionSet::new())
    }

    pub fn create_session_with_permissions(
        &self,
        id: AgentSessionId,
        mode: AgentMode,
        provider: ProviderName,
        started_at: impl Into<String>,
        permissions: PermissionSet,
    ) -> AgentSession {
        let session = AgentSession::new(
            id.clone(),
            self.workspace_scope.clone(),
            mode,
            provider,
            Arc::clone(&self.registry),
        )
        .with_started_at(started_at)
        .with_permissions(permissions);
        let runtime = Arc::new(SessionRuntime::new(
            session.clone(),
            self.session_dir.clone(),
        ));
        self.sessions
            .write()
            .expect("agent session registry poisoned")
            .insert(id, runtime);
        session
    }

    fn get_runtime(&self, id: &AgentSessionId) -> AppHostResult<Arc<SessionRuntime>> {
        self.sessions
            .read()
            .expect("agent session registry poisoned")
            .get(id)
            .cloned()
            .ok_or_else(|| AppHostError::UnknownAgentSession {
                session_id: id.clone(),
            })
    }

    pub fn get_session(&self, id: &AgentSessionId) -> AppHostResult<AgentSession> {
        Ok(self.get_runtime(id)?.session().clone())
    }

    pub fn list_sessions(&self) -> Vec<AgentSession> {
        self.sessions
            .read()
            .expect("agent session registry poisoned")
            .values()
            .map(|runtime| runtime.session().clone())
            .collect()
    }

    /// Run one turn for `request.session_id()`.
    ///
    /// Each session owns a single-turn permit: a second concurrent turn
    /// on the same session is rejected immediately with
    /// [`AppHostError::AgentTurnInProgress`], while different sessions
    /// run in parallel. The turn races its own `CancellationToken`
    /// (stored on the session runtime), so [`Self::cancel_turn`]
    /// aborts in-flight provider work.
    pub async fn run_turn(
        &self,
        request: AgentServiceTurnRequest,
    ) -> AppHostResult<AgentTurnResult> {
        let runtime = self.get_runtime(request.session_id())?;
        let _permit = runtime.turn_lock.clone().try_acquire_owned().map_err(|_| {
            AppHostError::AgentTurnInProgress {
                session_id: request.session_id().clone(),
            }
        })?;

        let provider = self
            .providers
            .get(runtime.session().provider())
            .ok_or_else(|| AppHostError::UnknownAgentProvider {
                provider: runtime.session().provider().clone(),
            })?;

        let cancel_token = CancellationToken::new();
        runtime.set_active_turn(cancel_token.clone());

        let mut turn_request = AgentTurnRequest::new(
            runtime.session().clone(),
            request.turn_id().clone(),
            request.model().clone(),
            request.input().to_vec(),
        )
        .with_cancel_token(cancel_token);
        if let Some(max_tool_steps) = request.max_tool_steps() {
            turn_request = turn_request.with_max_tool_steps(max_tool_steps);
        }

        let loop_harness = AgentLoop::new(provider, Arc::clone(&self.event_sink));
        let mut context_guard = runtime.context.lock().await;
        let result = loop_harness
            .run_turn_streaming(turn_request, Some(&mut context_guard))
            .await;

        // Persistence failures are non-fatal: the turn result is
        // authoritative, and the next turn continues from the in-memory
        // context. AR-22 makes lifecycle persistence first-class.
        if let Err(error) = context_guard.persist(request.session_id().as_str()) {
            tracing::warn!(
                session_id = request.session_id().as_str(),
                %error,
                "failed to persist agent session context"
            );
        }
        drop(context_guard);

        runtime.clear_active_turn();
        drop(_permit);
        Ok(result)
    }

    /// Cancel the in-flight turn on `session_id`, if any.
    ///
    /// Returns an error when the session is unknown or has no active
    /// turn. Cancellation is asynchronous from the caller's point of
    /// view: the running turn observes it at its next checkpoint and
    /// finishes with [`reimagine_agent_harness::AgentTurnStopReason::Cancelled`].
    pub fn cancel_turn(&self, session_id: &AgentSessionId) -> AppHostResult<()> {
        let runtime = self.get_runtime(session_id)?;
        if runtime.cancel_active_turn() {
            Ok(())
        } else {
            Err(AppHostError::NoActiveAgentTurn {
                session_id: session_id.clone(),
            })
        }
    }
}
