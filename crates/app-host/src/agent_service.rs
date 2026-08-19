use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use reimagine_agent_harness::{
    AgentEventSink, AgentLoop, AgentMode, AgentSession, AgentSessionId, AgentToolRegistry,
    AgentTurnId, AgentTurnRequest, AgentTurnResult, ContextConfig, ContextManager, Message,
    ModelName, PermissionSet, ProviderName, VecAgentEventSink, WorkspaceScope,
};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::AgentProviderCatalog;
use serde_json::json;

const DEFAULT_SYSTEM_PROMPT: &str = "You are Reimagine, a careful creative agent. Use tools deliberately and explain actions. Board edits must preserve project intent; Workflow edits must remain valid and use the workflow command path. Prefer inspection before mutation and respect tool results.";

fn model_options(config: Option<&crate::ProviderConfig>) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    if let Some(c) = config {
        if let Some(v) = c.max_tokens() {
            out.insert("max_tokens".into(), json!(v));
        }
        if let Some(v) = c.temperature() {
            out.insert("temperature".into(), json!(v));
        }
        if let Some(v) = c.top_p() {
            out.insert("top_p".into(), json!(v));
        }
        if let Some(v) = c.top_k() {
            out.insert("top_k".into(), json!(v));
        }
    }
    serde_json::Value::Object(out)
}

/// Conservative minimum window budget (RF-B1 / AR-17): a model whose
/// window cannot be determined (or a broken config) must never produce a
/// sub-usable budget.
const MINIMUM_CONTEXT_BUDGET: usize = 16_000;

/// Known model-family context windows (RF-B1 / AR-17), used when the
/// provider config does not declare `context_window`. Values are
/// conservative; unknown models fall back to
/// `DEFAULT_CONTEXT_WINDOW_TOKENS` with a warning.
fn model_context_window(model: &ModelName) -> Option<usize> {
    let name = model.as_str();
    if name.starts_with("gpt-5")
        || name.starts_with("gpt-4o")
        || name.starts_with("gpt-4.1")
        || name.starts_with("chatgpt-4o")
    {
        Some(128_000)
    } else if name.starts_with("gpt-4") {
        // gpt-4 family (8k/32k variants) - conservative.
        Some(32_000)
    } else if name.starts_with("claude") {
        Some(200_000)
    } else if name.starts_with("gemini") {
        Some(1_000_000)
    } else if name.starts_with("o1-") || name.starts_with("o3-") || name.starts_with("o4-") {
        Some(200_000)
    } else {
        None
    }
}

/// Resolve the per-turn window budget (RF-B1 / AR-17).
///
/// Source order: provider config `context_window` (honoured exactly when
/// it is non-zero and at least the conservative minimum; a zero or
/// garbage value falls back) -> known model-family table ->
/// `DEFAULT_CONTEXT_WINDOW_TOKENS`. Every non-explicit resolution logs a
/// warning so silent regressions are audible.
fn resolve_context_budget(
    model: &ModelName,
    provider_config: Option<&crate::ProviderConfig>,
) -> usize {
    if let Some(window) = provider_config.and_then(|c| c.context_window()) {
        if window == 0 {
            tracing::warn!(%model, "provider context_window is 0; using default budget");
            return DEFAULT_CONTEXT_WINDOW_TOKENS;
        }
        if window < MINIMUM_CONTEXT_BUDGET {
            tracing::warn!(%model, window, "context_window below conservative minimum; using minimum");
            return MINIMUM_CONTEXT_BUDGET;
        }
        return window;
    }
    if let Some(window) = model_context_window(model) {
        return window;
    }
    tracing::warn!(%model, "no context window known for model; using default budget");
    DEFAULT_CONTEXT_WINDOW_TOKENS
}
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
    system_prompt_override: Mutex<Option<String>>,
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
            system_prompt_override: Mutex::new(None),
        }
    }

    pub fn session(&self) -> &AgentSession {
        &self.session
    }

    /// Set the in-memory session system prompt override (AR-17). Durable
    /// thread metadata (thread.json) remains AR-22's scope.
    pub fn set_system_prompt(&self, prompt: impl Into<String>) {
        *self
            .system_prompt_override
            .lock()
            .expect("agent prompt lock poisoned") = Some(prompt.into());
    }

    /// Effective system prompt: the runtime override wins over the
    /// session's own (builder-set) prompt.
    pub fn effective_system_prompt(&self) -> Option<String> {
        self.system_prompt_override
            .lock()
            .expect("agent prompt lock poisoned")
            .clone()
            .or_else(|| self.session.system_prompt().map(str::to_owned))
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
    output_schema: Option<serde_json::Value>,
    turn_timeout: Option<Duration>,
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
            output_schema: None,
            turn_timeout: None,
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

    /// Require the final assistant response to satisfy `schema`
    /// (AR-30 structured output).
    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    pub fn with_turn_timeout(mut self, timeout: Duration) -> Self {
        self.turn_timeout = Some(timeout);
        self
    }

    pub fn turn_timeout(&self) -> Option<Duration> {
        self.turn_timeout
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

    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        self.output_schema.as_ref()
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
        // `VecAgentEventSink` is a test/minimal fallback only. Production
        // bootstrap injects `TauriAgentEventHub` (or another real sink) via
        // `with_registry_providers_and_sink`.
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
        // `VecAgentEventSink` is a test/minimal fallback only; callers
        // on the production path should inject the host event hub.
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

    /// Create a project-scoped agent thread (AR-14).
    ///
    /// Unlike [`Self::create_session`], the resulting `AgentSession` is
    /// bound to `project_id` (project_id is carried on the session and
    /// propagated into every tool context), so project-owned tools can
    /// reject cross-project calls. Sessions remain globally keyed by
    /// `AgentSessionId`; `list_threads(project_id)` filters the same
    /// registry.
    pub fn create_thread(
        &self,
        project_id: reimagine_core::model::ProjectId,
        id: AgentSessionId,
        mode: AgentMode,
        provider: ProviderName,
        started_at: impl Into<String>,
    ) -> AgentSession {
        let session = AgentSession::new(
            id.clone(),
            self.workspace_scope.clone(),
            mode,
            provider,
            Arc::clone(&self.registry),
        )
        .with_started_at(started_at)
        .with_permissions(PermissionSet::new())
        .with_project_id(project_id);
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

    /// List the threads bound to `project_id` (AR-14).
    pub fn list_threads(&self, project_id: &reimagine_core::model::ProjectId) -> Vec<AgentSession> {
        self.sessions
            .read()
            .expect("agent session registry poisoned")
            .values()
            .filter_map(|runtime| {
                let session = runtime.session();
                match session.project_id() {
                    Some(pid) if pid == project_id => Some(session.clone()),
                    _ => None,
                }
            })
            .collect()
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
        .with_cancel_token(cancel_token)
        .with_system_prompt(
            runtime
                .effective_system_prompt()
                .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_owned()),
        )
        .with_options(model_options(
            self.providers.config(runtime.session().provider()).as_ref(),
        ));
        if let Some(max_tool_steps) = request.max_tool_steps() {
            turn_request = turn_request.with_max_tool_steps(max_tool_steps);
        }
        if let Some(output_schema) = request.output_schema() {
            turn_request = turn_request.with_output_schema(output_schema.clone());
        }
        turn_request = turn_request
            .with_turn_timeout(request.turn_timeout().unwrap_or(Duration::from_secs(300)));

        let loop_harness = AgentLoop::new(provider, Arc::clone(&self.event_sink));
        let mut context_guard = runtime.context.lock().await;
        // AR-17: target the window budget to the model's context window
        // (configured value, else known-family table, else default).
        context_guard.set_max_tokens(resolve_context_budget(
            request.model(),
            self.providers.config(runtime.session().provider()).as_ref(),
        ));
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

    /// Set the in-memory system prompt override for a thread (AR-17).
    /// The override wins over the default template on the next turn.
    pub fn set_session_system_prompt(
        &self,
        session_id: &AgentSessionId,
        prompt: impl Into<String>,
    ) -> AppHostResult<()> {
        let runtime = self.get_runtime(session_id)?;
        runtime.set_system_prompt(prompt);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent_harness::ModelName;

    fn provider_config_with(
        window: Option<usize>,
        temperature: Option<f32>,
    ) -> crate::ProviderConfig {
        let mut json = serde_json::json!({
            "name": "mock",
            "protocol": "openai_chat_completions",
        });
        if let Some(w) = window {
            json["context_window"] = serde_json::json!(w);
        }
        if let Some(t) = temperature {
            json["temperature"] = serde_json::json!(t);
        }
        serde_json::from_value(json).expect("provider config parses")
    }

    #[test]
    fn resolve_context_budget_honors_explicit_config_window() {
        let config = provider_config_with(Some(20_000), None);
        let budget = resolve_context_budget(&ModelName::new("ghi"), Some(&config));
        assert_eq!(budget, 20_000, "explicit config window wins");
    }

    #[test]
    fn resolve_context_budget_zero_and_tiny_config_windows_fall_back() {
        let zero = provider_config_with(Some(0), None);
        assert_eq!(
            resolve_context_budget(&ModelName::new("x"), Some(&zero)),
            DEFAULT_CONTEXT_WINDOW_TOKENS,
            "zero window uses the default"
        );
        let tiny = provider_config_with(Some(100), None);
        assert_eq!(
            resolve_context_budget(&ModelName::new("x"), Some(&tiny)),
            MINIMUM_CONTEXT_BUDGET,
            "sub-minimum window clamps to the conservative floor"
        );
    }

    #[test]
    fn resolve_context_budget_uses_model_table_when_config_is_absent() {
        assert_eq!(
            resolve_context_budget(&ModelName::new("gpt-4o"), None),
            128_000,
            "128k-class model"
        );
        assert_eq!(
            resolve_context_budget(&ModelName::new("gpt-4"), None),
            32_000,
            "32k-class model differs from 128k"
        );
        assert_eq!(
            resolve_context_budget(&ModelName::new("unknown-model"), None),
            DEFAULT_CONTEXT_WINDOW_TOKENS,
            "unknown model falls back to the default"
        );
    }

    #[test]
    fn model_options_only_include_configured_fields() {
        let config = provider_config_with(None, Some(0.7));
        let options = model_options(Some(&config));
        let map = options.as_object().expect("options is an object");
        let temperature = map
            .get("temperature")
            .and_then(|v| v.as_f64())
            .expect("temperature present");
        assert!(
            (temperature - 0.7).abs() < 1e-6,
            "temperature {temperature} within tolerance"
        );
        assert!(!map.contains_key("max_tokens"), "unset fields are omitted");
        assert!(!map.contains_key("top_p"));
        assert!(!map.contains_key("top_k"));
    }

    #[test]
    fn model_options_are_empty_without_config() {
        let options = model_options(None);
        assert_eq!(options, serde_json::json!({}));
    }
}
