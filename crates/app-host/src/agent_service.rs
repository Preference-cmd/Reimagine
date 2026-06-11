use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use reimagine_agent::{
    AgentEventSink, AgentLoop, AgentMode, AgentSession, AgentSessionId, AgentToolRegistry,
    AgentTurnId, AgentTurnRequest, AgentTurnResult, Message, ModelName, PermissionSet,
    ProviderName, VecAgentEventSink, WorkspaceScope,
};

use crate::AgentProviderCatalog;
use crate::{AppHostError, AppHostResult};

pub struct AgentService {
    workspace_scope: WorkspaceScope,
    registry: Arc<AgentToolRegistry>,
    providers: AgentProviderCatalog,
    event_sink: Arc<dyn AgentEventSink>,
    sessions: RwLock<BTreeMap<AgentSessionId, AgentSession>>,
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

    pub fn with_registry_providers_and_sink(
        workspace_scope: WorkspaceScope,
        registry: Arc<AgentToolRegistry>,
        providers: AgentProviderCatalog,
        event_sink: Arc<dyn AgentEventSink>,
    ) -> Self {
        Self {
            workspace_scope,
            registry,
            providers,
            event_sink,
            sessions: RwLock::new(BTreeMap::new()),
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
        self.sessions
            .write()
            .expect("agent session registry poisoned")
            .insert(id, session.clone());
        session
    }

    pub fn get_session(&self, id: &AgentSessionId) -> AppHostResult<AgentSession> {
        self.sessions
            .read()
            .expect("agent session registry poisoned")
            .get(id)
            .cloned()
            .ok_or_else(|| AppHostError::UnknownAgentSession {
                session_id: id.clone(),
            })
    }

    pub fn list_sessions(&self) -> Vec<AgentSession> {
        self.sessions
            .read()
            .expect("agent session registry poisoned")
            .values()
            .cloned()
            .collect()
    }

    pub async fn run_turn(
        &self,
        request: AgentServiceTurnRequest,
    ) -> AppHostResult<AgentTurnResult> {
        let session = self.get_session(request.session_id())?;
        let provider = self.providers.get(session.provider()).ok_or_else(|| {
            AppHostError::UnknownAgentProvider {
                provider: session.provider().clone(),
            }
        })?;
        let mut turn_request = AgentTurnRequest::new(
            session,
            request.turn_id().clone(),
            request.model().clone(),
            request.input().to_vec(),
        );
        if let Some(max_tool_steps) = request.max_tool_steps() {
            turn_request = turn_request.with_max_tool_steps(max_tool_steps);
        }
        let loop_harness = AgentLoop::new(provider, Arc::clone(&self.event_sink));
        Ok(loop_harness.run_turn(turn_request).await)
    }
}
