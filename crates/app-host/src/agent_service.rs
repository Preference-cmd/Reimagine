use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use reimagine_agent::{
    AgentMode, AgentSession, AgentSessionId, AgentToolRegistry, ProviderName, WorkspaceScope,
};

use crate::{AppHostError, AppHostResult};

#[derive(Debug)]
pub struct AgentService {
    workspace_scope: WorkspaceScope,
    registry: Arc<AgentToolRegistry>,
    sessions: RwLock<BTreeMap<AgentSessionId, AgentSession>>,
}

impl AgentService {
    pub fn new(workspace_scope: WorkspaceScope) -> Self {
        Self::with_registry(workspace_scope, Arc::new(AgentToolRegistry::new()))
    }

    pub fn with_registry(
        workspace_scope: WorkspaceScope,
        registry: Arc<AgentToolRegistry>,
    ) -> Self {
        Self {
            workspace_scope,
            registry,
            sessions: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn workspace_scope(&self) -> &WorkspaceScope {
        &self.workspace_scope
    }

    pub fn registry(&self) -> &Arc<AgentToolRegistry> {
        &self.registry
    }

    pub fn create_session(
        &self,
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
        .with_started_at(started_at);
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
}
