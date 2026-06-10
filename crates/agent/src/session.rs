//! Workspace-scoped agent session.
//!
//! V1 keeps `AgentSession` as a small in-memory record that holds the
//! session id, the bound workspace scope, the active mode, the
//! provider, the registry, and a start timestamp. Concrete
//! orchestration (model selection, tool execution, streaming) lives
//! in `app-host`. The session is `Clone`-friendly because every field
//! is either trivially copyable or wrapped in `Arc`.

use std::sync::Arc;

use crate::ids::{AgentSessionId, ProviderName, WorkspaceScope};
use crate::mode::AgentMode;
use crate::registry::AgentToolRegistry;

/// In-memory V1 agent session. Bound to a single workspace scope.
#[derive(Clone)]
pub struct AgentSession {
    id: AgentSessionId,
    workspace_scope: WorkspaceScope,
    mode: AgentMode,
    provider: ProviderName,
    registry: Arc<AgentToolRegistry>,
    started_at: String,
}

impl std::fmt::Debug for AgentSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSession")
            .field("id", &self.id)
            .field("workspace_scope", &self.workspace_scope)
            .field("mode", &self.mode)
            .field("provider", &self.provider)
            .field("registry", &"Arc<AgentToolRegistry>")
            .field("started_at", &self.started_at)
            .finish()
    }
}

impl AgentSession {
    /// Build a new session bound to `workspace_scope`.
    pub fn new(
        id: AgentSessionId,
        workspace_scope: WorkspaceScope,
        mode: AgentMode,
        provider: ProviderName,
        registry: Arc<AgentToolRegistry>,
    ) -> Self {
        Self {
            id,
            workspace_scope,
            mode,
            provider,
            registry,
            started_at: String::new(),
        }
    }

    /// Builder-style: attach a `started_at` timestamp string. The
    /// runtime does not read the clock; the host supplies the value.
    pub fn with_started_at(mut self, ts: impl Into<String>) -> Self {
        self.started_at = ts.into();
        self
    }

    pub fn id(&self) -> &AgentSessionId {
        &self.id
    }

    pub fn workspace_scope(&self) -> &WorkspaceScope {
        &self.workspace_scope
    }

    pub fn mode(&self) -> AgentMode {
        self.mode
    }

    pub fn provider(&self) -> &ProviderName {
        &self.provider
    }

    pub fn registry(&self) -> &Arc<AgentToolRegistry> {
        &self.registry
    }

    pub fn started_at(&self) -> &str {
        &self.started_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ToolName;
    use crate::permissions::{ToolPermission, ToolRiskLevel};
    use crate::tool::ToolSpec;
    use crate::tool::{AgentTool, ToolResult};
    use async_trait::async_trait;
    use serde_json::Value;

    struct Noop;
    #[async_trait]
    impl AgentTool for Noop {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new(
                ToolName::new("noop"),
                "noop",
                [AgentMode::Agent],
                ToolPermission::new("workflow.read"),
                ToolRiskLevel::Read,
            )
        }
        async fn invoke(&self, _ctx: &crate::context::ToolContext, _input: Value) -> ToolResult {
            Ok(Value::Null)
        }
    }

    #[test]
    fn session_is_workspace_scoped() {
        let mut reg = AgentToolRegistry::new();
        reg.register(Noop).unwrap();
        let reg = Arc::new(reg);
        let sess = AgentSession::new(
            AgentSessionId::new("sess-1"),
            WorkspaceScope::new("ws-alpha"),
            AgentMode::Agent,
            ProviderName::new("openai"),
            reg,
        )
        .with_started_at("2026-06-10T00:00:00Z");
        assert_eq!(sess.workspace_scope().as_str(), "ws-alpha");
        assert_eq!(sess.mode(), AgentMode::Agent);
        assert_eq!(sess.provider().as_str(), "openai");
        assert_eq!(sess.started_at(), "2026-06-10T00:00:00Z");
    }
}
