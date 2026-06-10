//! Host-neutral tool execution context.
//!
//! `ToolContext` carries metadata only: workspace scope, agent session id,
//! mode, correlation id, actor, and permissions. It does not carry
//! `app-host` handles or a generic service locator bag. Concrete app-host
//! tool functions capture the app-host `Arc<WorkspaceHost>` directly and
//! verify the incoming context's `workspace_scope` matches the captured
//! workspace.

use reimagine_core::diagnostic::CorrelationId;

use crate::ids::{AgentSessionId, WorkspaceScope};
use crate::mode::AgentMode;
use crate::permissions::PermissionSet;

/// Identity of the caller driving a tool invocation. Hosts may pass
/// "user", "agent", or another subsystem name. The value is opaque to
/// the agent crate — it is propagated into the diagnostic bridge and
/// into the audit trail.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Actor(String);

impl Actor {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Actor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Actor {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Actor {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl Default for Actor {
    fn default() -> Self {
        Self::new("agent")
    }
}

/// Metadata-only execution context for a tool call. Hosts (app-host,
/// tests) construct a `ToolContext` and pass it into the registry when
/// invoking a tool. The agent runtime never places app-host handles or
/// generic service-locator bags inside the context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolContext {
    workspace_scope: WorkspaceScope,
    agent_session_id: AgentSessionId,
    mode: AgentMode,
    correlation_id: Option<CorrelationId>,
    actor: Actor,
    permissions: PermissionSet,
}

impl ToolContext {
    /// Create a context for `workspace_scope`, `agent_session_id`, and
    /// `mode`. Defaults: no correlation id, `actor = "agent"`, empty
    /// permissions.
    pub fn new(
        workspace_scope: WorkspaceScope,
        agent_session_id: AgentSessionId,
        mode: AgentMode,
    ) -> Self {
        Self {
            workspace_scope,
            agent_session_id,
            mode,
            correlation_id: None,
            actor: Actor::default(),
            permissions: PermissionSet::new(),
        }
    }

    /// Attach a correlation id. Replaces any previously set correlation id.
    pub fn with_correlation_id(mut self, id: CorrelationId) -> Self {
        self.correlation_id = Some(id);
        self
    }

    /// Attach the actor driving this tool call.
    pub fn with_actor(mut self, actor: Actor) -> Self {
        self.actor = actor;
        self
    }

    /// Attach a permission set. Replaces any previously set permissions.
    pub fn with_permissions(mut self, permissions: PermissionSet) -> Self {
        self.permissions = permissions;
        self
    }

    pub fn workspace_scope(&self) -> &WorkspaceScope {
        &self.workspace_scope
    }

    pub fn agent_session_id(&self) -> &AgentSessionId {
        &self.agent_session_id
    }

    pub fn mode(&self) -> AgentMode {
        self.mode
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn actor(&self) -> &Actor {
        &self.actor
    }

    pub fn permissions(&self) -> &PermissionSet {
        &self.permissions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_builder_defaults() {
        let ctx = ToolContext::new(
            WorkspaceScope::new("ws-1"),
            AgentSessionId::new("sess-1"),
            AgentMode::Agent,
        );
        assert_eq!(ctx.workspace_scope().as_str(), "ws-1");
        assert_eq!(ctx.agent_session_id().as_str(), "sess-1");
        assert_eq!(ctx.mode(), AgentMode::Agent);
        assert!(ctx.correlation_id().is_none());
        assert_eq!(ctx.actor().as_str(), "agent");
        assert!(ctx.permissions().is_empty());
    }

    #[test]
    fn context_builder_with_correlation_actor_and_permissions() {
        let ctx = ToolContext::new(
            WorkspaceScope::new("ws-1"),
            AgentSessionId::new("sess-1"),
            AgentMode::Build,
        )
        .with_correlation_id(CorrelationId::new("corr-1"))
        .with_actor(Actor::new("user"))
        .with_permissions(PermissionSet::from_iter([
            crate::permissions::ToolPermission::new("workflow.read"),
        ]));

        assert_eq!(ctx.correlation_id().unwrap().as_str(), "corr-1");
        assert_eq!(ctx.actor().as_str(), "user");
        assert_eq!(ctx.permissions().len(), 1);
        assert!(
            ctx.permissions()
                .contains(&crate::permissions::ToolPermission::new("workflow.read"))
        );
    }
}
