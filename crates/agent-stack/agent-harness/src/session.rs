//! Workspace-scoped agent session.
//!
//! V1 keeps `AgentSession` as a small in-memory record that holds the
//! session id, the bound workspace scope, the active mode, the
//! provider, the registry, a start timestamp, the granted permission
//! set, and the running conversation transcript. Concrete
//! orchestration (model selection, tool execution, streaming) lives
//! in `app-host`. The session is `Clone`-friendly because every field
//! is either trivially copyable or wrapped in `Arc`, and the history
//! uses `Arc<Mutex<...>>` so a cloned session shares the same transcript.

use std::sync::{Arc, Mutex};

use crate::ids::{AgentSessionId, ProviderName, WorkspaceScope};
use crate::mode::AgentMode;
use crate::permissions::PermissionSet;
use crate::provider::Message;
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
    /// Permissions granted to this session. The Agent loop copies them
    /// into every `ToolContext` it builds. V1 keeps this in-memory only;
    /// `app-host` is responsible for sourcing the permission set from
    /// the workspace / user / host policy and applying it here.
    permissions: PermissionSet,
    /// Running conversation transcript. V1 keeps this in-memory; the
    /// Agent loop prepends the session history to each turn's message
    /// list and appends the turn's new messages on completion. No
    /// memory / summarization / compaction / external persistence in
    /// V1.
    history: Arc<Mutex<Vec<Message>>>,
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
            .field("permissions", &self.permissions)
            .field("history_len", &self.history.lock().unwrap().len())
            .finish()
    }
}

impl AgentSession {
    /// Build a new session bound to `workspace_scope`. The session
    /// starts with an empty permission set and empty conversation
    /// history; callers use the builder methods to populate them.
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
            permissions: PermissionSet::new(),
            history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Builder-style: attach a `started_at` timestamp string. The
    /// runtime does not read the clock; the host supplies the value.
    pub fn with_started_at(mut self, ts: impl Into<String>) -> Self {
        self.started_at = ts.into();
        self
    }

    /// Builder-style: attach a permission set. Replaces any previously
    /// set permissions.
    pub fn with_permissions(mut self, permissions: PermissionSet) -> Self {
        self.permissions = permissions;
        self
    }

    /// Builder-style: pre-populate the conversation history. Replaces
    /// any previously set history. Used by hosts that hydrate a
    /// session from persistent storage; in normal turn flow the loop
    /// appends via [`AgentSession::append_history`].
    pub fn with_history(self, history: Vec<Message>) -> Self {
        *self.history.lock().unwrap() = history;
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

    pub fn permissions(&self) -> &PermissionSet {
        &self.permissions
    }

    /// Snapshot the running conversation history. The loop reads this
    /// at the start of each turn to seed the provider message list.
    pub fn history(&self) -> Vec<Message> {
        self.history.lock().unwrap().clone()
    }

    /// Number of messages currently in the session history. Useful for
    /// tests and for the loop to compute the slice of new messages to
    /// append after a turn.
    pub fn history_len(&self) -> usize {
        self.history.lock().unwrap().len()
    }

    /// Append `messages` to the session history. The Agent loop calls
    /// this at the end of `run_turn` with the messages produced during
    /// the turn (input + assistant + tool observations + final
    /// assistant). Hosts can also call it to seed conversation
    /// continuity from external storage.
    pub fn append_history(&self, messages: impl IntoIterator<Item = Message>) {
        self.history.lock().unwrap().extend(messages);
    }

    /// Replace the entire history. Used by tests and by hosts that
    /// reset or migrate a session.
    pub fn set_history(&self, messages: Vec<Message>) {
        *self.history.lock().unwrap() = messages;
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

    fn empty_session() -> AgentSession {
        let mut reg = AgentToolRegistry::new();
        reg.register(Noop).unwrap();
        AgentSession::new(
            AgentSessionId::new("sess-1"),
            WorkspaceScope::new("ws-alpha"),
            AgentMode::Agent,
            ProviderName::new("openai"),
            Arc::new(reg),
        )
    }

    #[test]
    fn session_is_workspace_scoped() {
        let sess = empty_session().with_started_at("2026-06-10T00:00:00Z");
        assert_eq!(sess.workspace_scope().as_str(), "ws-alpha");
        assert_eq!(sess.mode(), AgentMode::Agent);
        assert_eq!(sess.provider().as_str(), "openai");
        assert_eq!(sess.started_at(), "2026-06-10T00:00:00Z");
    }

    #[test]
    fn session_history_starts_empty_and_appends() {
        let sess = empty_session();
        assert_eq!(sess.history_len(), 0);
        assert!(sess.history().is_empty());

        sess.append_history([Message::user("hi"), Message::assistant("hello")]);
        assert_eq!(sess.history_len(), 2);
        assert_eq!(sess.history()[0].content(), "hi");
        assert_eq!(sess.history()[1].content(), "hello");
    }

    #[test]
    fn session_history_is_shared_across_clones() {
        let original = empty_session();
        let clone = original.clone();
        original.append_history([Message::user("from-original")]);
        // Cloned session sees the same history.
        assert_eq!(clone.history_len(), 1);
        assert_eq!(clone.history()[0].content(), "from-original");
    }

    #[test]
    fn session_history_can_be_seeded_via_builder_or_setter() {
        let seeded =
            empty_session().with_history(vec![Message::user("prior"), Message::assistant("ok")]);
        assert_eq!(seeded.history_len(), 2);

        let reset = empty_session().with_history(vec![Message::user("a")]);
        reset.set_history(vec![Message::user("b"), Message::user("c")]);
        assert_eq!(reset.history_len(), 2);
        assert_eq!(reset.history()[1].content(), "c");
    }
}
