//! Daemon workspace initialization.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use reimagine_agent::{AgentEventSink, AgentToolRegistry, WorkspaceScope};
use reimagine_app_host::{AgentProviderCatalog, AgentService, AppHostError, WorkspaceHost};
use reimagine_runtime::{BoxedRunEventSink, VecRunEventSink};

/// Errors that can occur while initializing the daemon workspace.
#[derive(Debug)]
pub enum DaemonInitError {
    /// Filesystem-level failure while preparing the workspace.
    Io { path: PathBuf, message: String },
    /// The full bootstrap path failed; the daemon should boot in degraded
    /// mode instead of treating this as fatal.
    Workspace(AppHostError),
}

impl std::fmt::Display for DaemonInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "io error at {}: {message}", path.display()),
            Self::Workspace(error) => write!(f, "workspace bootstrap failed: {error}"),
        }
    }
}

impl std::error::Error for DaemonInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { .. } => None,
            Self::Workspace(error) => Some(error),
        }
    }
}

impl From<AppHostError> for DaemonInitError {
    fn from(error: AppHostError) -> Self {
        Self::Workspace(error)
    }
}

/// Workspace host owned by the agent daemon.
pub struct DaemonWorkspace {
    host: WorkspaceHost,
}

impl DaemonWorkspace {
    /// Initialize the full daemon workspace rooted at `workspace_dir`.
    ///
    /// Uses [`WorkspaceHost::try_with_defaults_and_event_sinks`] to load the
    /// backend config and bootstrap the inference runtime with the installed
    /// worker inventory. When that bootstrap fails (for example no workers
    /// installed, or an invalid `inference_backend.json`), the daemon does
    /// not crash: it falls back to the infallible
    /// [`WorkspaceHost::with_defaults`] path and boots in degraded mode with
    /// an empty executor registry, keeping the injected agent event sink via
    /// [`WorkspaceHost::with_agent_event_sink`]. The run event sink is
    /// discarded in degraded mode.
    pub async fn initialize(
        workspace_dir: &Path,
        agent_event_sink: Arc<dyn AgentEventSink>,
    ) -> Result<Self, DaemonInitError> {
        let workspace_scope = workspace_scope_for(workspace_dir);
        let event_sink: BoxedRunEventSink = Arc::new(VecRunEventSink::new());
        match WorkspaceHost::try_with_defaults_and_event_sinks(
            workspace_scope.clone(),
            workspace_dir,
            event_sink,
            agent_event_sink.clone(),
        )
        .await
        {
            Ok(host) => Ok(Self { host }),
            Err(error) => {
                tracing::warn!(
                    path = %workspace_dir.display(),
                    %error,
                    "workspace bootstrap failed; booting in degraded mode"
                );
                let host = WorkspaceHost::with_defaults(workspace_scope, workspace_dir)
                    .with_agent_event_sink(agent_event_sink);
                Ok(Self { host })
            }
        }
    }

    /// The agent service owned by the workspace.
    pub fn agent_service(&self) -> &Arc<AgentService> {
        self.host.agent_service()
    }

    /// The tool registry exposed by the agent service.
    pub fn registry(&self) -> &Arc<AgentToolRegistry> {
        self.agent_service().registry()
    }

    /// The provider catalog exposed by the agent service.
    pub fn providers(&self) -> &AgentProviderCatalog {
        self.agent_service().providers()
    }

    /// The underlying workspace host.
    pub fn host(&self) -> &WorkspaceHost {
        &self.host
    }
}

/// Derive the workspace scope from the workspace directory name, falling
/// back to a fixed daemon scope when the path has no usable file name.
fn workspace_scope_for(workspace_dir: &Path) -> WorkspaceScope {
    workspace_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(WorkspaceScope::new)
        .unwrap_or_else(|| WorkspaceScope::new("agent-daemon"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent::VecAgentEventSink;
    use std::path::PathBuf;

    fn temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-agent-daemon-init-{prefix}-{nonce}"))
    }

    #[tokio::test]
    async fn initialize_exposes_agent_service() {
        let base = temp_dir("agent-service");
        let agent_sink: Arc<dyn AgentEventSink> = Arc::new(VecAgentEventSink::new());
        let workspace = DaemonWorkspace::initialize(&base, agent_sink)
            .await
            .expect("initialize should succeed on the full or degraded path");

        assert!(
            !workspace.registry().is_empty(),
            "registry should carry app tools"
        );
        assert!(workspace.providers().is_empty(), "no providers by default");
        assert_eq!(
            workspace.agent_service().workspace_scope().as_str(),
            base.file_name().and_then(|name| name.to_str()).unwrap(),
            "scope should derive from the workspace dir name"
        );

        let _ = std::fs::remove_dir_all(&base);
    }
}
