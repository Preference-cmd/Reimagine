//! JSON-RPC 2.0 client bridge to the `reimagine-agent-daemon` child process.
//!
//! `AgentBridge` spawns the daemon binary on construction, completes the
//! `initialize` → `initialized` handshake, and proxies session / turn /
//! provider requests over newline-delimited JSON on the child's stdio. A
//! dedicated reader task parses stdout: responses are matched to in-flight
//! requests by id, and `agent.*` streaming notifications are routed to the
//! per-turn Tauri channels passed to [`AgentBridge::run_turn`]. The daemon
//! subprocess is killed when the bridge is dropped.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reimagine_agent_daemon::protocol::{
    ClientCapabilities, ClientInfo, EmptyParams, InitializeRequest, InitializeResponse,
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, METHOD_AGENT_TURN_COMPLETED,
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_PROVIDERS_LIST, METHOD_SESSION_CREATE,
    METHOD_SESSION_LIST, METHOD_TURN_CANCEL, METHOD_TURN_RUN, ProvidersListResult,
    SessionCreateParams, SessionCreateResult, SessionInfo, SessionListResult, TurnCancelParams,
    TurnCancelResult, TurnRunParams, TurnRunResult,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tauri::ipc::Channel;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex as AsyncMutex, oneshot};

/// Environment variable that overrides the daemon binary path.
pub const DAEMON_BINARY_ENV: &str = "REIMAGINE_AGENT_DAEMON";
/// Timeout for the `initialize` → `initialized` handshake.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Timeout for a single request/response exchange.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

const INNER_LOCK_POISONED: &str = "agent bridge inner lock poisoned";
const DAEMON_BINARY_NAME: &str = "reimagine-agent-daemon";

/// Errors produced by the agent bridge.
#[derive(Debug)]
pub enum AgentBridgeError {
    /// The `reimagine-agent-daemon` binary could not be located.
    DaemonNotFound { searched: Vec<PathBuf> },
    /// The daemon binary exists but could not be spawned.
    Spawn { binary: PathBuf, source: io::Error },
    /// Writing to the daemon's stdin failed while it was still alive.
    Io {
        operation: &'static str,
        source: io::Error,
    },
    /// The daemon process exited or its stdout closed unexpectedly.
    ///
    /// The daemon-side equivalent of Codex's
    /// `CodexAppServerDisconnectedException`: a request was in flight
    /// when the connection died.
    Disconnected { detail: String },
    /// The daemon answered with a JSON-RPC error.
    Protocol { code: i64, message: String },
    /// The handshake or a request did not complete in time.
    Timeout { detail: String },
}

impl fmt::Display for AgentBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DaemonNotFound { searched } => write!(
                f,
                "agent daemon is not installed; searched: {}",
                searched
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Spawn { binary, source } => {
                write!(
                    f,
                    "failed to spawn agent daemon `{}`: {source}",
                    binary.display()
                )
            }
            Self::Io { operation, source } => {
                write!(f, "agent daemon I/O error while {operation}: {source}")
            }
            Self::Disconnected { detail } => {
                write!(f, "agent daemon connection lost: {detail}")
            }
            Self::Protocol { code, message } => {
                write!(f, "agent daemon error {code}: {message}")
            }
            Self::Timeout { detail } => {
                write!(
                    f,
                    "agent daemon {detail} timed out after {REQUEST_TIMEOUT:?}"
                )
            }
        }
    }
}

impl std::error::Error for AgentBridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn { source, .. } | Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// JSON-RPC client over the daemon child process's stdio.
pub struct AgentBridge {
    child: Option<Child>,
    inner: Arc<BridgeInner>,
    reader: Option<tokio::task::JoinHandle<()>>,
}

struct BridgeInner {
    stdin: AsyncMutex<BufWriter<ChildStdin>>,
    next_id: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Result<Value, AgentBridgeError>>>>,
    subscriptions: Mutex<HashMap<(String, String), Channel<Value>>>,
}

impl fmt::Debug for AgentBridge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentBridge")
            .field("child", &self.child)
            .field(
                "pending",
                &self.inner.pending.lock().expect(INNER_LOCK_POISONED).len(),
            )
            .field(
                "subscriptions",
                &self
                    .inner
                    .subscriptions
                    .lock()
                    .expect(INNER_LOCK_POISONED)
                    .len(),
            )
            .finish()
    }
}

impl AgentBridge {
    /// Spawn the daemon for `workspace_dir` and complete the handshake.
    ///
    /// Binary lookup order:
    /// 1. `REIMAGINE_AGENT_DAEMON` environment variable
    /// 2. `target/{debug,release}/reimagine-agent-daemon` under the
    ///    workspace root (walked up from the crate directory)
    /// 3. `reimagine-agent-daemon` on `PATH`
    pub async fn new(workspace_dir: &Path) -> Result<Self, AgentBridgeError> {
        let binary = locate_daemon_binary()?;
        Self::new_with_binary(workspace_dir, &binary).await
    }

    /// Spawn a specific daemon binary for `workspace_dir` and complete
    /// the handshake. Test entry point that bypasses binary lookup.
    pub async fn new_with_binary(
        workspace_dir: &Path,
        binary: &Path,
    ) -> Result<Self, AgentBridgeError> {
        Self::spawn_with_env(workspace_dir, binary, &[]).await
    }

    async fn spawn_with_env(
        workspace_dir: &Path,
        binary: &Path,
        env: &[(&str, &str)],
    ) -> Result<Self, AgentBridgeError> {
        let mut command = Command::new(binary);
        command
            .arg("--workspace-dir")
            .arg(workspace_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        for (key, value) in env {
            command.env(key, value);
        }
        let mut child = command.spawn().map_err(|source| AgentBridgeError::Spawn {
            binary: binary.to_path_buf(),
            source,
        })?;
        let stdin = child.stdin.take().expect("child stdin is piped");
        let stdout = child.stdout.take().expect("child stdout is piped");

        let inner = Arc::new(BridgeInner {
            stdin: AsyncMutex::new(BufWriter::new(stdin)),
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            subscriptions: Mutex::new(HashMap::new()),
        });

        let mut bridge = Self {
            child: Some(child),
            inner: Arc::clone(&inner),
            reader: None,
        };
        bridge.reader = Some(spawn_stdout_reader(stdout, Arc::clone(&inner)));

        let handshake = async {
            let _response: InitializeResponse = tokio::time::timeout(
                HANDSHAKE_TIMEOUT,
                bridge.request(METHOD_INITIALIZE, initialize_request()),
            )
            .await
            .map_err(|_| AgentBridgeError::Timeout {
                detail: "initialize handshake".into(),
            })??;
            bridge.notify(METHOD_INITIALIZED, EmptyParams {}).await
        };
        if let Err(error) = handshake.await {
            bridge.kill_child();
            return Err(error);
        }
        Ok(bridge)
    }

    /// Create a session on the daemon.
    ///
    /// `mode` is `"agent"` or `"build"`; `provider` must be a provider id
    /// known to the daemon (`providers.list`).
    pub async fn create_session(
        &mut self,
        mode: &str,
        provider: &str,
    ) -> Result<SessionInfo, AgentBridgeError> {
        let created: SessionCreateResult = self
            .request(
                METHOD_SESSION_CREATE,
                SessionCreateParams {
                    mode: mode.to_owned(),
                    provider: provider.to_owned(),
                    system_prompt: None,
                    workspace_dir: None,
                },
            )
            .await?;
        Ok(SessionInfo {
            session_id: created.session_id,
            mode: created.mode,
            provider: created.provider,
            created_at: created.created_at,
        })
    }

    /// Start a turn and stream `agent.*` notifications to `channel`.
    ///
    /// The channel receives each daemon notification for this session and
    /// turn as the raw JSON-RPC envelope (`{ "jsonrpc", "method",
    /// "params" }`), ending with `agent.turn_completed`, which
    /// unsubscribes the channel. The returned `TurnRunResult` is the
    /// daemon's acceptance of the turn; the outcome streams afterwards.
    pub async fn run_turn(
        &mut self,
        request: TurnRunParams,
        channel: Channel<Value>,
    ) -> Result<TurnRunResult, AgentBridgeError> {
        let key = (request.session_id.clone(), request.turn_id.clone());
        self.inner
            .subscriptions
            .lock()
            .expect(INNER_LOCK_POISONED)
            .insert(key.clone(), channel);
        let result = self.request(METHOD_TURN_RUN, request).await;
        if result.is_err() {
            self.inner
                .subscriptions
                .lock()
                .expect(INNER_LOCK_POISONED)
                .remove(&key);
        }
        result
    }

    /// Cancel a running turn and drop its notification subscription.
    pub async fn cancel_turn(
        &mut self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<TurnCancelResult, AgentBridgeError> {
        let key = (session_id.to_owned(), turn_id.to_owned());
        let result = self
            .request(
                METHOD_TURN_CANCEL,
                TurnCancelParams {
                    session_id: session_id.to_owned(),
                    turn_id: turn_id.to_owned(),
                },
            )
            .await;
        self.inner
            .subscriptions
            .lock()
            .expect(INNER_LOCK_POISONED)
            .remove(&key);
        result
    }

    /// List provider ids known to the daemon.
    pub async fn list_providers(&mut self) -> Result<Vec<String>, AgentBridgeError> {
        let result: ProvidersListResult =
            self.request(METHOD_PROVIDERS_LIST, EmptyParams {}).await?;
        Ok(result
            .providers
            .into_iter()
            .map(|provider| provider.id)
            .collect())
    }

    /// List live sessions on the daemon.
    pub async fn list_sessions(&mut self) -> Result<Vec<SessionInfo>, AgentBridgeError> {
        let result: SessionListResult = self.request(METHOD_SESSION_LIST, EmptyParams {}).await?;
        Ok(result.sessions)
    }

    /// Send a request and await the matching response.
    async fn request<T: Serialize, R: DeserializeOwned>(
        &mut self,
        method: &str,
        params: T,
    ) -> Result<R, AgentBridgeError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.inner
            .pending
            .lock()
            .expect(INNER_LOCK_POISONED)
            .insert(id, tx);

        let line = serde_json::to_string(&JsonRpcRequest::new(method, id, params))
            .expect("request serializes");
        if let Err(error) = self.write_line(&line).await {
            self.inner
                .pending
                .lock()
                .expect(INNER_LOCK_POISONED)
                .remove(&id);
            return Err(error);
        }

        match tokio::time::timeout(REQUEST_TIMEOUT, rx).await {
            Ok(Ok(Ok(result))) => {
                serde_json::from_value(result).map_err(|error| AgentBridgeError::Protocol {
                    code: 0,
                    message: format!("malformed result for method `{method}`: {error}"),
                })
            }
            Ok(Ok(Err(error))) => Err(error),
            Ok(Err(_)) => Err(AgentBridgeError::Disconnected {
                detail: format!("the daemon dropped the request for method `{method}`"),
            }),
            Err(_) => {
                self.inner
                    .pending
                    .lock()
                    .expect(INNER_LOCK_POISONED)
                    .remove(&id);
                Err(AgentBridgeError::Timeout {
                    detail: format!("no response for method `{method}` (id {id})"),
                })
            }
        }
    }

    /// Send a notification (no id) to the daemon.
    async fn notify<T: Serialize>(
        &mut self,
        method: &str,
        params: T,
    ) -> Result<(), AgentBridgeError> {
        let line = serde_json::to_string(&JsonRpcNotification::new(method, params))
            .expect("notification serializes");
        self.write_line(&line).await
    }

    /// Write one JSON line to the daemon's stdin.
    ///
    /// A write failure against an already-exited child is reported as
    /// [`AgentBridgeError::Disconnected`]; other failures are [`Self::Io`].
    async fn write_line(&mut self, line: &str) -> Result<(), AgentBridgeError> {
        let mut payload = line.to_owned();
        payload.push('\n');
        let write_result = {
            let mut stdin = self.inner.stdin.lock().await;
            match stdin.write_all(payload.as_bytes()).await {
                Ok(()) => stdin.flush().await,
                Err(error) => Err(error),
            }
        };
        match write_result {
            Ok(()) => Ok(()),
            Err(source) => {
                if let Some(child) = self.child.as_mut()
                    && child.try_wait().ok().flatten().is_some()
                {
                    return Err(AgentBridgeError::Disconnected {
                        detail: format!("the agent daemon process exited: {source}"),
                    });
                }
                Err(AgentBridgeError::Io {
                    operation: "write daemon stdin",
                    source,
                })
            }
        }
    }

    /// Kill the daemon subprocess and reap it in the background.
    fn kill_child(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let _ = child.wait().await;
                });
            }
        }
    }
}

impl Drop for AgentBridge {
    /// Kill the daemon subprocess and unblock any in-flight requests.
    fn drop(&mut self) {
        if let Some(reader) = self.reader.take() {
            reader.abort();
        }
        self.kill_child();
        self.inner
            .fail_all_pending(|| AgentBridgeError::Disconnected {
                detail: "the agent bridge was dropped".into(),
            });
    }
}

impl BridgeInner {
    fn route_response(&self, message: Value) {
        let Some(id) = message.get("id").and_then(Value::as_u64) else {
            return;
        };
        let Ok(response) = serde_json::from_value::<JsonRpcResponse<Value>>(message) else {
            return;
        };
        let outcome = match response.error {
            Some(error) => Err(AgentBridgeError::Protocol {
                code: error.code,
                message: error.message,
            }),
            None => Ok(response.result.unwrap_or(Value::Null)),
        };
        let tx = self.pending.lock().expect(INNER_LOCK_POISONED).remove(&id);
        if let Some(tx) = tx {
            let _ = tx.send(outcome);
        }
    }

    /// Forward a daemon notification to every matching subscription.
    ///
    /// Turn-scoped notifications (`agent.*` carrying `turn_id`) go to the
    /// channel registered for that session+turn; session-scoped ones go to
    /// every channel of the session. `agent.turn_completed` unsubscribes
    /// the channel after delivery. Dead channels are dropped silently.
    fn route_notification(&self, method: &str, message: Value) {
        let Some(session_id) = message
            .pointer("/params/session_id")
            .and_then(Value::as_str)
        else {
            return;
        };
        let turn_id = message.pointer("/params/turn_id").and_then(Value::as_str);

        let mut subscriptions = self.subscriptions.lock().expect(INNER_LOCK_POISONED);
        for (session, turn) in subscriptions.keys().cloned().collect::<Vec<_>>() {
            if session != session_id {
                continue;
            }
            if let Some(want) = turn_id
                && turn != want
            {
                continue;
            }
            let key = (session, turn);
            let Some(channel) = subscriptions.get(&key) else {
                continue;
            };
            if channel.send(message.clone()).is_err() {
                subscriptions.remove(&key);
                continue;
            }
            if method == METHOD_AGENT_TURN_COMPLETED {
                subscriptions.remove(&key);
            }
        }
    }

    /// Fail every in-flight request (called when the connection dies).
    fn fail_all_pending(&self, error: impl Fn() -> AgentBridgeError) {
        let pending = std::mem::take(&mut *self.pending.lock().expect(INNER_LOCK_POISONED));
        for (_, tx) in pending {
            let _ = tx.send(Err(error()));
        }
    }
}

/// Spawn the task that reads the daemon's stdout and routes messages.
///
/// Lines that do not parse as JSON (e.g. daemon log output) are skipped so
/// the protocol stream is tolerant of interleaved diagnostics. EOF or a
/// read error fails every in-flight request with `Disconnected`.
fn spawn_stdout_reader(
    stdout: tokio::process::ChildStdout,
    inner: Arc<BridgeInner>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        loop {
            let line = match lines.next_line().await {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(_) => break,
            };
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if message.get("id").is_some() {
                inner.route_response(message);
            } else if let Some(method) = message
                .get("method")
                .and_then(Value::as_str)
                .map(str::to_owned)
            {
                inner.route_notification(&method, message);
            }
        }
        inner.fail_all_pending(|| AgentBridgeError::Disconnected {
            detail: "the agent daemon process exited or its stdout closed".into(),
        });
    })
}

/// Locate the daemon binary in order: env override, workspace target
/// directories, then `PATH`.
fn locate_daemon_binary() -> Result<PathBuf, AgentBridgeError> {
    if let Some(path) = std::env::var_os(DAEMON_BINARY_ENV) {
        let path = PathBuf::from(path);
        if daemon_file(&path) {
            return Ok(path);
        }
        return Err(AgentBridgeError::DaemonNotFound {
            searched: vec![path],
        });
    }

    let mut searched = Vec::new();
    for candidate in target_candidates() {
        searched.push(candidate.clone());
        if daemon_file(&candidate) {
            return Ok(candidate);
        }
    }
    if let Some(path) = path_lookup() {
        return Ok(path);
    }
    Err(AgentBridgeError::DaemonNotFound { searched })
}

/// `target/{debug,release}/reimagine-agent-daemon` candidates walked up
/// from the crate directory.
fn target_candidates() -> Vec<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut candidates = Vec::new();
    let mut dir = Some(manifest_dir.as_path());
    while let Some(current) = dir {
        for profile in ["debug", "release"] {
            candidates.push(
                current
                    .join("target")
                    .join(profile)
                    .join(daemon_binary_name()),
            );
        }
        dir = current.parent();
    }
    candidates
}

fn daemon_binary_name() -> &'static str {
    if cfg!(windows) {
        "reimagine-agent-daemon.exe"
    } else {
        DAEMON_BINARY_NAME
    }
}

fn daemon_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Find `reimagine-agent-daemon` on `PATH`.
fn path_lookup() -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(daemon_binary_name()))
        .find(|candidate| daemon_file(candidate))
}

/// Build the `initialize` request params for this host.
fn initialize_request() -> InitializeRequest {
    InitializeRequest {
        client_info: ClientInfo {
            name: "reimagine-tauri".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
        },
        capabilities: ClientCapabilities {
            experimental_api: Some(true),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent_daemon::protocol::{
        METHOD_AGENT_CONTENT_DELTA, TurnCancelStatus, TurnRunStatus,
    };
    use serde_json::json;
    use std::sync::mpsc;

    /// Contract with the mock daemon: with this env var set the mock
    /// serves the scripted protocol; with `MOCK_EXIT_AFTER_INIT_ENV` it
    /// exits right after the `initialized` notification (see
    /// `tests/mock_daemon.rs`).
    const MOCK_SERVE_ENV: &str = "REIMAGINE_MOCK_DAEMON_SERVE";
    const MOCK_EXIT_AFTER_INIT_ENV: &str = "REIMAGINE_MOCK_DAEMON_EXIT_AFTER_INIT";

    fn mock_env(extra: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
        let mut env: Vec<(&'static str, &'static str)> = vec![(MOCK_SERVE_ENV, "1")];
        env.extend(extra.iter().copied());
        env
    }

    fn mock_daemon_path() -> PathBuf {
        let deps = std::env::current_exe()
            .expect("test binary path")
            .parent()
            .expect("test deps dir")
            .to_path_buf();
        let mut matches: Vec<PathBuf> = std::fs::read_dir(&deps)
            .expect("read deps dir")
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("");
                name.starts_with("mock_daemon-") && !name.contains('.') && daemon_file(path)
            })
            .collect();
        matches.sort();
        matches.pop().unwrap_or_else(|| {
            panic!(
                "mock daemon binary not found in `{}`; run `cargo test -p reimagine` \
                 (all test targets) so the `mock_daemon` target is built",
                deps.display()
            )
        })
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn handshake_roundtrip_and_notification_streaming() {
        let workspace = tempfile::tempdir().expect("workspace tempdir");
        let mut bridge =
            AgentBridge::spawn_with_env(workspace.path(), &mock_daemon_path(), &mock_env(&[]))
                .await
                .expect("handshake completes");

        let session = bridge
            .create_session("agent", "openai")
            .await
            .expect("session.create roundtrip");
        assert_eq!(session.provider, "openai");

        let sessions = bridge
            .list_sessions()
            .await
            .expect("session.list roundtrip");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, session.session_id);

        let providers = bridge
            .list_providers()
            .await
            .expect("providers.list roundtrip");
        assert_eq!(providers, ["openai", "anthropic"]);

        let (tx, rx) = mpsc::channel();
        let channel = Channel::<Value>::new(move |payload| {
            if let Ok(value) = payload.deserialize::<Value>() {
                tx.send(value).ok();
            }
            Ok(())
        });
        let accepted = bridge
            .run_turn(
                TurnRunParams {
                    session_id: session.session_id.clone(),
                    turn_id: "turn-1".into(),
                    model: "gpt-4o-mini".into(),
                    input: json!({"text": "hello"}),
                },
                channel,
            )
            .await
            .expect("turn.run roundtrip");
        assert_eq!(accepted.status, TurnRunStatus::Accepted);

        let delta = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("content_delta streamed");
        assert_eq!(delta["method"], METHOD_AGENT_CONTENT_DELTA);
        assert_eq!(
            delta["params"]["session_id"].as_str(),
            Some(session.session_id.as_str())
        );

        let completed = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("turn_completed streamed");
        assert_eq!(completed["method"], METHOD_AGENT_TURN_COMPLETED);

        let cancelled = bridge
            .cancel_turn(&session.session_id, "turn-1")
            .await
            .expect("turn.cancel roundtrip");
        assert_eq!(cancelled.status, TurnCancelStatus::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn new_reports_daemon_not_found() {
        unsafe {
            std::env::set_var(DAEMON_BINARY_ENV, "/nonexistent/reimagine-agent-daemon");
        }
        let workspace = tempfile::tempdir().expect("workspace tempdir");
        let Err(AgentBridgeError::DaemonNotFound { .. }) = AgentBridge::new(workspace.path()).await
        else {
            panic!("expected DaemonNotFound");
        };
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn daemon_exit_fails_pending_requests_with_disconnected() {
        let workspace = tempfile::tempdir().expect("workspace tempdir");
        let mut bridge = AgentBridge::spawn_with_env(
            workspace.path(),
            &mock_daemon_path(),
            &mock_env(&[(MOCK_EXIT_AFTER_INIT_ENV, "1")]),
        )
        .await
        .expect("handshake completes before the mock exits");

        let Err(AgentBridgeError::Disconnected { .. }) =
            bridge.create_session("agent", "openai").await
        else {
            panic!("expected Disconnected after daemon exit");
        };
    }
}
