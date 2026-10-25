//! Local worker process transport.
//!
//! Spawns a worker as a child process, hands it a private listen
//! endpoint and a per-launch bearer token, waits for its READY line on
//! stdout, and returns a connected gRPC client. The child's stdin stays
//! open for the worker's whole lifetime as the lifeline: the worker
//! exits when the host dies and stdin reaches EOF. Dropping the process
//! handle kills the child and removes the private socket directory.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reimagine_backend_worker_protocol::{TransportDescription, TransportError, TransportKind};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};
use tokio::sync::Mutex;
use tokio::time::timeout;

use crate::launch::WorkerLaunchSpec;

pub use reimagine_backend_worker_transport_grpc::auth::{GrpcAuth, TOKEN_ENV};
pub use reimagine_backend_worker_transport_grpc::local::LocalEndpoint;
use reimagine_backend_worker_transport_grpc::{client::WorkerClient, local::connect_local};

/// How long a single connect attempt waits for the named pipe on
/// Windows before surfacing the busy state as a failed startup.
#[cfg(windows)]
const PIPE_CONNECT_RETRIES: usize = 50;

/// Lifecycle handle for a launched worker.
///
/// The supervisor kills the worker through this handle on shutdown
/// timeout; dropping the last handle kills the child and cleans up the
/// private endpoint.
#[async_trait]
pub trait WorkerLifecycle: Send + Sync + 'static {
    /// Descriptive metadata about this transport.
    fn description(&self) -> TransportDescription;

    /// Force-kill the worker process and release its endpoint.
    async fn shutdown(&self) -> Result<(), TransportError>;
}

/// A connected worker: lifecycle handle plus an authenticated client.
pub struct WorkerConnection {
    /// Lifecycle handle for the connected transport.
    pub transport: Arc<dyn WorkerLifecycle>,
    /// Client speaking the v2 contract with the launch token attached.
    pub client: WorkerClient,
    /// Optional stderr stream (local process transports have one).
    pub stderr: Option<Box<dyn tokio::io::AsyncRead + Send + Unpin>>,
}

/// Resolves a worker launch spec into a connected transport.
///
/// `WorkerSupervisor` is transport-agnostic and only observes
/// [`WorkerConnection`]; the default factory spawns local processes.
#[async_trait]
pub trait WorkerTransportFactory: Send + Sync + 'static {
    /// Connect to a worker described by `spec`.
    async fn connect(&self, spec: &WorkerLaunchSpec) -> Result<WorkerConnection, TransportError>;
}

/// Factory that launches local worker processes over private sockets.
#[derive(Clone, Copy, Default)]
pub struct LocalProcessTransportFactory;

#[async_trait]
impl WorkerTransportFactory for LocalProcessTransportFactory {
    async fn connect(&self, spec: &WorkerLaunchSpec) -> Result<WorkerConnection, TransportError> {
        let mut launch = spawn_local(spec)?;
        let mut stdout = launch
            .stdout
            .take()
            .ok_or_else(|| TransportError::ConnectionFailed("stdout already consumed".into()))?;
        wait_for_ready(&mut stdout, &launch.endpoint, spec.limits.startup_timeout).await?;
        // stdout carries only the READY line; release our read end.
        drop(stdout);
        let auth = GrpcAuth {
            token: Some(launch.token.clone()),
            tls: None,
        };
        let client = connect_local(&launch.endpoint, &auth)
            .await
            .map_err(|status| TransportError::ConnectionFailed(status.to_string()))?;
        Ok(WorkerConnection {
            transport: Arc::new(launch.process),
            client,
            stderr: Some(Box::new(launch.stderr)),
        })
    }
}

/// A freshly spawned local worker plus the streams the host keeps.
pub struct LocalLaunch {
    pub process: LocalWorkerProcess,
    /// Worker stdout; carries only the READY line and is released right
    /// after it is consumed.
    pub stdout: Option<ChildStdout>,
    pub stderr: ChildStderr,
    pub endpoint: LocalEndpoint,
    pub token: String,
}

/// Handle over a spawned local worker process.
///
/// The child is killed when this handle is dropped (including on
/// supervisor drop), and the private socket directory is removed.
pub struct LocalWorkerProcess {
    child: Mutex<Child>,
    executable: PathBuf,
    endpoint: LocalEndpoint,
    /// Directory holding the worker's socket; removed on drop.
    socket_dir: Option<PathBuf>,
    /// The worker's stdin write end. Never written to: it exists to
    /// stay open for the worker's lifetime (the lifeline that makes the
    /// worker exit when the host dies) and to close when this handle
    /// drops.
    _stdin: ChildStdin,
}

#[async_trait]
impl WorkerLifecycle for LocalWorkerProcess {
    fn description(&self) -> TransportDescription {
        TransportDescription {
            kind: TransportKind::Local,
            endpoint: self.endpoint.to_uri(),
        }
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        let mut child = self.child.lock().await;
        let _ = child.start_kill();
        let _ = child.wait().await;
        drop(child);
        if let Some(dir) = &self.socket_dir {
            let _ = std::fs::remove_dir_all(dir);
        }
        Ok(())
    }
}

impl Drop for LocalWorkerProcess {
    fn drop(&mut self) {
        // The `child` field drops right after this method (kill_on_drop
        // fires), then the stdin write end closes; remove the private
        // directory now so an explicit drop cleans up immediately.
        if let Some(dir) = self.socket_dir.take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

impl std::fmt::Debug for LocalWorkerProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalWorkerProcess")
            .field("executable", &self.executable)
            .finish()
    }
}

/// Spawn a local worker with a private endpoint and token.
///
/// The worker is started with `--listen <endpoint>` and
/// `REIMAGINE_WORKER_TOKEN`; its environment is otherwise empty so no
/// host state leaks into the worker.
pub fn spawn_local(spec: &WorkerLaunchSpec) -> Result<LocalLaunch, TransportError> {
    let token = generate_launch_token();
    let (endpoint, socket_dir) = private_endpoint(&token)?;

    let mut command = tokio::process::Command::new(&spec.executable);
    command
        .env_clear()
        .envs(spec.environment.iter().cloned())
        .env(TOKEN_ENV, &token)
        .arg("--listen")
        .arg(endpoint.to_uri())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);

    let mut child = command.spawn().map_err(|error| {
        TransportError::ConnectionFailed(format!(
            "failed to spawn `{}`: {error}",
            spec.executable.display()
        ))
    })?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| TransportError::ConnectionFailed("child stdin not piped".into()))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| TransportError::ConnectionFailed("child stdout not piped".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| TransportError::ConnectionFailed("child stderr not piped".into()))?;

    let process = LocalWorkerProcess {
        child: Mutex::new(child),
        executable: spec.executable.clone(),
        endpoint: endpoint.clone(),
        socket_dir,
        _stdin: stdin,
    };

    Ok(LocalLaunch {
        process,
        stdout: Some(stdout),
        stderr,
        endpoint,
        token,
    })
}

/// Read the worker's stdout until its READY line, which must echo the
/// endpoint the host passed on the command line.
///
/// `deadline` is a backstop for direct factory use; the supervisor
/// bounds the whole startup with the same budget so its own timeout
/// fires first, and this wait only trips when no supervisor is
/// watching.
pub async fn wait_for_ready(
    stdout: &mut (impl tokio::io::AsyncRead + Unpin),
    expected: &LocalEndpoint,
    deadline: Duration,
) -> Result<(), TransportError> {
    // Give the supervisor's outer startup timeout room to fire first
    // when both budgets come from the same spec.
    let deadline = deadline.saturating_add(Duration::from_secs(1));
    let expected_uri = expected.to_uri();
    let wait = async {
        let mut reader = tokio::io::BufReader::new(stdout);
        let mut line = Vec::new();
        loop {
            line.clear();
            use tokio::io::AsyncBufReadExt;
            let count = reader
                .read_until(b'\n', &mut line)
                .await
                .map_err(|error| TransportError::Io(format!("reading worker stdout: {error}")))?;
            if count == 0 {
                return Err(TransportError::ConnectionFailed(
                    "worker exited before printing its READY line".to_owned(),
                ));
            }
            let text = String::from_utf8_lossy(&line);
            let trimmed = text.trim_end_matches(['\r', '\n']);
            if let Some(echoed) = trimmed.strip_prefix("READY ") {
                if echoed == expected_uri {
                    return Ok(());
                }
                return Err(TransportError::ConnectionFailed(format!(
                    "worker reported READY for `{echoed}` but was launched for `{expected_uri}`"
                )));
            }
            if trimmed.is_empty() {
                continue;
            }
            return Err(TransportError::ConnectionFailed(format!(
                "worker wrote `{trimmed}` on stdout instead of a READY line"
            )));
        }
    };
    timeout(deadline, wait)
        .await
        .map_err(|_| TransportError::Timeout)?
}

/// Generate a fresh per-launch bearer token.
///
/// The token protects the worker's private endpoint from other local
/// processes. It is derived from OS-level per-process entropy (ASLR
/// addresses), the process id, and a high-resolution timestamp, hashed
/// with SHA-256 — unpredictable to any process that cannot read our
/// memory, without pulling in an RNG dependency.
fn generate_launch_token() -> String {
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Monotonic per-process launch counter: guarantees uniqueness
    /// within a host even when the clock granularity is coarse.
    static LAUNCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    // Per-process entropy without an RNG dependency: a monotonic launch
    // counter, ASLR addresses (stack and a heap allocation), the
    // process id, and a high-resolution timestamp.
    let heap = Box::new(0_u8);
    let stack = 0_u8;
    let mut hasher = Sha256::new();
    hasher.update(
        LAUNCH_SEQUENCE
            .fetch_add(1, Ordering::Relaxed)
            .to_le_bytes(),
    );
    hasher.update(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0)
            .to_le_bytes(),
    );
    hasher.update(std::process::id().to_le_bytes());
    hasher.update((std::ptr::from_ref(&*heap) as usize).to_le_bytes());
    hasher.update((std::ptr::from_ref(&stack) as usize).to_le_bytes());
    hex::encode(hasher.finalize())
}

/// Choose the private endpoint for a launch: a socket inside a fresh
/// 0700 directory (Unix) or a uniquely named pipe (Windows).
fn private_endpoint(token: &str) -> Result<(LocalEndpoint, Option<PathBuf>), TransportError> {
    let suffix = token.get(..16).unwrap_or(token);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("reimagine-worker-{suffix}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).map_err(|error| {
            TransportError::ConnectionFailed(format!(
                "cannot create worker socket directory {}: {error}",
                dir.display()
            ))
        })?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).map_err(
            |error| {
                TransportError::ConnectionFailed(format!(
                    "cannot restrict worker socket directory {}: {error}",
                    dir.display()
                ))
            },
        )?;
        Ok((
            LocalEndpoint::UnixSocket(dir.join("worker.sock")),
            Some(dir),
        ))
    }
    #[cfg(windows)]
    {
        let _ = std::fs::remove_dir_all;
        Ok((
            LocalEndpoint::NamedPipe(format!(r"\\.\pipe\reimagine-worker-{suffix}")),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{generate_launch_token, private_endpoint, wait_for_ready};
    use crate::launch::{WorkerLaunchSpec, WorkerLimits};
    use std::time::Duration;

    fn minimal_spec() -> WorkerLaunchSpec {
        WorkerLaunchSpec {
            executable: std::path::PathBuf::from("unused"),
            expected: crate::ExpectedWorkerIdentity {
                backend_instance_id: reimagine_backend_worker_protocol::BackendInstanceId::from(
                    "fake:cpu:default",
                ),
                installation_id: reimagine_backend_worker_protocol::WorkerInstallationId::from(
                    "fake-installation",
                ),
                backend_kind: "fake".to_owned(),
                target: std::env::consts::ARCH.to_owned(),
                manifest_digest: "test-manifest".to_owned(),
            },
            supported_protocols: reimagine_backend_worker_protocol::ProtocolRange::new(2, 2),
            limits: WorkerLimits::default(),
            environment: Vec::new(),
        }
    }

    #[test]
    fn launch_tokens_are_unique_and_long() {
        let first = generate_launch_token();
        let second = generate_launch_token();
        assert_ne!(first, second);
        assert_eq!(first.len(), 64);
    }

    #[tokio::test]
    async fn ready_wait_accepts_the_matching_endpoint_line() {
        let endpoint = private_endpoint(&generate_launch_token()).unwrap().0;
        // Spawn `cat`-like echo of the READY line through a real process
        // is overkill; drive the parser through a pipe instead.
        let (mut reader, mut writer) = tokio::io::simplex(64);
        let uri = endpoint.to_uri();
        let expected = endpoint;
        let task = tokio::spawn(async move {
            wait_for_ready(&mut reader, &expected, Duration::from_secs(2)).await
        });
        use tokio::io::AsyncWriteExt;
        writer
            .write_all(format!("READY {uri}\n").as_bytes())
            .await
            .unwrap();
        assert!(task.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn ready_wait_rejects_unexpected_stdout() {
        let endpoint = private_endpoint(&generate_launch_token()).unwrap().0;
        let (mut reader, mut writer) = tokio::io::simplex(64);
        let expected = endpoint;
        let task = tokio::spawn(async move {
            wait_for_ready(&mut reader, &expected, Duration::from_secs(2)).await
        });
        use tokio::io::AsyncWriteExt;
        writer.write_all(b"hello there\n").await.unwrap();
        let error = task.await.unwrap().unwrap_err();
        assert!(error.to_string().contains("hello there"));
    }

    #[tokio::test]
    async fn ready_wait_times_out_without_a_line() {
        let endpoint = private_endpoint(&generate_launch_token()).unwrap().0;
        let (mut reader, _writer) = tokio::io::simplex(64);
        let expected = endpoint;
        let error = wait_for_ready(&mut reader, &expected, Duration::from_millis(20))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            reimagine_backend_worker_protocol::TransportError::Timeout
        ));
    }

    #[test]
    fn spec_has_no_transport_field_left() {
        // Compile-time canary that WorkerLaunchSpec carries only the
        // local-launch fields.
        let spec = minimal_spec();
        assert!(spec.environment.is_empty());
    }
}
