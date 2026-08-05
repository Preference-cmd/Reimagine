use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use reimagine_backend_worker_protocol::{
    TransportDescription, TransportError, TransportKind, WorkerTransport,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::launch::WorkerLaunchSpec;

/// A connected worker transport with its protocol stream halves.
///
/// The supervisor performs the worker handshake and session handling
/// over the generic halves; the transport handle owns the connection
/// lifecycle (process kill for stdio, connection close for network
/// transports).
pub struct WorkerConnection {
    /// Lifecycle handle for the connected transport.
    pub transport: Arc<dyn WorkerTransport>,
    /// Read half for worker protocol frames.
    pub reader: Box<dyn AsyncRead + Send + Unpin>,
    /// Write half for worker protocol frames.
    pub writer: Box<dyn AsyncWrite + Send + Unpin>,
    /// Optional stderr stream (only stdio transports have one).
    pub stderr: Option<Box<dyn AsyncRead + Send + Unpin>>,
}

/// Resolves a worker launch spec into a connected transport.
///
/// Each transport kind (stdio process spawn, QUIC/gRPC connect)
/// implements this seam; `WorkerSupervisor` is transport-agnostic
/// and only observes [`WorkerConnection`].
#[async_trait]
pub trait WorkerTransportFactory: Send + Sync + 'static {
    /// Connect to a worker described by `spec`.
    async fn connect(&self, spec: &WorkerLaunchSpec) -> Result<WorkerConnection, TransportError>;
}

/// Stdio-backed factory that spawns local worker processes.
#[derive(Clone, Copy, Default)]
pub struct StdioTransportFactory;

#[async_trait]
impl WorkerTransportFactory for StdioTransportFactory {
    async fn connect(&self, spec: &WorkerLaunchSpec) -> Result<WorkerConnection, TransportError> {
        let (transport, stdin, stdout, stderr) =
            StdioTransport::spawn(&spec.executable, &spec.environment).await?;
        Ok(WorkerConnection {
            transport: Arc::new(transport),
            reader: Box::new(stdout),
            writer: Box::new(stdin),
            stderr: Some(Box::new(stderr)),
        })
    }
}

/// Stdio-based transport for local worker processes.
///
/// Wraps a `tokio::process::Child` and manages its lifecycle.
/// The child's stdin/stdout are extracted and used as the read/write
/// halves of the transport. The child handle is kept for force-kill
/// on shutdown.
pub struct StdioTransport {
    child: Arc<Mutex<Child>>,
}

impl StdioTransport {
    /// Spawn a child process and return the transport plus the
    /// read/write halves for protocol communication.
    ///
    /// The returned transport holds the child handle for lifecycle
    /// management. The caller receives the stdin/stdout halves to
    /// use with the worker protocol (FrameCodec).
    pub async fn spawn(
        executable: &PathBuf,
        environment: &[(String, String)],
    ) -> Result<
        (
            Self,
            tokio::process::ChildStdin,
            tokio::process::ChildStdout,
            tokio::process::ChildStderr,
        ),
        TransportError,
    > {
        let mut command = Command::new(executable);
        command
            .env_clear()
            .envs(environment.iter().cloned())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = command
            .spawn()
            .map_err(|e| TransportError::ConnectionFailed(format!("failed to spawn: {e}")))?;

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

        let transport = Self {
            child: Arc::new(Mutex::new(child)),
        };

        Ok((transport, stdin, stdout, stderr))
    }

    /// Force-kill the child process.
    pub async fn force_kill(&self) -> Result<(), TransportError> {
        let mut child = self.child.lock().await;
        let _ = child.start_kill();
        let _ = child.wait().await;
        Ok(())
    }
}

#[async_trait]
impl WorkerTransport for StdioTransport {
    fn description(&self) -> TransportDescription {
        TransportDescription {
            kind: TransportKind::Stdio,
            endpoint: "child process stdio".to_owned(),
        }
    }

    async fn shutdown(&self) -> Result<(), TransportError> {
        self.force_kill().await
    }
}
