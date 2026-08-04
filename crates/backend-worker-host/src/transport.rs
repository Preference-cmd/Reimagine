use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use reimagine_backend_worker_protocol::{
    TransportDescription, TransportError, TransportKind, WorkerTransport,
};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

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
