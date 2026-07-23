//! TCP listener and server bootstrap for the Axum host.
//!
//! `server.rs` is the only place that runs the listener. The host
//! adapter that calls in is responsible for building the
//! [`AxumHostState`]; we deliberately do not own `WorkspaceHost`
//! construction so that the same workspace can be shared with the
//! Tauri host without duplicating service composition.

use std::net::SocketAddr;

use axum::serve;
use tokio::net::TcpListener;

use crate::router::build_router;
use crate::state::AxumHostState;

/// Handle for a running Axum server. Drop the handle to stop
/// accepting new connections; the background task drains in-flight
/// requests before returning.
pub struct AxumServerHandle {
    local_addr: SocketAddr,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    join: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
}

impl std::fmt::Debug for AxumServerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AxumServerHandle")
            .field("local_addr", &self.local_addr)
            .finish_non_exhaustive()
    }
}

impl AxumServerHandle {
    /// The local address the server is bound to. Useful in tests that
    /// pick a free port via `127.0.0.1:0`.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Trigger a graceful shutdown and wait for the server task to
    /// exit. Calling this more than once is a no-op.
    pub async fn shutdown(mut self) -> std::io::Result<()> {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            join.await
                .map_err(|err| std::io::Error::other(format!("server join failed: {err}")))?
        } else {
            Ok(())
        }
    }
}

/// Bind a listener on `addr`, build the V1 router, and serve until
/// the process is interrupted. Returns the bound [`AxumServerHandle`].
pub async fn run_server(
    state: AxumHostState,
    addr: SocketAddr,
) -> std::io::Result<AxumServerHandle> {
    let listener = TcpListener::bind(addr).await?;
    run_server_with_listener(state, listener).await
}

/// Like [`run_server`] but accepts a pre-bound listener. Used by
/// tests that want to bind `127.0.0.1:0` and discover the picked port.
pub async fn run_server_with_listener(
    state: AxumHostState,
    listener: TcpListener,
) -> std::io::Result<AxumServerHandle> {
    let local_addr = listener.local_addr()?;
    let app = build_router().with_state(state);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    Ok(AxumServerHandle {
        local_addr,
        shutdown: Some(shutdown_tx),
        join: Some(join),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use reimagine_agent::WorkspaceScope;
    use reimagine_app_host::WorkspaceHost;
    use reimagine_runtime::RunEventSink;

    use crate::recorder::RunEventRecorder;
    use crate::state::AxumHostState;

    #[tokio::test]
    async fn server_starts_and_serves_health() {
        let base = unique_temp_dir("server");
        let host = WorkspaceHost::with_defaults(WorkspaceScope::new("ws-server"), &base);
        let recorder = Arc::new(RunEventRecorder::new());
        // Sanity: the recorder is a valid `RunEventSink`.
        let _ = <RunEventRecorder as RunEventSink>::emit(
            &recorder,
            reimagine_core::event::RunEvent::new(
                "evt-1",
                reimagine_core::model::RunId::new("run-warmup"),
                reimagine_core::model::WorkflowId::new("wf-warmup"),
                reimagine_core::model::WorkflowVersion::new(1),
                reimagine_core::event::RunEventKind::RunQueued,
                reimagine_core::event::Timestamp::new("2026-06-13T00:00:00Z"),
            ),
        );
        let state = AxumHostState::new(Arc::new(host), recorder);
        let handle = run_server(state, "127.0.0.1:0".parse().unwrap())
            .await
            .expect("server should start");

        let url = format!("http://{}/health", handle.local_addr());
        let response = reqwest_get(&url).await;
        assert_eq!(response.status, 200);
        assert!(response.body.contains("\"status\":\"ok\""));

        handle.shutdown().await.expect("server should stop");
    }

    #[derive(serde::Deserialize)]
    struct Response {
        status: u16,
        body: String,
    }

    async fn reqwest_get(url: &str) -> Response {
        // Use `tokio::net::TcpStream` directly. We avoid pulling in a
        // new HTTP-client crate just for tests.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpStream;

        let url = url.trim_start_matches("http://");
        let (host_port, path) = match url.split_once('/') {
            Some((hp, p)) => (hp, format!("/{p}")),
            None => (url, "/".to_string()),
        };
        let (host, port) = host_port.rsplit_once(':').expect("host:port in url");
        let mut stream = TcpStream::connect((host, port.parse::<u16>().unwrap()))
            .await
            .expect("connect");
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n");
        stream.write_all(request.as_bytes()).await.expect("write");
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.expect("read");
        let raw = String::from_utf8(buf).expect("utf8");
        let status = raw
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let body = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
        Response { status, body }
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-axum-host-{prefix}-{nonce}"))
    }
}
