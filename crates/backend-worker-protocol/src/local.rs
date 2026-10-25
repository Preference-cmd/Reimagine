//! Local transport plumbing: Unix domain sockets and Windows named
//! pipes.
//!
//! The host chooses a private endpoint for every worker launch:
//! a freshly created socket path inside a 0700 directory (Unix) or a
//! randomly named pipe (Windows). The worker binds that endpoint,
//! prints a single `READY <endpoint>` line on stdout, and serves the
//! v2 contract over it; the host connects only after reading the line.

use std::path::PathBuf;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tonic::Status;
use tonic::transport::Server;

use crate::client::WorkerClient;
use crate::proto::worker_server::WorkerServer;
use crate::{auth::GrpcAuth, proto};

/// A local endpoint a worker listens on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LocalEndpoint {
    /// Unix domain socket at an absolute path (`unix://<path>`).
    #[cfg(unix)]
    UnixSocket(PathBuf),
    /// Windows named pipe (`pipe://\\.\pipe\<name>`).
    #[cfg(windows)]
    NamedPipe(String),
}

impl LocalEndpoint {
    /// Parse an endpoint string: `unix://<absolute path>` or
    /// `pipe://<pipe name>`.
    pub fn parse(endpoint: &str) -> Result<Self, LocalListenError> {
        if let Some(path) = endpoint.strip_prefix("unix://") {
            #[cfg(unix)]
            {
                let path = PathBuf::from(path);
                if !path.is_absolute() {
                    return Err(LocalListenError::InvalidEndpoint(format!(
                        "unix socket path `{}` must be absolute",
                        path.display()
                    )));
                }
                return Ok(Self::UnixSocket(path));
            }
            #[cfg(windows)]
            {
                let _ = path;
                return Err(LocalListenError::InvalidEndpoint(
                    "unix sockets are not available on Windows; use pipe://".to_owned(),
                ));
            }
        }
        if let Some(name) = endpoint.strip_prefix("pipe://") {
            #[cfg(windows)]
            {
                if name.is_empty() {
                    return Err(LocalListenError::InvalidEndpoint(
                        "pipe name must not be empty".to_owned(),
                    ));
                }
                return Ok(Self::NamedPipe(name.to_owned()));
            }
            #[cfg(unix)]
            {
                let _ = name;
                return Err(LocalListenError::InvalidEndpoint(
                    "named pipes are only available on Windows; use unix://".to_owned(),
                ));
            }
        }
        Err(LocalListenError::InvalidEndpoint(format!(
            "local worker endpoints must start with unix:// or pipe://, got `{endpoint}`"
        )))
    }

    /// The canonical endpoint string (`unix://<path>` / `pipe://<name>`),
    /// as printed in the worker's READY line.
    #[must_use]
    pub fn to_uri(&self) -> String {
        match self {
            #[cfg(unix)]
            Self::UnixSocket(path) => format!("unix://{}", path.display()),
            #[cfg(windows)]
            Self::NamedPipe(name) => format!("pipe://{name}"),
        }
    }
}

/// Errors from binding a local endpoint.
#[derive(Debug)]
pub enum LocalListenError {
    Io(String),
    InvalidEndpoint(String),
}

impl std::fmt::Display for LocalListenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) => write!(f, "local worker listen failed: {message}"),
            Self::InvalidEndpoint(message) => write!(f, "invalid local endpoint: {message}"),
        }
    }
}

impl std::error::Error for LocalListenError {}

/// A connected local stream (Unix socket or named pipe), erased behind
/// one type so the tonic server can serve both platforms uniformly.
pub struct LocalIo {
    #[cfg(unix)]
    inner: tokio::net::UnixStream,
    #[cfg(windows)]
    inner: tokio::net::windows::named_pipe::NamedPipeServer,
}

impl AsyncRead for LocalIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for LocalIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_shutdown(cx)
    }
}

impl tonic::transport::server::Connected for LocalIo {
    type ConnectInfo = ();

    fn connect_info(&self) -> Self::ConnectInfo {}
}

/// An incoming stream of local connections, ready for
/// `Server::serve_with_incoming`.
pub struct LocalIncoming {
    #[cfg(unix)]
    listener: tokio::net::UnixListener,
    #[cfg(windows)]
    pipe_name: String,
    #[cfg(windows)]
    current: Option<tokio::net::windows::named_pipe::NamedPipeServer>,
    /// Whether a pipe instance was already created for `pipe_name`; only
    /// the very first instance may use `first_pipe_instance`.
    #[cfg(windows)]
    created: bool,
}

impl futures_core::Stream for LocalIncoming {
    type Item = std::io::Result<LocalIo>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        #[cfg(unix)]
        {
            match this.listener.poll_accept(cx) {
                Poll::Ready(Ok((stream, _addr))) => {
                    Poll::Ready(Some(Ok(LocalIo { inner: stream })))
                }
                Poll::Ready(Err(error)) => Poll::Ready(Some(Err(error))),
                Poll::Pending => Poll::Pending,
            }
        }
        #[cfg(windows)]
        {
            loop {
                if this.current.is_none() {
                    match tokio::net::windows::named_pipe::ServerOptions::new()
                        .first_pipe_instance(!this.created)
                        .create(&this.pipe_name)
                    {
                        Ok(server) => this.current = Some(server),
                        Err(error) => return Poll::Ready(Some(Err(error))),
                    }
                    this.created = true;
                }
                let server = this.current.as_mut().expect("pipe instance created above");
                match server.poll_connect(cx) {
                    Poll::Ready(Ok(())) => {
                        let server = this.current.take().expect("instance present");
                        return Poll::Ready(Some(Ok(LocalIo { inner: server })));
                    }
                    Poll::Ready(Err(error)) => {
                        this.current = None;
                        return Poll::Ready(Some(Err(error)));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        }
    }
}

/// Bind a local endpoint so a worker can serve the v2 contract on it.
///
/// A stale socket file at the path is removed first; the created socket
/// is restricted to its owner. Must run inside a Tokio runtime.
pub async fn bind(endpoint: &LocalEndpoint) -> Result<LocalIncoming, LocalListenError> {
    match endpoint {
        #[cfg(unix)]
        LocalEndpoint::UnixSocket(path) => {
            let _ = std::fs::remove_file(path);
            let listener = tokio::net::UnixListener::bind(path)
                .map_err(|error| LocalListenError::Io(format!("{}: {error}", path.display())))?;
            use std::os::unix::fs::PermissionsExt;
            if let Err(error) =
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            {
                tracing::warn!("could not restrict worker socket permissions: {error}");
            }
            Ok(LocalIncoming { listener })
        }
        #[cfg(windows)]
        LocalEndpoint::NamedPipe(name) => {
            let mut incoming = LocalIncoming {
                pipe_name: name.clone(),
                current: None,
                created: false,
            };
            // Create the first pipe instance eagerly so a bind failure
            // surfaces before the READY line is printed.
            incoming.current = Some(
                tokio::net::windows::named_pipe::ServerOptions::new()
                    .first_pipe_instance(true)
                    .create(name)
                    .map_err(|error| LocalListenError::Io(format!("pipe {name}: {error}")))?,
            );
            incoming.created = true;
            Ok(incoming)
        }
    }
}

/// Serve the v2 contract for `service` on a bound local endpoint.
///
/// This future runs until the incoming stream errors or the runtime is
/// torn down; the worker supervises it alongside its stdin lifeline.
pub async fn serve<S>(
    incoming: LocalIncoming,
    service: WorkerServer<S>,
) -> Result<(), tonic::transport::Error>
where
    S: proto::worker_server::Worker + Send + Sync + 'static,
{
    Server::builder()
        .add_service(
            service
                .max_decoding_message_size(crate::MAX_MESSAGE_BYTES)
                .max_encoding_message_size(crate::MAX_MESSAGE_BYTES),
        )
        .serve_with_incoming(incoming)
        .await
}

/// Connect a [`WorkerClient`] to a worker serving on a local endpoint.
pub async fn connect_local(
    endpoint: &LocalEndpoint,
    auth: &GrpcAuth,
) -> Result<WorkerClient, Status> {
    match endpoint {
        #[cfg(unix)]
        LocalEndpoint::UnixSocket(path) => {
            WorkerClient::connect(&format!("unix://{}", path.display()), auth).await
        }
        #[cfg(windows)]
        LocalEndpoint::NamedPipe(name) => connect_named_pipe(name, auth).await,
    }
}

#[cfg(windows)]
async fn connect_named_pipe(name: &str, auth: &GrpcAuth) -> Result<WorkerClient, Status> {
    use tonic::transport::Endpoint;

    let connector = NamedPipeConnector {
        pipe_name: name.to_owned(),
    };
    // The endpoint URI is a placeholder: the connector ignores it and
    // opens the named pipe directly.
    let endpoint = Endpoint::from_shared("http://localhost")
        .map_err(|error| Status::internal(format!("placeholder endpoint: {error}")))?;
    let channel = endpoint
        .connect_with_connector(connector)
        .await
        .map_err(|error| Status::unavailable(format!("connect to pipe {name} failed: {error}")))?;
    Ok(WorkerClient::from_channel(
        channel,
        format!("pipe://{name}"),
        auth,
    ))
}

/// Windows named-pipe connector: opens a client pipe for every new
/// connection hyper requests.
#[cfg(windows)]
#[derive(Clone)]
struct NamedPipeConnector {
    pipe_name: String,
}

#[cfg(windows)]
impl tower::Service<tonic::codegen::http::Uri> for NamedPipeConnector {
    type Response = hyper_util::rt::TokioIo<tokio::net::windows::named_pipe::NamedPipeClient>;
    type Error = std::io::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _uri: tonic::codegen::http::Uri) -> Self::Future {
        let pipe_name = self.pipe_name.clone();
        Box::pin(async move {
            // `connect` fails with ERROR_PIPE_BUSY while the server is
            // between instances; retry briefly instead of failing.
            let mut attempt = 0;
            loop {
                match tokio::net::windows::named_pipe::ClientOptions::new().open(&pipe_name) {
                    Ok(client) => return Ok(hyper_util::rt::TokioIo::new(client)),
                    Err(error) if attempt < 50 => {
                        attempt += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        let _ = error;
                    }
                    Err(error) => return Err(error),
                }
            }
        })
    }
}
