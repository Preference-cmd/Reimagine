//! Server-side glue for the `reimagine.worker.v2` contract.
//!
//! Worker binaries implement the generated [`proto::worker_server::Worker`]
//! trait, parse their `--listen` endpoint into a [`ListenEndpoint`], bind
//! it with [`bind_listen`], and serve it with [`serve_bound`]; token
//! enforcement is a one-line call per RPC via [`require_bearer`].

use std::net::SocketAddr;

use tonic::Request;
use tonic::Status;
use tonic::transport::Server;

use crate::auth::check_bearer;
use crate::local::{LocalEndpoint, LocalIncoming, LocalListenError};
use crate::proto;
use crate::proto::worker_server::WorkerServer;
use crate::tls::SelfSignedIdentity;

/// Build a tonic server for `service` with the worker message-size
/// limits applied.
#[must_use]
pub fn worker_server<S>(service: S) -> WorkerServer<S>
where
    S: proto::worker_server::Worker,
{
    WorkerServer::new(service)
        .max_decoding_message_size(crate::MAX_MESSAGE_BYTES)
        .max_encoding_message_size(crate::MAX_MESSAGE_BYTES)
}

/// Serve the v2 contract on a bound local endpoint.
///
/// This future runs until the incoming stream errors or the runtime is
/// torn down; workers supervise it alongside their stdin lifeline.
pub async fn serve_local<S>(
    incoming: LocalIncoming,
    service: WorkerServer<S>,
) -> Result<(), tonic::transport::Error>
where
    S: proto::worker_server::Worker + Send + Sync + 'static,
{
    crate::local::serve(incoming, service).await
}

/// Reject a request whose bearer token does not match `expected`.
///
/// Every RPC on a token-authenticated worker must call this first.
pub fn require_bearer<T>(request: &Request<T>, expected: Option<&str>) -> Result<(), Status> {
    check_bearer(request.metadata(), expected)
}

/// An endpoint a worker may be asked to listen on, as passed on the
/// worker command line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ListenEndpoint {
    /// A private local endpoint (`unix://<path>`, Windows `pipe://<name>`).
    Local(LocalEndpoint),
    /// TLS over TCP for remote hosts (`https://<ip>:<port>`): a
    /// self-signed certificate, the same bearer-token authentication as
    /// local endpoints, and mDNS registration for discovery.
    Https(SocketAddr),
}

impl ListenEndpoint {
    /// Parse an endpoint string. Only `unix://`, `pipe://` and
    /// `https://` are accepted; every other scheme is an error.
    pub fn parse(endpoint: &str) -> Result<Self, LocalListenError> {
        if let Some(addr) = endpoint.strip_prefix("https://") {
            let addr: SocketAddr = addr.parse().map_err(|_| {
                LocalListenError::InvalidEndpoint(format!(
                    "https endpoint must be `<ip>:<port>`, got `{endpoint}`"
                ))
            })?;
            return Ok(Self::Https(addr));
        }
        if endpoint.starts_with("unix://") || endpoint.starts_with("pipe://") {
            return LocalEndpoint::parse(endpoint).map(Self::Local);
        }
        Err(LocalListenError::InvalidEndpoint(format!(
            "worker listen endpoints must start with unix://, pipe:// or https://, got `{endpoint}`"
        )))
    }

    /// The canonical endpoint URI (`unix://<path>`, `pipe://<name>` or
    /// `https://<ip>:<port>`), as printed in the worker's READY line.
    #[must_use]
    pub fn to_uri(&self) -> String {
        match self {
            Self::Local(local) => local.to_uri(),
            Self::Https(addr) => format!("https://{addr}"),
        }
    }
}

/// A bound worker listener, ready for [`serve_bound`].
///
/// Binding is separate from serving so a worker can resolve a `:0` port,
/// publish the real address (mDNS registration, READY line), and only
/// then start accepting requests.
pub struct BoundListen {
    uri: String,
    inner: BoundListener,
}

enum BoundListener {
    Local(LocalIncoming),
    Https {
        listener: tokio::net::TcpListener,
        identity: SelfSignedIdentity,
    },
}

impl BoundListen {
    /// Bind an `https://` endpoint with the given self-signed identity.
    ///
    /// The escape hatch beside [`bind_listen`] for callers that must
    /// control the certificate (tests, tooling that re-presents a saved
    /// identity).
    pub async fn bind_https_with_identity(
        addr: SocketAddr,
        identity: SelfSignedIdentity,
    ) -> Result<Self, LocalListenError> {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|error| LocalListenError::Io(format!("https listen on {addr}: {error}")))?;
        let real_addr = listener
            .local_addr()
            .map_err(|error| LocalListenError::Io(format!("https local address: {error}")))?;
        Ok(Self {
            uri: format!("https://{real_addr}"),
            inner: BoundListener::Https { listener, identity },
        })
    }

    /// The canonical endpoint URI actually bound (`:0` resolved to the
    /// real port), as printed in the worker's READY line.
    #[must_use]
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// The self-signed identity serving the `https://` endpoint.
    ///
    /// `None` for local endpoints, which carry no certificate.
    #[must_use]
    pub fn identity(&self) -> Option<&SelfSignedIdentity> {
        match &self.inner {
            BoundListener::Local(_) => None,
            BoundListener::Https { identity, .. } => Some(identity),
        }
    }

    /// The bound TCP address of the `https://` listener.
    ///
    /// `None` for local endpoints. An explicitly requested `:0` is
    /// resolved to the real port; an unspecified bind address
    /// (`0.0.0.0`/`::`) stays unspecified.
    #[must_use]
    pub fn https_addr(&self) -> Option<SocketAddr> {
        match &self.inner {
            BoundListener::Local(_) => None,
            BoundListener::Https { listener, .. } => listener.local_addr().ok(),
        }
    }
}

/// Bind a parsed listen endpoint so a worker can serve the v2 contract.
///
/// Local endpoints get a stale-socket cleanup and owner-restricted
/// permissions (see [`crate::local::bind`]); `https://` endpoints get a
/// freshly generated self-signed certificate. Must run inside a Tokio
/// runtime.
pub async fn bind_listen(listen: &ListenEndpoint) -> Result<BoundListen, LocalListenError> {
    match listen {
        ListenEndpoint::Local(local) => {
            let incoming = crate::local::bind(local).await?;
            Ok(BoundListen {
                uri: local.to_uri(),
                inner: BoundListener::Local(incoming),
            })
        }
        ListenEndpoint::Https(addr) => {
            let identity = crate::tls::generate_self_signed_identity("localhost")
                .map_err(LocalListenError::Io)?;
            BoundListen::bind_https_with_identity(*addr, identity).await
        }
    }
}

/// Serve the v2 contract for `service` on a bound listener.
///
/// Local endpoints serve plain HTTP/2 over the socket; `https://`
/// endpoints serve TLS with the bound self-signed identity. This future
/// runs until the incoming stream errors or the runtime is torn down;
/// workers supervise it alongside their stdin lifeline.
pub async fn serve_bound<S>(
    bound: BoundListen,
    service: WorkerServer<S>,
) -> Result<(), tonic::transport::Error>
where
    S: proto::worker_server::Worker + Send + Sync + 'static,
{
    match bound.inner {
        BoundListener::Local(incoming) => crate::local::serve(incoming, service).await,
        BoundListener::Https { listener, identity } => {
            let tls = tonic::transport::ServerTlsConfig::new().identity(
                tonic::transport::Identity::from_pem(identity.cert_pem, identity.key_pem),
            );
            Server::builder()
                .tls_config(tls)?
                .add_service(service)
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
        }
    }
}
