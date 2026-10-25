//! Typed client for the `reimagine.worker.v2` contract.
//!
//! One `WorkerClient` wraps a tonic channel and attaches the bearer
//! token to every call. Connections are made over an endpoint URI:
//! `unix://<path>` (Unix socket), `http(s)://<host:port>` (remote), or a
//! Windows named pipe via [`crate::local`].

use tonic::Status;
use tonic::Streaming;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

use crate::auth::BearerInterceptor;
use crate::auth::GrpcAuth;
use crate::proto::worker_client::WorkerClient as GeneratedClient;
use crate::proto::{ExecuteEvent, ExecuteRequest, HelloRequest};
use crate::tls::{GrpcTls, ensure_crypto_provider};

/// A connected worker client speaking the v2 contract.
#[derive(Clone, Debug)]
pub struct WorkerClient {
    inner: GeneratedClient<InterceptedService<Channel, BearerInterceptor>>,
    endpoint: String,
}

impl WorkerClient {
    /// Connect to a worker at `endpoint` and authenticate with `auth`.
    ///
    /// `endpoint` is a URI: `unix://<absolute socket path>` for a local
    /// worker, `http(s)://host:port` for a remote one. On Windows, named
    /// pipe connections go through [`crate::local::connect_local`] instead.
    pub async fn connect(endpoint: &str, auth: &GrpcAuth) -> Result<Self, Status> {
        let channel = match connect_channel(endpoint, auth).await {
            Ok(channel) => channel,
            // Surface connect failures as `unavailable`: the worker is
            // not reachable, which is a transport condition rather than
            // a logic error.
            Err(error) => return Err(Status::unavailable(error.to_string())),
        };
        Ok(Self::from_channel(channel, endpoint.to_owned(), auth))
    }

    /// Wrap an already-connected channel (used for in-process tests and
    /// custom transports).
    #[must_use]
    pub fn from_channel(channel: Channel, endpoint: String, auth: &GrpcAuth) -> Self {
        let inner =
            GeneratedClient::with_interceptor(channel, BearerInterceptor::new(auth.token.clone()))
                .max_decoding_message_size(crate::MAX_MESSAGE_BYTES)
                .max_encoding_message_size(crate::MAX_MESSAGE_BYTES);
        Self { inner, endpoint }
    }

    /// The endpoint description this client was built from.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Fetch the worker's hello (protocol version, identity, profile).
    pub async fn hello(&self) -> Result<crate::proto::HelloResponse, Status> {
        Ok(self
            .inner
            .clone()
            .hello(HelloRequest {})
            .await?
            .into_inner())
    }

    /// Start an `Execute` call and return the event stream.
    ///
    /// The stream must be consumed; dropping it cancels the call, which
    /// is how the host cancels a running operation.
    pub async fn execute(
        &self,
        operation: impl Into<String>,
        payload: Vec<u8>,
    ) -> Result<Streaming<ExecuteEvent>, Status> {
        let request = ExecuteRequest {
            operation: operation.into(),
            payload,
        };
        Ok(self.inner.clone().execute(request).await?.into_inner())
    }

    /// Check worker liveness.
    pub async fn health(&self) -> Result<crate::proto::HealthResponse, Status> {
        Ok(self
            .inner
            .clone()
            .health(crate::proto::HealthRequest {})
            .await?
            .into_inner())
    }

    /// Release run-scoped payloads held by the worker.
    pub async fn cleanup(
        &self,
        run_id: Option<String>,
        object_ids: Vec<String>,
    ) -> Result<crate::proto::CleanupResponse, Status> {
        Ok(self
            .inner
            .clone()
            .cleanup(crate::proto::CleanupRequest { run_id, object_ids })
            .await?
            .into_inner())
    }

    /// Ask the worker to drain and exit.
    pub async fn shutdown(&self) -> Result<crate::proto::ShutdownResponse, Status> {
        Ok(self
            .inner
            .clone()
            .shutdown(crate::proto::ShutdownRequest {})
            .await?
            .into_inner())
    }
}

async fn connect_channel(endpoint: &str, auth: &GrpcAuth) -> Result<Channel, ConnectError> {
    // Never send a bearer token in cleartext: a token without TLS is a
    // credential leak. Local worker sockets (`unix://`) are exempt —
    // they are protected by the private socket path plus the token.
    if auth.token.is_some()
        && !endpoint.to_ascii_lowercase().starts_with("https://")
        && !endpoint.to_ascii_lowercase().starts_with("unix://")
    {
        return Err(ConnectError(
            "refusing to send a bearer token over a cleartext endpoint; use https:// (or a local unix:// socket)".to_owned(),
        ));
    }
    let mut channel_builder = Endpoint::from_shared(endpoint.to_owned())
        .map_err(|error| ConnectError(format!("invalid endpoint `{endpoint}`: {error}")))?;

    if let Some(tls) = &auth.tls {
        match tls {
            GrpcTls::TrustCert { ca_pem, domain } => {
                let tls_config = ClientTlsConfig::new()
                    .ca_certificate(Certificate::from_pem(ca_pem))
                    .domain_name(domain.clone());
                channel_builder = channel_builder
                    .tls_config(tls_config)
                    .map_err(|error| ConnectError(format!("invalid TLS config: {error}")))?;
            }
            GrpcTls::InsecureSkipVerify { domain } => {
                let tls_config = ClientTlsConfig::new().domain_name(domain.clone());
                let verifier: std::sync::Arc<dyn rustls::client::danger::ServerCertVerifier> =
                    std::sync::Arc::new(ensure_crypto_provider_and_verifier());
                channel_builder = channel_builder
                    .tls_config_with_verifier(tls_config, verifier)
                    .map_err(|error| ConnectError(format!("invalid TLS config: {error}")))?;
            }
        }
    }

    channel_builder
        .connect()
        .await
        .map_err(|error| ConnectError(format!("connect to `{endpoint}` failed: {error}")))
}

fn ensure_crypto_provider_and_verifier() -> crate::tls::AcceptAnyServerVerifier {
    ensure_crypto_provider();
    crate::tls::AcceptAnyServerVerifier::new()
}

/// Connection failure text for the caller to wrap.
#[derive(Debug)]
struct ConnectError(String);

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConnectError {}
