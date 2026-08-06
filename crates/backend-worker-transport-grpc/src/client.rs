use std::sync::Arc;

use tonic::transport::{Certificate, Channel, ClientTlsConfig};

use crate::auth::bearer_interceptor;
use crate::proto::worker_service_client::WorkerServiceClient;
use crate::transport::GrpcTransport;

/// TLS mode for a gRPC client connect.
#[derive(Clone, Debug)]
pub enum GrpcTls {
    /// Trust the given PEM certificate (e.g. the worker's self-signed
    /// certificate or a custom CA). The endpoint URI must use `https://`.
    TrustCert { ca_pem: String, domain: String },
    /// Accept any server certificate (dev only). The endpoint URI must
    /// use `https://`.
    InsecureSkipVerify { domain: String },
}

/// Authentication settings for a gRPC client connect.
#[derive(Clone, Debug, Default)]
pub struct GrpcAuth {
    /// Bearer token attached as `authorization: Bearer <token>` on every
    /// request. `None` keeps the plain (open) transport.
    pub token: Option<String>,
    /// TLS mode; `None` keeps plain HTTP.
    pub tls: Option<GrpcTls>,
}

impl GrpcAuth {
    /// Plain, open connection (backward compatible with T7 behavior).
    #[must_use]
    pub fn plain() -> Self {
        Self::default()
    }

    /// Read the token from the `REIMAGINE_WORKER_TOKEN` environment
    /// variable (empty/unset => no token).
    ///
    /// Note: `connect_with` refuses to send a configured token over a
    /// plain `http://` endpoint (see [`GrpcAuth`] docs); pair this with
    /// a `tls` setting for real cloud deployments.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            token: crate::auth::token_from_env(),
            tls: None,
        }
    }
}

/// Connect to a remote gRPC worker over plain HTTP without auth.
///
/// Backward-compatible shorthand for [`connect_with`] with
/// [`GrpcAuth::plain`].
pub async fn connect(endpoint: &str) -> Result<GrpcTransport, tonic::Status> {
    connect_with(endpoint, &GrpcAuth::plain()).await
}

/// Connect to a remote gRPC worker and perform the initial handshake.
///
/// Returns a `GrpcTransport` ready for protocol communication.
///
/// `endpoint` should be a URI like `http://127.0.0.1:50051` (plain) or
/// `https://worker.example:50051` (when `auth.tls` is set).
pub async fn connect_with(endpoint: &str, auth: &GrpcAuth) -> Result<GrpcTransport, tonic::Status> {
    crate::tls::ensure_crypto_provider();
    // Never send a bearer token in cleartext: a token without TLS is a
    // credential leak. Callers must pair tokens with a `tls` mode
    // (or accept the risk by spelling out the endpoint as http:// AND
    // setting auth.tls = Some(GrpcTls::InsecureSkipVerify) — the
    // explicit dev escape hatch, which also fails this check for
    // https:// endpoints only if the caller chooses to).
    if auth.token.is_some()
        && auth.tls.is_none()
        && !endpoint.to_ascii_lowercase().starts_with("https://")
    {
        return Err(tonic::Status::unauthenticated(
            "bearer token configured but transport is not TLS (refusing cleartext token)",
        ));
    }
    let mut channel_builder = Channel::from_shared(endpoint.to_owned())
        .map_err(|e| tonic::Status::internal(format!("invalid endpoint: {e}")))?;

    if let Some(tls) = &auth.tls {
        match tls {
            GrpcTls::TrustCert { ca_pem, domain } => {
                let tls_config = ClientTlsConfig::new()
                    .ca_certificate(Certificate::from_pem(ca_pem))
                    .domain_name(domain.clone());
                channel_builder = channel_builder
                    .tls_config(tls_config)
                    .map_err(|e| tonic::Status::internal(format!("invalid TLS config: {e}")))?;
            }
            GrpcTls::InsecureSkipVerify { domain } => {
                let tls_config = ClientTlsConfig::new().domain_name(domain.clone());
                let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
                    Arc::new(crate::tls::AcceptAnyServerVerifier::new());
                channel_builder = channel_builder
                    .tls_config_with_verifier(tls_config, verifier)
                    .map_err(|e| tonic::Status::internal(format!("invalid TLS config: {e}")))?;
            }
        }
    }

    let channel = channel_builder
        .connect()
        .await
        .map_err(|e| tonic::Status::internal(format!("connect failed: {e}")))?;

    let mut client =
        WorkerServiceClient::with_interceptor(channel, bearer_interceptor(auth.token.clone()));

    let (tx, rx) = GrpcTransport::channel();
    let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    let response = client.communication(rx_stream).await?;
    let stream = response.into_inner();

    Ok(GrpcTransport::new(tx, stream, endpoint.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto;
    use crate::proto::worker_service_server::WorkerServiceServer;
    use crate::server::{GrpcWorkerService, MessageHandler};
    use reimagine_backend_worker_protocol::WorkerTransport;
    use std::sync::Arc;

    fn echo_handler() -> MessageHandler {
        Arc::new(|msg| {
            Box::pin(async move {
                match msg.message {
                    Some(proto::host_to_worker::Message::Request(r)) => Some(proto::WorkerToHost {
                        message: Some(proto::worker_to_host::Message::Terminal(proto::Terminal {
                            protocol_version: r.protocol_version,
                            incarnation_id: r.incarnation_id,
                            request_id: r.request_id,
                            correlation_id: r.correlation_id,
                            outcome: Some(proto::TerminalOutcome {
                                outcome: Some(proto::terminal_outcome::Outcome::Success(
                                    proto::Success { output: r.payload },
                                )),
                            }),
                        })),
                    }),
                    _ => None,
                }
            })
        })
    }

    #[tokio::test]
    async fn client_connects_and_performs_handshake() {
        let service = GrpcWorkerService::new(echo_handler());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tonic::transport::Server::builder()
            .add_service(WorkerServiceServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

        tokio::spawn(server);

        let transport = connect(&format!("http://{addr}")).await.unwrap();

        assert_eq!(
            transport.description().kind,
            reimagine_backend_worker_protocol::TransportKind::Grpc
        );

        // Send a request and receive a terminal response
        transport
            .send(proto::HostToWorker {
                message: Some(proto::host_to_worker::Message::Request(proto::Request {
                    protocol_version: 1,
                    incarnation_id: "inc-1".into(),
                    request_id: "req-1".into(),
                    correlation_id: "corr-1".into(),
                    operation: "echo".into(),
                    payload: serde_json::to_vec(&serde_json::json!({"hello": "world"})).unwrap(),
                })),
            })
            .await
            .unwrap();

        let response = transport.recv().await.unwrap().unwrap();
        assert!(matches!(
            response.message,
            Some(proto::worker_to_host::Message::Terminal(_))
        ));
    }
}
