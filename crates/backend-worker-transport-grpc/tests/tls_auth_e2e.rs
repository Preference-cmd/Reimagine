//! End-to-end tests for gRPC TLS + bearer-token auth (T19).

use std::sync::Arc;

use reimagine_backend_worker_protocol::WorkerTransport as _;
use reimagine_backend_worker_transport_grpc::client::{self, GrpcAuth, GrpcTls};
use reimagine_backend_worker_transport_grpc::proto;
use reimagine_backend_worker_transport_grpc::proto::worker_service_server::WorkerServiceServer;
use reimagine_backend_worker_transport_grpc::server::{GrpcWorkerService, MessageHandler};
use reimagine_backend_worker_transport_grpc::tls::{
    AcceptAnyServerVerifier, generate_self_signed_identity,
};
use tonic::transport::{Identity, ServerTlsConfig};

/// A worker handler that echoes requests back as successful terminals.
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

/// Start a gRPC server with an optional bearer token and optional TLS
/// identity; returns the endpoint URI and the PEM cert (when TLS).
async fn start_server(
    token: Option<String>,
    tls: Option<(String, String)>,
) -> (String, Option<String>) {
    let service = GrpcWorkerService::with_token(echo_handler(), token);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let scheme = if tls.is_some() { "https" } else { "http" };
    let mut builder = tonic::transport::Server::builder();
    if let Some((cert_pem, key_pem)) = tls {
        builder = builder
            .tls_config(ServerTlsConfig::new().identity(Identity::from_pem(cert_pem, key_pem)))
            .unwrap();
    }
    let server = builder
        .add_service(WorkerServiceServer::new(service))
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

    tokio::spawn(server);

    (format!("{scheme}://{addr}"), None)
}

async fn health_check(
    endpoint: &str,
    auth: &GrpcAuth,
) -> Result<proto::HealthResponse, tonic::Status> {
    // HealthCheck is exercised directly (non-streaming path) with the
    // same token interceptor the client transport uses. Apply the same
    // TLS modes as client::connect_with so https endpoints work.
    let mut channel_builder = tonic::transport::Channel::from_shared(endpoint.to_owned()).unwrap();
    if let Some(tls) = &auth.tls {
        let tls_config = tonic::transport::ClientTlsConfig::new().domain_name(match tls {
            GrpcTls::TrustCert { domain, .. } | GrpcTls::InsecureSkipVerify { domain } => {
                domain.clone()
            }
        });
        match tls {
            GrpcTls::TrustCert { ca_pem, .. } => {
                channel_builder = channel_builder
                    .tls_config(
                        tls_config.ca_certificate(tonic::transport::Certificate::from_pem(ca_pem)),
                    )
                    .unwrap();
            }
            GrpcTls::InsecureSkipVerify { .. } => {
                let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> = Arc::new(
                    reimagine_backend_worker_transport_grpc::tls::AcceptAnyServerVerifier::new(),
                );
                channel_builder = channel_builder
                    .tls_config_with_verifier(tls_config, verifier)
                    .unwrap();
            }
        }
    }
    let channel = channel_builder.connect_lazy();
    let mut client = proto::worker_service_client::WorkerServiceClient::with_interceptor(
        channel,
        reimagine_backend_worker_transport_grpc::auth::bearer_interceptor(auth.token.clone()),
    );
    client
        .health_check(proto::HealthRequest {})
        .await
        .map(tonic::Response::into_inner)
}

#[tokio::test]
async fn auth_rejects_missing_and_wrong_token() {
    let (endpoint, _) = start_server(Some("s3cret".to_owned()), None).await;

    // No token configured on the client -> rejected.
    let error = client::connect(&endpoint).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);

    // Wrong token -> rejected.
    let error = client::connect_with(
        &endpoint,
        &GrpcAuth {
            token: Some("wrong".to_owned()),
            tls: None,
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);

    // HealthCheck is also protected.
    let error = health_check(
        &endpoint,
        &GrpcAuth {
            token: None,
            tls: None,
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn auth_refuses_cleartext_token_transport() {
    // MAJOR-2 guard: a configured token must never travel over plain
    // HTTP; connect_with refuses before any request is sent.
    let (endpoint, _) = start_server(Some("s3cret".to_owned()), None).await;
    let error = client::connect_with(
        &endpoint,
        &GrpcAuth {
            token: Some("s3cret".to_owned()),
            tls: None,
        },
    )
    .await
    .expect_err("plain HTTP + token must be refused");
    assert_eq!(error.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn auth_accepts_correct_token() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _) = start_server(
        Some("s3cret".to_owned()),
        Some((identity.cert_pem, identity.key_pem)),
    )
    .await;

    // A token requires TLS (cleartext-token guard); the dev escape
    // hatch (InsecureSkipVerify) satisfies the guard for this test.
    let transport = client::connect_with(
        &endpoint,
        &GrpcAuth {
            token: Some("s3cret".to_owned()),
            tls: Some(GrpcTls::InsecureSkipVerify {
                domain: "localhost".to_owned(),
            }),
        },
    )
    .await
    .expect("correct token must connect");

    // Echo roundtrip over the authenticated bidi stream.
    transport
        .send(proto::HostToWorker {
            message: Some(proto::host_to_worker::Message::Request(proto::Request {
                protocol_version: 1,
                incarnation_id: "inc-1".into(),
                request_id: "req-1".into(),
                correlation_id: "corr-1".into(),
                operation: "echo".into(),
                payload: serde_json::to_vec(&serde_json::json!({"ok": true})).unwrap(),
            })),
        })
        .await
        .unwrap();
    let response = transport.recv().await.unwrap().unwrap();
    assert!(matches!(
        response.message,
        Some(proto::worker_to_host::Message::Terminal(_))
    ));

    // HealthCheck with token succeeds (TLS server ⇒ TLS client).
    let health = health_check(
        &endpoint,
        &GrpcAuth {
            token: Some("s3cret".to_owned()),
            tls: Some(GrpcTls::InsecureSkipVerify {
                domain: "localhost".to_owned(),
            }),
        },
    )
    .await
    .unwrap();
    assert!(health.healthy);
}

#[tokio::test]
async fn tls_trusted_cert_rejects_wrong_ca() {
    // A client trusting a DIFFERENT CA must be rejected by the server's
    // self-signed identity (webpki chain verification).
    let identity = generate_self_signed_identity("localhost").unwrap();
    let other = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _) = start_server(None, Some((identity.cert_pem, identity.key_pem))).await;

    let error = client::connect_with(
        &endpoint,
        &GrpcAuth {
            token: None,
            tls: Some(GrpcTls::TrustCert {
                ca_pem: other.cert_pem,
                domain: "localhost".to_owned(),
            }),
        },
    )
    .await
    .expect_err("wrong CA must be rejected");
    // The transport error surfaces through connect_with's mapping
    // (Internal "connect failed: ..."); any non-success is the point.
    assert_ne!(error.code(), tonic::Code::Ok, "got {error:?}");
}

#[tokio::test]
async fn no_token_configured_stays_open() {
    let (endpoint, _) = start_server(None, None).await;

    // Plain connect against an open server: backward-compatible behavior.
    let transport = client::connect(&endpoint)
        .await
        .expect("open server accepts");
    transport
        .send(proto::HostToWorker {
            message: Some(proto::host_to_worker::Message::Request(proto::Request {
                protocol_version: 1,
                incarnation_id: "inc-1".into(),
                request_id: "req-1".into(),
                correlation_id: "corr-1".into(),
                operation: "echo".into(),
                payload: Vec::new(),
            })),
        })
        .await
        .unwrap();
    let response = transport.recv().await.unwrap().unwrap();
    assert!(matches!(
        response.message,
        Some(proto::worker_to_host::Message::Terminal(_))
    ));

    // HealthCheck stays open too.
    assert!(health_check(&endpoint, &GrpcAuth::plain()).await.is_ok());
}

#[tokio::test]
async fn tls_roundtrip_with_trusted_cert() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _) =
        start_server(None, Some((identity.cert_pem.clone(), identity.key_pem))).await;

    let transport = client::connect_with(
        &endpoint,
        &GrpcAuth {
            token: None,
            tls: Some(GrpcTls::TrustCert {
                ca_pem: identity.cert_pem.clone(),
                domain: "localhost".to_owned(),
            }),
        },
    )
    .await
    .expect("TLS connect with trusted self-signed cert must succeed");

    transport
        .send(proto::HostToWorker {
            message: Some(proto::host_to_worker::Message::Request(proto::Request {
                protocol_version: 1,
                incarnation_id: "inc-1".into(),
                request_id: "req-1".into(),
                correlation_id: "corr-1".into(),
                operation: "echo".into(),
                payload: serde_json::to_vec(&serde_json::json!({"tls": true})).unwrap(),
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

#[tokio::test]
async fn tls_insecure_skip_verify_accepts_any_cert() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _) = start_server(None, Some((identity.cert_pem, identity.key_pem))).await;

    let transport = client::connect_with(
        &endpoint,
        &GrpcAuth {
            token: None,
            tls: Some(GrpcTls::InsecureSkipVerify {
                domain: "localhost".to_owned(),
            }),
        },
    )
    .await
    .expect("insecure-skip TLS connect must succeed");

    transport.shutdown().await.unwrap();
}

#[tokio::test]
async fn plain_http_against_tls_server_fails() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _) = start_server(None, Some((identity.cert_pem, identity.key_pem))).await;

    // Plain HTTP client against a TLS-only server: the handshake fails.
    let error = client::connect(&endpoint).await.unwrap_err();
    assert!(
        matches!(
            error.code(),
            tonic::Code::Internal | tonic::Code::Unavailable
        ),
        "expected connect failure, got {error:?}"
    );
}

#[tokio::test]
async fn tls_and_token_combined() {
    let identity = generate_self_signed_identity("localhost").unwrap();
    let (endpoint, _) = start_server(
        Some("cloud-token".to_owned()),
        Some((identity.cert_pem.clone(), identity.key_pem)),
    )
    .await;

    // Missing token over TLS -> unauthenticated.
    let error = client::connect_with(
        &endpoint,
        &GrpcAuth {
            token: None,
            tls: Some(GrpcTls::TrustCert {
                ca_pem: identity.cert_pem.clone(),
                domain: "localhost".to_owned(),
            }),
        },
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Unauthenticated);

    // Correct token over TLS -> works.
    let transport = client::connect_with(
        &endpoint,
        &GrpcAuth {
            token: Some("cloud-token".to_owned()),
            tls: Some(GrpcTls::TrustCert {
                ca_pem: identity.cert_pem.clone(),
                domain: "localhost".to_owned(),
            }),
        },
    )
    .await
    .expect("token + TLS connect must succeed");
    transport.shutdown().await.unwrap();
}
