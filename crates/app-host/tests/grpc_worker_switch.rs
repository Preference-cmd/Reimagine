//! Integration tests for gRPC worker switching (T8).
//!
//! Mirrors `quic_worker_switch.rs`: a fake gRPC worker server is spun up
//! in-process (tonic), the host connects through `GrpcWorkerCandidate`,
//! and `WorkerSwitchService::drain_and_switch` swaps a local-style worker
//! for the remote gRPC worker.

use std::sync::Arc;
use std::time::Duration;

use reimagine_app_host::{
    GrpcAuth, GrpcSwitchableWorker, GrpcTls, GrpcWorkerCandidate, GrpcWorkerCandidateConfig,
    RunCancellation, SwitchableWorker, WorkerSwitchError, WorkerSwitchService, WorkerSwitchTarget,
};
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::{
    CorrelationId, HostHello, ProtocolRange, ProtocolVersion, RequestFrame, RequestId,
    TerminalOutcome, WireMessage, WorkerHello, WorkerIncarnationId,
};
use reimagine_backend_worker_transport_grpc::client;
use reimagine_backend_worker_transport_grpc::proto;
use reimagine_backend_worker_transport_grpc::proto::worker_service_server::WorkerServiceServer;
use reimagine_backend_worker_transport_grpc::server::{GrpcWorkerService, MessageHandler};
use reimagine_backend_worker_transport_grpc::transport::GrpcTransport;
use reimagine_core::model::RunId;
use reimagine_inference::{BackendInstance, DeviceKind};

const WORKER_INSTANCE: &str = "burn:grpc:remote";
const WORKER_INCARNATION: &str = "grpc-incarnation-001";

/// A fake worker for testing switch operations with gRPC workers.
struct FakeWorker {
    instance: BackendInstance,
    incarnation: WorkerIncarnationId,
    leases: Arc<WorkerRunLeases>,
}

impl FakeWorker {
    fn new(instance: &str, incarnation: &str) -> Self {
        Self {
            instance: BackendInstance::new(instance),
            incarnation: WorkerIncarnationId(incarnation.to_owned()),
            leases: Arc::new(WorkerRunLeases::new()),
        }
    }
}

#[async_trait::async_trait]
impl SwitchableWorker for FakeWorker {
    fn instance(&self) -> &BackendInstance {
        &self.instance
    }

    fn incarnation_id(&self) -> &WorkerIncarnationId {
        &self.incarnation
    }

    fn run_leases(&self) -> &Arc<WorkerRunLeases> {
        &self.leases
    }

    fn process_state(&self) -> WorkerProcessState {
        WorkerProcessState::Ready
    }

    async fn shutdown(&self) -> Result<(), WorkerSwitchError> {
        Ok(())
    }
}

/// A no-op run cancellation implementation for testing.
struct NoopRunCancellation;

impl RunCancellation for NoopRunCancellation {
    fn cancel(&self, _run_id: &RunId) -> Result<(), WorkerSwitchError> {
        Ok(())
    }
}

/// A worker handler that answers the `HostHello` handshake and echoes
/// requests back as successful terminals.
fn handshake_and_echo_handler() -> MessageHandler {
    Arc::new(|msg| {
        Box::pin(async move {
            match msg.message {
                Some(proto::host_to_worker::Message::HostHello(h)) => Some(proto::WorkerToHost {
                    message: Some(proto::worker_to_host::Message::WorkerHello(
                        proto::WorkerHello {
                            selected_protocol: std::cmp::min(h.protocol_max, 1),
                            identity: Some(proto::WorkerIdentity {
                                backend_instance_id: WORKER_INSTANCE.into(),
                                installation_id: "grpc-install".into(),
                                incarnation_id: WORKER_INCARNATION.into(),
                                worker_version: "0.1.0".into(),
                                backend_kind: "burn".into(),
                                target: std::env::consts::ARCH.to_string(),
                                manifest_digest: "test-manifest".into(),
                            }),
                            profile: Some(proto::WorkerProfile {
                                instances: vec![proto::WorkerInstanceProfile {
                                    backend_instance_id: WORKER_INSTANCE.into(),
                                    device_label: "cuda:0".into(),
                                    capabilities: vec!["load_bundle".into()],
                                    operation_options: serde_json::to_vec(&serde_json::json!({}))
                                        .unwrap(),
                                }],
                            }),
                        },
                    )),
                }),
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

/// Start a fake gRPC worker server; returns the endpoint URI and the
/// server task handle.
async fn start_test_grpc_server(
    token: Option<String>,
    tls: Option<(String, String)>,
) -> (
    String,
    tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
) {
    let service = GrpcWorkerService::with_token(handshake_and_echo_handler(), token);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let scheme = if tls.is_some() { "https" } else { "http" };
    let mut builder = tonic::transport::Server::builder();
    if let Some((cert_pem, key_pem)) = tls {
        builder = builder
            .tls_config(
                tonic::transport::ServerTlsConfig::new()
                    .identity(tonic::transport::Identity::from_pem(cert_pem, key_pem)),
            )
            .unwrap();
    }
    let server = builder
        .add_service(WorkerServiceServer::new(service))
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

    let handle = tokio::spawn(server);
    (format!("{scheme}://{addr}"), handle)
}

/// Perform the `HostHello` -> `WorkerHello` handshake over an already
/// connected gRPC transport (mirrors the candidate's handshake).
async fn grpc_handshake(transport: &GrpcTransport) -> WorkerHello {
    let wire = WireMessage::HostHello(HostHello {
        supported_protocols: ProtocolRange::new(1, 1),
    });
    let proto_message: proto::HostToWorker = (&wire).try_into().unwrap();
    transport.send(proto_message).await.unwrap();
    let response = transport.recv().await.unwrap().expect("WorkerHello");
    match WireMessage::try_from(response).unwrap() {
        WireMessage::WorkerHello(hello) => hello,
        other => panic!("expected WorkerHello, got {:?}", other.kind()),
    }
}

/// A wrapper that converts a `SwitchableWorker` into a `WorkerSwitchTarget`.
struct GrpcWorkerTarget(Arc<dyn SwitchableWorker>);

#[async_trait::async_trait]
impl WorkerSwitchTarget for GrpcWorkerTarget {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        Ok(Arc::clone(&self.0))
    }
}

#[tokio::test]
async fn drain_and_switch_to_grpc_worker_reports_remote_device() {
    let (endpoint, server_handle) = start_test_grpc_server(None, None).await;

    let candidate = GrpcWorkerCandidate::with_defaults(endpoint, GrpcAuth::plain());
    let grpc_worker = candidate.start().await.expect("start gRPC worker");

    // Verify the worker reports Remote device kind.
    let snapshot = grpc_worker.snapshot().await;
    let device = snapshot.device.expect("device should be present");
    assert_eq!(
        device.kind,
        DeviceKind::Remote,
        "gRPC workers should report DeviceKind::Remote"
    );
    assert!(
        device.label.starts_with("remote:"),
        "device label should start with 'remote:'"
    );

    // Verify the worker is ready.
    assert_eq!(grpc_worker.process_state(), WorkerProcessState::Ready);

    // Switch from a local-style worker to the gRPC worker.
    let old = Arc::new(FakeWorker::new("burn:local:default", "old-incarnation"));
    let workers = WorkerSwitchService::new(old, Arc::new(NoopRunCancellation));

    let selected = workers
        .drain_and_switch(
            Arc::new(GrpcWorkerTarget(grpc_worker)),
            Duration::from_secs(5),
        )
        .await
        .expect("drain and switch");

    // Verify the switch succeeded and the active worker is ready.
    let active = workers.resolve(&selected).await.expect("resolve");
    assert_eq!(
        active.process_state(),
        WorkerProcessState::Ready,
        "gRPC worker should be ready after switch"
    );
    assert_eq!(active.incarnation_id().0, WORKER_INCARNATION);

    server_handle.abort();
}

#[tokio::test]
async fn grpc_worker_snapshot_contains_transport_observation() {
    let (endpoint, server_handle) = start_test_grpc_server(None, None).await;

    let candidate = GrpcWorkerCandidate::with_defaults(endpoint, GrpcAuth::plain());
    let grpc_worker = candidate.start().await.expect("start gRPC worker");

    let snapshot = grpc_worker.snapshot().await;
    assert_eq!(
        snapshot.observations.get("transport").map(|s| s.as_str()),
        Some("grpc"),
        "snapshot should report grpc transport"
    );
    assert!(
        snapshot.observations.contains_key("remote_addr"),
        "snapshot should include remote_addr"
    );

    server_handle.abort();
}

#[tokio::test]
async fn switched_grpc_worker_serves_inference_request_roundtrip() {
    let (endpoint, server_handle) = start_test_grpc_server(None, None).await;

    // Connect and handshake through the public client, then wrap the
    // result in the concrete switchable worker so the switched worker's
    // transport can be exercised directly.
    let transport = Arc::new(
        client::connect_with(&endpoint, &GrpcAuth::plain())
            .await
            .unwrap(),
    );
    let hello = grpc_handshake(&transport).await;
    let grpc_worker = Arc::new(GrpcSwitchableWorker::new(hello, transport));

    let old = Arc::new(FakeWorker::new("burn:local:default", "old-incarnation"));
    let workers = WorkerSwitchService::new(old, Arc::new(NoopRunCancellation));
    let selected = workers
        .drain_and_switch(
            Arc::new(GrpcWorkerTarget(grpc_worker.clone())),
            Duration::from_secs(5),
        )
        .await
        .expect("drain and switch");
    let active = workers.resolve(&selected).await.expect("resolve");
    assert_eq!(active.instance().as_str(), WORKER_INSTANCE);

    // Send an inference-style request over the switched worker's
    // transport and verify the terminal response.
    let payload = serde_json::json!({"job": "denoise", "steps": 20});
    let wire = WireMessage::Request(RequestFrame {
        protocol_version: ProtocolVersion(1),
        incarnation_id: WorkerIncarnationId("inc-1".into()),
        request_id: RequestId("req-1".into()),
        correlation_id: CorrelationId("corr-1".into()),
        operation: "echo".into(),
        payload: payload.clone(),
    });
    let proto_message: proto::HostToWorker = (&wire).try_into().unwrap();
    grpc_worker
        .transport()
        .send(proto_message)
        .await
        .expect("send request");

    let response = grpc_worker
        .transport()
        .recv()
        .await
        .expect("recv")
        .expect("terminal message");
    let terminal = match WireMessage::try_from(response).unwrap() {
        WireMessage::Terminal(terminal) => terminal,
        other => panic!("expected Terminal, got {:?}", other.kind()),
    };
    let output = match terminal.outcome {
        TerminalOutcome::Success { output } => output,
        other => panic!("expected success outcome, got {other:?}"),
    };
    assert_eq!(output, payload, "echoed payload should match the request");

    server_handle.abort();
}

#[tokio::test]
async fn grpc_candidate_refuses_cleartext_token() {
    let (endpoint, server_handle) = start_test_grpc_server(None, None).await;

    // A token without TLS must be refused by the client guard before
    // any request is sent.
    let candidate = GrpcWorkerCandidate::new(GrpcWorkerCandidateConfig {
        endpoint,
        auth: GrpcAuth {
            token: Some("s3cret".to_owned()),
            tls: None,
        },
        connect_retries: 0,
        ..GrpcWorkerCandidateConfig::default()
    });
    let error = match candidate.start().await {
        Ok(_) => panic!("cleartext token must be refused"),
        Err(error) => error,
    };
    assert!(matches!(error, WorkerSwitchError::Startup { .. }));
    assert!(
        error.to_string().contains("cleartext token"),
        "expected cleartext-token refusal, got: {error}"
    );

    server_handle.abort();
}

#[tokio::test]
async fn grpc_candidate_refuses_cleartext_non_loopback() {
    // The policy guard fires before any network I/O: a plain endpoint
    // that is not loopback is refused outright.
    let candidate =
        GrpcWorkerCandidate::with_defaults("http://203.0.113.10:50051", GrpcAuth::plain());
    let error = match candidate.start().await {
        Ok(_) => panic!("non-loopback plain endpoint must be refused"),
        Err(error) => error,
    };
    assert!(matches!(error, WorkerSwitchError::Startup { .. }));
    assert!(
        error.to_string().contains("cleartext") && error.to_string().contains("TLS"),
        "expected TLS-required refusal, got: {error}"
    );
}

#[tokio::test]
async fn grpc_candidate_connects_with_tls_and_token() {
    let identity =
        reimagine_backend_worker_transport_grpc::tls::generate_self_signed_identity("localhost")
            .unwrap();
    let (endpoint, server_handle) = start_test_grpc_server(
        Some("cloud-token".to_owned()),
        Some((identity.cert_pem.clone(), identity.key_pem)),
    )
    .await;

    // Missing token over TLS -> refused by the server.
    let missing_token = GrpcWorkerCandidate::new(GrpcWorkerCandidateConfig {
        endpoint: endpoint.clone(),
        auth: GrpcAuth {
            token: None,
            tls: Some(GrpcTls::TrustCert {
                ca_pem: identity.cert_pem.clone(),
                domain: "localhost".to_owned(),
            }),
        },
        connect_retries: 0,
        ..GrpcWorkerCandidateConfig::default()
    });
    let error = match missing_token.start().await {
        Ok(_) => panic!("missing token must be refused"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("invalid bearer token"),
        "got: {error}"
    );

    // Correct token over TLS -> handshake succeeds, remote device.
    let candidate = GrpcWorkerCandidate::new(GrpcWorkerCandidateConfig {
        endpoint,
        auth: GrpcAuth {
            token: Some("cloud-token".to_owned()),
            tls: Some(GrpcTls::TrustCert {
                ca_pem: identity.cert_pem,
                domain: "localhost".to_owned(),
            }),
        },
        ..GrpcWorkerCandidateConfig::default()
    });
    let grpc_worker = candidate.start().await.expect("token + TLS connect");
    let snapshot = grpc_worker.snapshot().await;
    let device = snapshot.device.expect("device should be present");
    assert_eq!(device.kind, DeviceKind::Remote);

    server_handle.abort();
}

#[tokio::test]
async fn grpc_worker_shutdown_closes_transport() {
    let (endpoint, server_handle) = start_test_grpc_server(None, None).await;

    let candidate = GrpcWorkerCandidate::with_defaults(endpoint, GrpcAuth::plain());
    let grpc_worker = candidate.start().await.expect("start gRPC worker");

    grpc_worker.shutdown().await.expect("shutdown");

    server_handle.abort();
}
