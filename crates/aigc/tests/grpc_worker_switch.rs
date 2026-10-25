//! Integration tests for gRPC worker switching.
//!
//! A fake gRPC worker (the shared fake service) is spun up in-process,
//! the host connects through `GrpcWorkerCandidate`, and
//! `WorkerSwitchService::drain_and_switch` swaps a local-style worker
//! for the remote gRPC worker.

use std::sync::Arc;
use std::time::Duration;

use aigc::{
    GrpcAuth, GrpcSwitchableWorker, GrpcTls, GrpcWorkerCandidate, GrpcWorkerCandidateConfig,
    RunCancellation, SwitchableWorker, WorkerSwitchError, WorkerSwitchService, WorkerSwitchTarget,
};
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, TerminalOutcome, WorkerIdentity, WorkerIncarnationId, WorkerInstallationId,
    WorkerInstanceProfile, WorkerProfile,
};
use reimagine_backend_worker_transport_grpc::client::WorkerClient;
use reimagine_backend_worker_transport_grpc::proto::worker_server::WorkerServer;
use reimagine_backend_worker_transport_grpc::testing::{FakeWorkerConfig, FakeWorkerService};
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

fn fake_worker_identity() -> WorkerIdentity {
    WorkerIdentity {
        backend_instance_id: BackendInstanceId::from(WORKER_INSTANCE),
        installation_id: WorkerInstallationId::from("grpc-install"),
        incarnation_id: WorkerIncarnationId::from(WORKER_INCARNATION),
        worker_version: "0.1.0".to_owned(),
        backend_kind: "burn".to_owned(),
        target: std::env::consts::ARCH.to_owned(),
        manifest_digest: "test-manifest".to_owned(),
    }
}

fn fake_worker_profile() -> WorkerProfile {
    WorkerProfile {
        instances: vec![WorkerInstanceProfile {
            backend_instance_id: BackendInstanceId::from(WORKER_INSTANCE),
            device_label: "cuda:0".to_owned(),
            capabilities: vec!["load_bundle".to_owned()],
            operation_options: serde_json::json!({}),
        }],
    }
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
    let service = FakeWorkerService::new(FakeWorkerConfig {
        identity: Some(fake_worker_identity()),
        profile: Some(fake_worker_profile()),
        token,
        hello_protocol: 2,
        ..FakeWorkerConfig::default()
    });
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
    let server: WorkerServer<FakeWorkerService> =
        WorkerServer::new(service).max_decoding_message_size(64 * 1024 * 1024);
    let server = builder
        .add_service(server)
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener));

    let handle = tokio::spawn(server);
    (format!("{scheme}://{addr}"), handle)
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

    // Connect through the public client, then wrap the result in the
    // concrete switchable worker so the switched worker's client can be
    // exercised directly.
    let client = Arc::new(
        WorkerClient::connect(&endpoint, &GrpcAuth::plain())
            .await
            .unwrap(),
    );
    let hello = client.hello().await.unwrap();
    let hello = reimagine_backend_worker_transport_grpc::conversion::hello_from_proto(hello)
        .expect("complete hello");
    let grpc_worker = Arc::new(GrpcSwitchableWorker::new(hello, client));

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
    // client and verify the terminal response.
    let payload = serde_json::json!({"job": "denoise", "steps": 20});
    let mut stream = grpc_worker
        .client()
        .execute("echo", serde_json::to_vec(&payload).unwrap())
        .await
        .expect("send request");

    let mut terminal = None;
    while let Ok(Some(event)) = stream.message().await {
        if let Some(reimagine_backend_worker_transport_grpc::proto::execute_event::Event::Result(
            result,
        )) = event.event
        {
            terminal = Some(result);
            break;
        }
    }
    let outcome = reimagine_backend_worker_transport_grpc::conversion::outcome_from_proto(
        terminal.expect("terminal event"),
    )
    .expect("outcome");
    match outcome {
        TerminalOutcome::Success { output } => {
            assert_eq!(output, payload, "echoed payload should match the request");
        }
        other => panic!("expected success outcome, got {other:?}"),
    }

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
        error.to_string().contains("cleartext"),
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

    // Correct token over TLS -> hello succeeds, remote device.
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
async fn grpc_candidate_rejects_wrong_ca() {
    // MITM simulation: the client pins a CA that did not sign the
    // server's certificate — the TLS handshake must fail.
    let server_identity =
        reimagine_backend_worker_transport_grpc::tls::generate_self_signed_identity("localhost")
            .unwrap();
    let attacker_identity =
        reimagine_backend_worker_transport_grpc::tls::generate_self_signed_identity("localhost")
            .unwrap();
    let (endpoint, server_handle) = start_test_grpc_server(
        None,
        Some((server_identity.cert_pem.clone(), server_identity.key_pem)),
    )
    .await;

    let candidate = GrpcWorkerCandidate::new(GrpcWorkerCandidateConfig {
        endpoint,
        auth: GrpcAuth {
            token: None,
            tls: Some(GrpcTls::TrustCert {
                ca_pem: attacker_identity.cert_pem,
                domain: "localhost".to_owned(),
            }),
        },
        connect_retries: 0,
        ..GrpcWorkerCandidateConfig::default()
    });
    let error = match candidate.start().await {
        Ok(_) => panic!("wrong CA must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, WorkerSwitchError::Startup { .. }));

    server_handle.abort();
}

#[tokio::test]
async fn grpc_worker_shutdown_reaches_the_worker() {
    let (endpoint, server_handle) = start_test_grpc_server(None, None).await;

    let client = Arc::new(
        WorkerClient::connect(&endpoint, &GrpcAuth::plain())
            .await
            .unwrap(),
    );
    let hello = client.hello().await.unwrap();
    let hello = reimagine_backend_worker_transport_grpc::conversion::hello_from_proto(hello)
        .expect("complete hello");
    let grpc_worker = GrpcSwitchableWorker::new(hello, client);

    // The shutdown RPC reaches the fake worker and is acknowledged.
    grpc_worker.shutdown().await.expect("shutdown");

    server_handle.abort();
}
