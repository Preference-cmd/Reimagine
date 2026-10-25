//! Integration tests for QUIC remote-worker registration.
//!
//! Remote workers are registration-only today: the QUIC path carries
//! no request protocol, so a switch target connects the certificate-
//! pinned transport and derives its identity from the discovery
//! metadata instead of a worker hello.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use aigc::{
    QuicWorkerCandidate, QuicWorkerCandidateConfig, SwitchableWorker, WorkerSwitchError,
    WorkerSwitchService, WorkerSwitchTarget,
};
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::WorkerIncarnationId;
use reimagine_backend_worker_transport_quic::tls::SelfSignedCert;
use reimagine_core::model::RunId;
use reimagine_inference::{BackendInstance, DeviceKind};

/// A fake worker for testing switch operations.
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

impl aigc::RunCancellation for NoopRunCancellation {
    fn cancel(&self, _run_id: &RunId) -> Result<(), WorkerSwitchError> {
        Ok(())
    }
}

/// Start a QUIC endpoint that accepts connections and holds them until
/// they close (registration-only: no request protocol).
async fn start_test_quic_server(
    cert: &SelfSignedCert,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let endpoint = reimagine_backend_worker_transport_quic::tls::server_endpoint(
        "127.0.0.1:0".parse().unwrap(),
        cert,
    )
    .unwrap();
    let addr = endpoint.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            if let Ok(connection) = incoming.await {
                let _ = connection.closed().await;
            }
        }
    });

    (addr, handle)
}

fn discovered_worker(
    addr: SocketAddr,
) -> reimagine_backend_worker_transport_quic::discovery::DiscoveredWorker {
    let mut props = HashMap::new();
    props.insert("endpoint".to_string(), format!("quic://{addr}"));
    props.insert("backend".to_string(), "burn".to_string());
    props.insert("devices".to_string(), "cuda:0".to_string());
    props.insert("capabilities".to_string(), "load_bundle".to_string());
    reimagine_backend_worker_transport_quic::discovery::DiscoveredWorker {
        id: "test-worker._reimagine-worker._tcp.local.".to_string(),
        addr,
        properties: props,
    }
}

#[tokio::test]
async fn drain_and_switch_to_quic_worker_reports_remote_device() {
    let cert = Arc::new(SelfSignedCert::generate("localhost").unwrap());
    let (server_addr, server_handle) = start_test_quic_server(&cert).await;

    let config = QuicWorkerCandidateConfig {
        server_addr,
        server_name: "localhost".to_string(),
        cert: cert.clone(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
    };

    let candidate = QuicWorkerCandidate::new(config, discovered_worker(server_addr));
    let quic_worker = candidate.start().await.expect("start QUIC worker");

    // Verify the worker reports Remote device kind.
    let snapshot = quic_worker.snapshot().await;
    let device = snapshot.device.expect("device should be present");
    assert_eq!(
        device.kind,
        DeviceKind::Remote,
        "QUIC workers should report DeviceKind::Remote"
    );
    assert!(
        device.label.starts_with("remote:"),
        "device label should start with 'remote:'"
    );

    // Verify the worker is ready.
    assert_eq!(quic_worker.process_state(), WorkerProcessState::Ready);

    // Switch from a local-style worker to the QUIC worker.
    let old = Arc::new(FakeWorker::new("burn:local:default", "old-incarnation"));
    let workers = WorkerSwitchService::new(old, Arc::new(NoopRunCancellation));

    let selected = workers
        .drain_and_switch(
            Arc::new(QuicWorkerTarget(quic_worker)),
            Duration::from_secs(5),
        )
        .await
        .expect("drain and switch");

    // Verify the switch succeeded and the active worker is ready.
    let active = workers.resolve(&selected).await.expect("resolve");
    assert_eq!(
        active.process_state(),
        WorkerProcessState::Ready,
        "QUIC worker should be ready after switch"
    );

    server_handle.abort();
}

#[tokio::test]
async fn quic_worker_snapshot_contains_transport_observation() {
    let cert = Arc::new(SelfSignedCert::generate("localhost").unwrap());
    let (server_addr, server_handle) = start_test_quic_server(&cert).await;

    let config = QuicWorkerCandidateConfig {
        server_addr,
        server_name: "localhost".to_string(),
        cert: cert.clone(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
    };

    let candidate = QuicWorkerCandidate::new(config, discovered_worker(server_addr));
    let quic_worker = candidate.start().await.expect("start QUIC worker");

    let snapshot = quic_worker.snapshot().await;
    assert_eq!(
        snapshot.observations.get("transport").map(|s| s.as_str()),
        Some("quic"),
        "snapshot should report quic transport"
    );
    assert!(
        snapshot.observations.contains_key("remote_addr"),
        "snapshot should include remote_addr"
    );

    server_handle.abort();
}

#[tokio::test]
async fn quic_worker_shutdown_closes_connection() {
    let cert = Arc::new(SelfSignedCert::generate("localhost").unwrap());
    let (server_addr, server_handle) = start_test_quic_server(&cert).await;

    let config = QuicWorkerCandidateConfig {
        server_addr,
        server_name: "localhost".to_string(),
        cert: cert.clone(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
    };

    let candidate = QuicWorkerCandidate::new(config, discovered_worker(server_addr));
    let quic_worker = candidate.start().await.expect("start QUIC worker");

    // Verify shutdown succeeds.
    quic_worker.shutdown().await.expect("shutdown");

    server_handle.abort();
}

/// A wrapper that converts a `SwitchableWorker` into a `WorkerSwitchTarget`.
struct QuicWorkerTarget(Arc<dyn SwitchableWorker>);

#[async_trait::async_trait]
impl WorkerSwitchTarget for QuicWorkerTarget {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        Ok(Arc::clone(&self.0))
    }
}
