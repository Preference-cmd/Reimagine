use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use reimagine_app_host::{
    QuicWorkerCandidate, QuicWorkerCandidateConfig, SwitchableWorker, WorkerSwitchError,
    WorkerSwitchService, WorkerSwitchTarget,
};
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::{
    BackendInstanceId, HostHello, ProtocolRange, WireMessage, WorkerHello, WorkerIdentity,
    WorkerIncarnationId, WorkerInstallationId, WorkerInstanceProfile, WorkerProfile,
};
use reimagine_backend_worker_transport_quic::tls::SelfSignedCert;
use reimagine_core::model::RunId;
use reimagine_inference::{BackendInstance, DeviceKind};

/// A fake worker for testing switch operations with QUIC workers.
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

impl reimagine_app_host::RunCancellation for NoopRunCancellation {
    fn cancel(&self, _run_id: &RunId) -> Result<(), WorkerSwitchError> {
        Ok(())
    }
}

/// Start a minimal QUIC "server" that accepts one connection,
/// reads HostHello, sends WorkerHello, and keeps the connection alive.
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
        let incoming = endpoint.accept().await.expect("accept incoming");
        let connection = incoming.await.expect("accept connection");
        let (mut send, mut recv) = connection.accept_bi().await.expect("accept bi");

        // Read HostHello from client
        let mut prefix = [0u8; 4];
        recv.read_exact(&mut prefix).await.expect("read prefix");
        let len = u32::from_be_bytes(prefix) as usize;
        let mut payload = vec![0u8; len];
        recv.read_exact(&mut payload).await.expect("read payload");
        let _host_hello: HostHello = match serde_json::from_slice::<WireMessage>(&payload).unwrap()
        {
            WireMessage::HostHello(h) => h,
            other => panic!("expected HostHello, got {:?}", other.kind()),
        };

        // Send WorkerHello back
        let worker_hello = WorkerHello {
            selected_protocol: ProtocolRange::new(1, 1).minimum,
            identity: WorkerIdentity {
                backend_instance_id: BackendInstanceId::from("burn:remote:default"),
                installation_id: WorkerInstallationId::from("remote-install"),
                incarnation_id: WorkerIncarnationId("remote-incarnation-001".to_string()),
                worker_version: "0.1.0".to_string(),
                backend_kind: "burn".to_string(),
                target: std::env::consts::ARCH.to_string(),
                manifest_digest: "test-manifest".to_string(),
            },
            profile: WorkerProfile {
                instances: vec![WorkerInstanceProfile {
                    backend_instance_id: BackendInstanceId::from("burn:remote:default"),
                    device_label: "cuda:0".to_string(),
                    capabilities: vec!["load_bundle".to_string()],
                    operation_options: serde_json::json!({}),
                }],
            },
        };

        let hello_json = serde_json::to_vec(&WireMessage::WorkerHello(worker_hello)).unwrap();
        send.write_all(&(hello_json.len() as u32).to_be_bytes())
            .await
            .unwrap();
        send.write_all(&hello_json).await.unwrap();
        send.finish().unwrap();

        // Keep connection alive until client disconnects
        let _ = connection.closed().await;
    });

    (addr, handle)
}

#[tokio::test]
async fn drain_and_switch_to_quic_worker_reports_remote_device() {
    let cert = Arc::new(SelfSignedCert::generate("localhost").unwrap());
    let (server_addr, server_handle) = start_test_quic_server(&cert).await;

    // Give server a moment to start listening
    tokio::time::sleep(Duration::from_millis(50)).await;

    let config = QuicWorkerCandidateConfig {
        server_addr,
        server_name: "localhost".to_string(),
        cert: cert.clone(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
    };

    let mut props = HashMap::new();
    props.insert("endpoint".to_string(), format!("quic://{server_addr}"));
    props.insert("backend".to_string(), "burn".to_string());
    props.insert("devices".to_string(), "cuda:0".to_string());
    props.insert("capabilities".to_string(), "load_bundle".to_string());
    let discovered = reimagine_backend_worker_transport_quic::discovery::DiscoveredWorker {
        id: "test-worker._reimagine-worker._tcp.local.".to_string(),
        addr: server_addr,
        properties: props,
    };

    let candidate = QuicWorkerCandidate::new(config, discovered);
    let quic_worker = candidate.start().await.expect("start QUIC worker");

    // Verify the worker reports Remote device kind
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

    // Verify the worker is ready
    assert_eq!(quic_worker.process_state(), WorkerProcessState::Ready);

    // Verify we can use it with WorkerSwitchService
    let old = Arc::new(FakeWorker::new("burn:local:default", "old-incarnation"));
    let workers = WorkerSwitchService::new(old, Arc::new(NoopRunCancellation));

    // Wrap QUIC worker in a target for drain_and_switch
    let quic_target = QuicWorkerTarget(quic_worker);

    // Switch from old worker to QUIC worker
    let selected = workers
        .drain_and_switch(Arc::new(quic_target), Duration::from_secs(5))
        .await
        .expect("drain and switch");

    // Verify the switch succeeded
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

    tokio::time::sleep(Duration::from_millis(50)).await;

    let config = QuicWorkerCandidateConfig {
        server_addr,
        server_name: "localhost".to_string(),
        cert: cert.clone(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
    };

    let mut props = HashMap::new();
    props.insert("endpoint".to_string(), format!("quic://{server_addr}"));
    props.insert("backend".to_string(), "burn".to_string());
    props.insert("devices".to_string(), "cuda:0".to_string());
    props.insert("capabilities".to_string(), "load_bundle".to_string());
    let discovered = reimagine_backend_worker_transport_quic::discovery::DiscoveredWorker {
        id: "test-worker._reimagine-worker._tcp.local.".to_string(),
        addr: server_addr,
        properties: props,
    };

    let candidate = QuicWorkerCandidate::new(config, discovered);
    let quic_worker = candidate.start().await.expect("start QUIC worker");

    // Verify snapshot contains transport observation
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

    tokio::time::sleep(Duration::from_millis(50)).await;

    let config = QuicWorkerCandidateConfig {
        server_addr,
        server_name: "localhost".to_string(),
        cert: cert.clone(),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
    };

    let mut props = HashMap::new();
    props.insert("endpoint".to_string(), format!("quic://{server_addr}"));
    props.insert("backend".to_string(), "burn".to_string());
    props.insert("devices".to_string(), "cuda:0".to_string());
    props.insert("capabilities".to_string(), "load_bundle".to_string());
    let discovered = reimagine_backend_worker_transport_quic::discovery::DiscoveredWorker {
        id: "test-worker._reimagine-worker._tcp.local.".to_string(),
        addr: server_addr,
        properties: props,
    };

    let candidate = QuicWorkerCandidate::new(config, discovered);
    let quic_worker = candidate.start().await.expect("start QUIC worker");

    // Verify shutdown succeeds
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
