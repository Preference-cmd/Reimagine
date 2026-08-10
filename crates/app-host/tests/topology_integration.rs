//! WorkspaceHost topology integration tests (T13).
//!
//! Verifies that a workspace configured with remote workers bootstraps
//! a connection topology manager + discovery orchestrator and exposes
//! registered topology workers in the compute profile DTO — and that a
//! workspace without configured workers stays byte-identical to the
//! single-worker mode (no topology manager at all).

use std::sync::Arc;

use reimagine_agent_harness::WorkspaceScope;
use reimagine_app_host::{EmptyWorkerInventoryProvider, WorkspaceHost};
use reimagine_config::{InferenceBackendConfig, WorkerEndpointConfig};
use reimagine_runtime::VecRunEventSink;

fn temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("reimagine-t13-{prefix}-{nonce}"))
}

fn worker_config(id: &str, transport: &str, address: &str) -> WorkerEndpointConfig {
    WorkerEndpointConfig {
        id: id.to_owned(),
        transport: transport.to_owned(),
        address: address.to_owned(),
        capabilities: vec!["load_bundle".to_owned()],
        device_label: "cuda:0".to_owned(),
        fingerprint: Some("aabbccdd".to_owned()),
    }
}

async fn host_with_backend_config(
    scope: &str,
    base: &std::path::Path,
    backend_config: InferenceBackendConfig,
) -> WorkspaceHost {
    WorkspaceHost::try_with_backend_config_and_worker_inventory(
        WorkspaceScope::new(scope),
        base,
        backend_config,
        Arc::new(VecRunEventSink::new()),
        Arc::new(EmptyWorkerInventoryProvider),
    )
    .await
    .expect("workspace bootstrap")
}

#[tokio::test]
async fn configured_workers_bootstrap_topology_and_profile() {
    let base = temp_dir("topology-on");
    let config = InferenceBackendConfig {
        workers: vec![
            worker_config("quic-worker-a", "quic", "quic://192.168.1.10:9100"),
            worker_config("grpc-cloud", "grpc", "https://cloud.example:50051"),
        ],
        ..InferenceBackendConfig::default()
    };
    let workspace = host_with_backend_config("t13-on", &base, config).await;

    // Topology manager + discovery orchestrator exist.
    let topology = workspace.topology().expect("topology manager");
    let discovery = workspace.discovery().expect("discovery orchestrator");

    // Run one reconcile pass deterministically (the background poll
    // loop also does this, on its own cadence).
    let report = discovery.reconcile_once().await;
    assert_eq!(report.registered, 2, "config endpoints registered");

    // The config endpoints are registered in the pool (mDNS discovery
    // adds nothing in the test environment).
    let guard = topology.try_lock().expect("lock");
    assert_eq!(guard.pool().len(), 2, "configured workers registered");
    assert!(guard.pool().get("quic-worker-a").is_some());
    assert!(guard.pool().get("grpc-cloud").is_some());
    drop(guard);

    // Compute profile DTO exposes the topology workers.
    let dto = workspace.compute_profile_dto();
    assert_eq!(dto.topology_workers.len(), 2);
    let quic = dto
        .topology_workers
        .iter()
        .find(|w| w.id == "quic-worker-a")
        .expect("quic worker in dto");
    assert_eq!(quic.transport, "quic");
    assert_eq!(quic.address, "quic://192.168.1.10:9100");
    // Config endpoints with a pre-shared fingerprint are trusted.
    assert!(quic.trusted);
    let grpc = dto
        .topology_workers
        .iter()
        .find(|w| w.id == "grpc-cloud")
        .expect("grpc worker in dto");
    assert_eq!(grpc.transport, "grpc");
    assert!(grpc.trusted);
}

#[tokio::test]
async fn no_configured_workers_means_no_topology() {
    let base = temp_dir("topology-off");
    let workspace =
        host_with_backend_config("t13-off", &base, InferenceBackendConfig::default()).await;

    // Graceful degradation: no topology manager, no orchestrator.
    assert!(workspace.topology().is_none());
    assert!(workspace.discovery().is_none());

    // The compute profile DTO carries no topology workers (and still
    // round-trips: the field is absent/empty, not an error).
    let dto = workspace.compute_profile_dto();
    assert!(dto.topology_workers.is_empty());
}

#[tokio::test]
async fn untrusted_worker_surfaces_as_untrusted_in_profile() {
    let base = temp_dir("topology-untrusted");
    let config = InferenceBackendConfig {
        workers: vec![WorkerEndpointConfig {
            id: "mdns-like".to_owned(),
            transport: "quic".to_owned(),
            address: "quic://192.168.1.99:9100".to_owned(),
            capabilities: Vec::new(),
            device_label: String::new(),
            // No pre-shared fingerprint: registered but untrusted
            // (pending the T19 trust flow).
            fingerprint: None,
        }],
        ..InferenceBackendConfig::default()
    };
    let workspace = host_with_backend_config("t13-untrusted", &base, config).await;
    let discovery = workspace.discovery().expect("discovery orchestrator");
    discovery.reconcile_once().await;

    let dto = workspace.compute_profile_dto();
    let worker = dto
        .topology_workers
        .iter()
        .find(|w| w.id == "mdns-like")
        .expect("worker in dto");
    assert!(!worker.trusted, "no pin means untrusted");
    assert_eq!(worker.state, "Connecting");
}
