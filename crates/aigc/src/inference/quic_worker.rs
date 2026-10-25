use std::sync::Arc;

use async_trait::async_trait;
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::WorkerIncarnationId;
use reimagine_backend_worker_transport_quic::{
    QuicTransport, discovery::DiscoveredWorker, tls::SelfSignedCert,
};
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceSnapshot, DeviceProfile, InferenceBackend,
};

use super::switch::{SwitchableWorker, WorkerSwitchError, WorkerSwitchTarget};

/// A remote QUIC worker registered through discovery, implementing
/// [`SwitchableWorker`].
///
/// Remote workers are registration-only today: the QUIC path carries no
/// request protocol, so identity and profile come from the discovery
/// metadata (mDNS TXT records) rather than a worker hello.
pub struct QuicSwitchableWorker {
    instance: BackendInstance,
    incarnation_id: WorkerIncarnationId,
    run_leases: Arc<WorkerRunLeases>,
    device_label: String,
    backend_kind: String,
    endpoint: String,
    transport: Arc<QuicTransport>,
}

impl QuicSwitchableWorker {
    /// Create a new `QuicSwitchableWorker` from a discovered worker and
    /// a connected transport.
    pub fn new(discovered: &DiscoveredWorker, transport: Arc<QuicTransport>) -> Self {
        let backend_kind = discovered
            .properties
            .get("backend")
            .cloned()
            .unwrap_or_else(|| "burn".to_owned());
        let device_label = discovered
            .properties
            .get("devices")
            .and_then(|devices| devices.split(',').next())
            .filter(|label| !label.is_empty())
            .unwrap_or("remote")
            .to_owned();
        Self {
            instance: BackendInstance::new(format!("{backend_kind}:remote:{}", discovered.addr)),
            incarnation_id: WorkerIncarnationId(discovered.id.clone()),
            run_leases: Arc::new(WorkerRunLeases::new()),
            device_label,
            backend_kind,
            endpoint: transport.description().endpoint,
            transport,
        }
    }

    /// Access the underlying QUIC transport.
    pub fn transport(&self) -> &Arc<QuicTransport> {
        &self.transport
    }
}

#[async_trait]
impl SwitchableWorker for QuicSwitchableWorker {
    fn instance(&self) -> &BackendInstance {
        &self.instance
    }

    fn incarnation_id(&self) -> &WorkerIncarnationId {
        &self.incarnation_id
    }

    fn run_leases(&self) -> &Arc<WorkerRunLeases> {
        &self.run_leases
    }

    fn process_state(&self) -> WorkerProcessState {
        // Registration-only workers are always reported ready once the
        // transport connects.
        WorkerProcessState::Ready
    }

    fn inference_backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        None // remote workers do not expose a local inference backend
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        BackendInstanceSnapshot {
            backend_instance: self.instance.clone(),
            backend: Backend::new(self.backend_kind.clone()),
            plugin: None,
            extension: None,
            // Remote workers always report DeviceKind::Remote
            device: Some(DeviceProfile::new(format!("remote:{}", self.device_label))),
            observations: std::collections::BTreeMap::from([
                ("transport".to_owned(), "quic".to_owned()),
                ("remote_addr".to_owned(), self.endpoint.clone()),
            ]),
            diagnostics: Vec::new(),
        }
    }

    async fn shutdown(&self) -> Result<(), WorkerSwitchError> {
        self.transport.close();
        Ok(())
    }
}

/// Configuration for connecting to a remote QUIC worker.
#[derive(Clone)]
pub struct QuicWorkerCandidateConfig {
    /// The server address to connect to.
    pub server_addr: std::net::SocketAddr,
    /// The server name for TLS verification.
    pub server_name: String,
    /// The certificate to trust.
    pub cert: Arc<SelfSignedCert>,
    /// Optional bind address for the client socket.
    pub bind_addr: std::net::SocketAddr,
}

impl std::fmt::Debug for QuicWorkerCandidateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuicWorkerCandidateConfig")
            .field("server_addr", &self.server_addr)
            .field("server_name", &self.server_name)
            .field("cert", &"<SelfSignedCert>")
            .field("bind_addr", &self.bind_addr)
            .finish()
    }
}

/// A candidate that connects to a remote QUIC worker.
///
/// Implements [`WorkerSwitchTarget`]. Remote workers are
/// registration-only today, so the candidate connects the transport
/// (certificate pinned) but there is no request protocol over it.
pub struct QuicWorkerCandidate {
    config: QuicWorkerCandidateConfig,
    discovered: DiscoveredWorker,
}

impl QuicWorkerCandidate {
    /// Create a new candidate from a discovered worker.
    pub fn new(config: QuicWorkerCandidateConfig, discovered: DiscoveredWorker) -> Self {
        Self { config, discovered }
    }

    /// Connect the QUIC transport with the pinned certificate.
    async fn connect(&self) -> Result<Arc<QuicTransport>, WorkerSwitchError> {
        let transport = QuicTransport::connect(
            self.config.bind_addr,
            self.config.server_addr,
            &self.config.server_name,
            &self.config.cert,
        )
        .await
        .map_err(|error| WorkerSwitchError::Startup {
            message: format!("QUIC connect failed: {error}"),
        })?;
        Ok(Arc::new(transport))
    }
}

#[async_trait]
impl WorkerSwitchTarget for QuicWorkerCandidate {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        let transport = self.connect().await?;
        Ok(Arc::new(QuicSwitchableWorker::new(
            &self.discovered,
            transport,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::net::{Ipv4Addr, SocketAddr};

    fn test_discovered_worker(endpoint: &str) -> DiscoveredWorker {
        let mut props = HashMap::new();
        props.insert("endpoint".to_string(), format!("quic://{endpoint}"));
        props.insert("backend".to_string(), "burn".to_string());
        props.insert("devices".to_string(), "cuda:0".to_string());
        props.insert("capabilities".to_string(), "load_bundle".to_string());

        DiscoveredWorker {
            id: "test-worker._reimagine-worker._tcp.local.".to_string(),
            addr: endpoint.parse().unwrap(),
            properties: props,
        }
    }

    #[test]
    fn quic_worker_candidate_config_is_clone() {
        let cert = Arc::new(SelfSignedCert::generate("localhost").unwrap());
        let config = QuicWorkerCandidateConfig {
            server_addr: "127.0.0.1:9100".parse().unwrap(),
            server_name: "localhost".to_string(),
            cert,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
        };
        let _ = config.clone();
    }

    #[test]
    fn discovered_worker_device_label_used() {
        let discovered = test_discovered_worker("192.168.1.100:9100");
        let transport_device = discovered
            .properties
            .get("devices")
            .and_then(|devices| devices.split(',').next())
            .unwrap_or("remote");
        assert_eq!(transport_device, "cuda:0");
    }

    #[tokio::test]
    async fn switchable_worker_derives_identity_from_discovery() {
        let cert = Arc::new(SelfSignedCert::generate("localhost").unwrap());
        let listen_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0);
        let endpoint =
            reimagine_backend_worker_transport_quic::tls::server_endpoint(listen_addr, &cert)
                .unwrap();
        let server_addr = endpoint.local_addr().unwrap();
        let server = tokio::spawn(async move {
            while let Some(incoming) = endpoint.accept().await {
                if let Ok(connection) = incoming.await {
                    let _ = connection.closed().await;
                }
            }
        });
        let transport = QuicTransport::connect(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            server_addr,
            "localhost",
            &cert,
        )
        .await
        .unwrap();
        let discovered = test_discovered_worker(&server_addr.to_string());
        let worker = QuicSwitchableWorker::new(&discovered, Arc::new(transport));
        assert!(worker.instance().as_str().starts_with("burn:remote:"));
        let snapshot = worker.snapshot().await;
        let device = snapshot.device.expect("device present");
        assert!(
            device.label.starts_with("remote:"),
            "device label should start with 'remote:'"
        );
        worker.shutdown().await.unwrap();
        server.abort();
    }
}
