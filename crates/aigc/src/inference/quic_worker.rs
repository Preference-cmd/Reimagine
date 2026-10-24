use std::sync::Arc;

use async_trait::async_trait;
use reimagine_backend_worker_host::{WorkerProcessState, WorkerRunLeases};
use reimagine_backend_worker_protocol::{WorkerHello, WorkerIncarnationId, WorkerTransport};
use reimagine_backend_worker_transport_quic::{
    QuicTransport, discovery::DiscoveredWorker, tls::SelfSignedCert,
};
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceSnapshot, DeviceProfile, InferenceBackend,
};

use super::switch::{SwitchableWorker, WorkerSwitchError, WorkerSwitchTarget};

/// A worker connected via QUIC, implementing [`SwitchableWorker`].
///
/// This wraps a QUIC-connected remote worker and provides the same
/// interface as [`ProcessSwitchableWorker`] for use with
/// [`WorkerSwitchService`].
pub struct QuicSwitchableWorker {
    instance: BackendInstance,
    incarnation_id: WorkerIncarnationId,
    run_leases: Arc<WorkerRunLeases>,
    device_label: String,
    hello: WorkerHello,
    transport: Arc<QuicTransport>,
}

impl QuicSwitchableWorker {
    /// Create a new `QuicSwitchableWorker` from a discovered worker and handshake.
    pub fn new(
        discovered: &DiscoveredWorker,
        hello: WorkerHello,
        transport: Arc<QuicTransport>,
    ) -> Self {
        let device_label = discovered
            .devices()
            .first()
            .unwrap_or(&"remote")
            .to_string();
        Self {
            instance: BackendInstance::new(hello.identity.backend_instance_id.0.clone()),
            incarnation_id: hello.identity.incarnation_id.clone(),
            run_leases: Arc::new(WorkerRunLeases::new()),
            device_label,
            hello,
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
        // QUIC workers are always ready once connected
        WorkerProcessState::Ready
    }

    fn inference_backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        None // QUIC workers don't expose a local inference backend
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        BackendInstanceSnapshot {
            backend_instance: self.instance.clone(),
            backend: Backend::new(self.hello.identity.backend_kind.clone()),
            plugin: None,
            extension: None,
            // Remote workers always report DeviceKind::Remote
            device: Some(DeviceProfile::new(format!("remote:{}", self.device_label))),
            observations: std::collections::BTreeMap::from([
                ("transport".to_owned(), "quic".to_owned()),
                (
                    "remote_addr".to_owned(),
                    self.transport.description().endpoint.clone(),
                ),
            ]),
            diagnostics: Vec::new(),
        }
    }

    async fn shutdown(&self) -> Result<(), WorkerSwitchError> {
        self.transport
            .shutdown()
            .await
            .map_err(|error| WorkerSwitchError::Shutdown {
                instance: self.instance.clone(),
                message: error.to_string(),
            })
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

/// A candidate that connects to a remote worker via QUIC.
///
/// Implements [`WorkerSwitchTarget`] to enable switching from a local
/// stdio worker to a remote QUIC worker.
pub struct QuicWorkerCandidate {
    config: QuicWorkerCandidateConfig,
    discovered: DiscoveredWorker,
}

impl QuicWorkerCandidate {
    /// Create a new candidate from a discovered worker.
    pub fn new(config: QuicWorkerCandidateConfig, discovered: DiscoveredWorker) -> Self {
        Self { config, discovered }
    }

    /// Perform the QUIC handshake and return the transport and hello.
    async fn connect(&self) -> Result<(Arc<QuicTransport>, WorkerHello), WorkerSwitchError> {
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

        // Perform the handshake: send HostHello, receive WorkerHello
        let (mut send, mut recv) =
            transport
                .open_bi()
                .await
                .map_err(|error| WorkerSwitchError::Startup {
                    message: format!("QUIC open_bi failed: {error}"),
                })?;

        // Send HostHello
        use reimagine_backend_worker_protocol::{HostHello, ProtocolRange, WireMessage};
        let host_hello = HostHello {
            supported_protocols: ProtocolRange::new(1, 1),
        };
        let hello_json =
            serde_json::to_vec(&WireMessage::HostHello(host_hello)).map_err(|error| {
                WorkerSwitchError::Startup {
                    message: format!("serialize HostHello: {error}"),
                }
            })?;
        send.write_all(&(hello_json.len() as u32).to_be_bytes())
            .await
            .map_err(|error| WorkerSwitchError::Startup {
                message: format!("write hello length: {error}"),
            })?;
        send.write_all(&hello_json)
            .await
            .map_err(|error| WorkerSwitchError::Startup {
                message: format!("write hello payload: {error}"),
            })?;

        // Read WorkerHello
        let mut prefix = [0u8; 4];
        recv.read_exact(&mut prefix)
            .await
            .map_err(|error| WorkerSwitchError::Startup {
                message: format!("read WorkerHello prefix: {error}"),
            })?;
        let len = u32::from_be_bytes(prefix) as usize;
        let mut payload = vec![0u8; len];
        recv.read_exact(&mut payload)
            .await
            .map_err(|error| WorkerSwitchError::Startup {
                message: format!("read WorkerHello payload: {error}"),
            })?;
        let worker_hello: WorkerHello = match serde_json::from_slice::<WireMessage>(&payload)
            .map_err(|error| WorkerSwitchError::Startup {
                message: format!("deserialize WorkerHello: {error}"),
            })? {
            WireMessage::WorkerHello(h) => h,
            other => {
                return Err(WorkerSwitchError::Startup {
                    message: format!("expected WorkerHello, got {:?}", other.kind()),
                });
            }
        };

        Ok((Arc::new(transport), worker_hello))
    }
}

#[async_trait]
impl WorkerSwitchTarget for QuicWorkerCandidate {
    async fn start(&self) -> Result<Arc<dyn SwitchableWorker>, WorkerSwitchError> {
        let (transport, hello) = self.connect().await?;
        Ok(Arc::new(QuicSwitchableWorker::new(
            &self.discovered,
            hello,
            transport,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
        assert_eq!(discovered.devices(), vec!["cuda:0"]);
    }
}
