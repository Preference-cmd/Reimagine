//! Pluggable worker discovery orchestration (T12).
//!
//! A [`DiscoveryOrchestrator`] aggregates multiple [`WorkerDiscovery`]
//! sources (mDNS, manual config) and drives dynamic worker
//! registration/deregistration in the [`ConnectionTopologyManager`]:
//! endpoints are polled at a fixed interval, newly discovered workers
//! are registered, and workers that vanished (missed mDNS removal
//! events, host restarts) are deregistered.
//!
//! # Trust gate (T19)
//!
//! mDNS TXT records carry no certificate fingerprint, so an attacker on
//! the LAN can advertise a spoofed worker. Discovery therefore **never
//! auto-connects**: every discovered endpoint is registered as
//! *untrusted* ([`WorkerEndpoint::trusted`] = `false`) and the
//! orchestrator stops at pool registration. Connecting to an untrusted
//! endpoint is the caller's job (T8/T13) and must go through the T19
//! trust flow (`connect_with_trust` + [`TrustedKeyStore`],
//! [`ConnectTrust::require_pin`]) or an explicit user trust action.
//! [`ConfigDiscovery`] endpoints with a pre-shared `fingerprint` are
//! registered as trusted.
//!
//! Landed ahead of the T13 integration; no call site constructs an
//! orchestrator yet.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use reimagine_backend_worker_protocol::TransportKind;
use reimagine_backend_worker_transport_quic::discovery::{DiscoveredWorker, MdnsWorkerDiscovery};
use reimagine_config::{InferenceBackendConfig, WorkerEndpointConfig};
use tokio::sync::Mutex;

use super::pool::WorkerEndpoint;
use super::topology::{ConnectionTopologyManager, TopologyError};

/// Default poll interval for reconciling discovered endpoints.
pub const DEFAULT_DISCOVERY_INTERVAL: Duration = Duration::from_secs(10);

/// A pluggable source of worker endpoints.
///
/// `start`/`stop` are idempotent; a source that has not been started
/// reports no endpoints. Implementations must be cheap to call and must
/// not connect to any endpoint (registration only — see the module-level
/// trust gate).
pub trait WorkerDiscovery: Send + Sync {
    /// Start the discovery mechanism. Idempotent.
    fn start(&self) -> Result<(), String>;
    /// Stop the discovery mechanism. Idempotent.
    fn stop(&self) -> Result<(), String>;
    /// The endpoints currently known to this source.
    fn discovered(&self) -> Vec<WorkerEndpoint>;
}

/// Outcome of a single [`DiscoveryOrchestrator::reconcile_once`] pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconcileReport {
    /// Endpoints reported across all sources (after deduplication).
    pub discovered: usize,
    /// Endpoints newly registered in the topology pool.
    pub registered: usize,
    /// Endpoints removed from the topology pool after disappearing.
    pub deregistered: usize,
    /// Non-fatal errors encountered while reconciling.
    pub errors: Vec<String>,
}

/// mDNS-based discovery source.
///
/// Wraps [`MdnsWorkerDiscovery`] and converts each [`DiscoveredWorker`]
/// into an untrusted [`WorkerEndpoint`]. The worker's advertised
/// `fingerprint` TXT (T19) is preserved in
/// `metadata["fingerprint"]` as advisory information for the caller's
/// trust flow — it is not itself a pin.
pub struct MdnsDiscovery {
    inner: std::sync::Mutex<Option<MdnsWorkerDiscovery>>,
}

impl MdnsDiscovery {
    /// Create an (unstarted) mDNS discovery source.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(None),
        }
    }
}

impl Default for MdnsDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerDiscovery for MdnsDiscovery {
    fn start(&self) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "mdns lock poisoned".to_owned())?;
        if inner.is_none() {
            *inner = Some(MdnsWorkerDiscovery::start().map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn stop(&self) -> Result<(), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "mdns lock poisoned".to_owned())?;
        if let Some(discovery) = inner.take() {
            discovery.stop().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn discovered(&self) -> Vec<WorkerEndpoint> {
        let guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };
        match guard.as_ref() {
            Some(discovery) => discovery
                .discovered()
                .into_iter()
                .map(mdns_endpoint)
                .collect(),
            None => Vec::new(),
        }
    }
}

/// Map a discovered mDNS worker to a [`WorkerEndpoint`].
///
/// # Trust gating (T19)
///
/// mDNS is unauthenticated, so the endpoint is always registered as
/// **untrusted** (`trusted: false`): the orchestrator registers it in
/// the pool but never connects. The advertised `fingerprint` TXT is kept
/// in `metadata` so the caller can pre-validate it against pinned keys
/// before running the T19 trust flow
/// (`connect_with_trust`, [`ConnectTrust::require_pin`]).
fn mdns_endpoint(worker: DiscoveredWorker) -> WorkerEndpoint {
    let address = worker
        .quic_endpoint()
        .map(|addr| format!("quic://{addr}"))
        .unwrap_or_else(|| format!("quic://{}", worker.addr));
    let capabilities: Vec<String> = worker
        .capabilities()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let device_label = worker
        .devices()
        .first()
        .map(|d| (*d).to_owned())
        .unwrap_or_else(|| "remote".to_owned());
    let fingerprint = worker.fingerprint().map(str::to_owned);
    WorkerEndpoint {
        id: worker.id,
        transport_kind: TransportKind::Quic,
        address,
        capabilities,
        device_label,
        trusted: false,
        metadata: serde_json::json!({
            "source": "mdns",
            "fingerprint": fingerprint,
        }),
    }
}

/// Static/manual discovery source backed by [`InferenceBackendConfig`].
///
/// Endpoints come from `config.workers` (manual configuration; also the
/// T12 home for T19 pre-shared fingerprints). An endpoint configured
/// with a `fingerprint` is registered as **trusted** (verifiable
/// identity); one without a fingerprint is untrusted pending the T19
/// trust flow.
pub struct ConfigDiscovery {
    endpoints: Vec<WorkerEndpoint>,
}

impl ConfigDiscovery {
    /// Build from `InferenceBackendConfig::workers`.
    ///
    /// Returns an error if any configured endpoint is malformed (e.g. an
    /// unsupported transport kind).
    pub fn from_config(config: &InferenceBackendConfig) -> Result<Self, String> {
        let endpoints = config
            .workers
            .iter()
            .map(config_endpoint)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { endpoints })
    }

    /// Build from a pre-resolved endpoint list (test/programmatic seam).
    #[must_use]
    pub fn from_endpoints(endpoints: Vec<WorkerEndpoint>) -> Self {
        Self { endpoints }
    }

    /// The configured endpoints.
    #[must_use]
    pub fn endpoints(&self) -> &[WorkerEndpoint] {
        &self.endpoints
    }
}

impl WorkerDiscovery for ConfigDiscovery {
    fn start(&self) -> Result<(), String> {
        Ok(())
    }

    fn stop(&self) -> Result<(), String> {
        Ok(())
    }

    fn discovered(&self) -> Vec<WorkerEndpoint> {
        self.endpoints.clone()
    }
}

/// Map a configured endpoint to a [`WorkerEndpoint`].
fn config_endpoint(cfg: &WorkerEndpointConfig) -> Result<WorkerEndpoint, String> {
    let transport_kind = match cfg.transport.as_str() {
        "quic" => TransportKind::Quic,
        "stdio" => TransportKind::Stdio,
        "grpc" => TransportKind::Grpc,
        other => return Err(format!("unsupported transport kind `{other}`")),
    };
    // A pre-shared fingerprint (T19) is the only verifiable identity a
    // manual config can hold; without one the endpoint stays untrusted
    // pending the trust flow.
    let trusted = cfg.fingerprint.is_some();
    Ok(WorkerEndpoint {
        id: cfg.id.clone(),
        transport_kind,
        address: cfg.address.clone(),
        capabilities: cfg.capabilities.clone(),
        device_label: cfg.device_label.clone(),
        trusted,
        metadata: serde_json::json!({
            "source": "config",
            "fingerprint": cfg.fingerprint,
        }),
    })
}

/// Aggregates [`WorkerDiscovery`] sources and drives pool updates in a
/// [`ConnectionTopologyManager`].
///
/// On each reconcile pass the orchestrator:
///
/// 1. Collects and deduplicates the endpoints reported by every source
///    (by id; a trusted endpoint wins over an untrusted one for the
///    same id);
/// 2. Registers endpoints that are not yet in the pool;
/// 3. Deregisters endpoints that the orchestrator previously registered
///    and which have vanished — never endpoints registered by other
///    subsystems (tracked via a `known` id set).
pub struct DiscoveryOrchestrator {
    sources: Vec<Arc<dyn WorkerDiscovery>>,
    topology: Arc<Mutex<ConnectionTopologyManager>>,
    known: std::sync::Mutex<HashSet<String>>,
    interval: Duration,
}

impl std::fmt::Debug for DiscoveryOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscoveryOrchestrator")
            .field("sources", &self.sources.len())
            .field("interval", &self.interval)
            .finish_non_exhaustive()
    }
}

impl DiscoveryOrchestrator {
    /// Create an orchestrator over `sources`, reconciling into
    /// `topology` every `interval`.
    ///
    /// `interval` is injectable so tests can use short poll periods
    /// without sleeping; [`DEFAULT_DISCOVERY_INTERVAL`] is the
    /// production default.
    pub fn new(
        sources: Vec<Arc<dyn WorkerDiscovery>>,
        topology: Arc<Mutex<ConnectionTopologyManager>>,
        interval: Duration,
    ) -> Self {
        Self {
            sources,
            topology,
            known: std::sync::Mutex::new(HashSet::new()),
            interval,
        }
    }

    /// The poll interval.
    #[must_use]
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Start every source. Returns one `Err` per failed source; a
    /// failure does not stop the other sources from starting.
    pub fn start(&self) -> Vec<Result<(), String>> {
        self.sources.iter().map(|s| s.start()).collect()
    }

    /// Stop every source. Returns one `Err` per failed source.
    pub fn stop(&self) -> Vec<Result<(), String>> {
        self.sources.iter().map(|s| s.stop()).collect()
    }

    /// Run one aggregate-and-reconcile pass over all sources.
    ///
    /// Safe to call concurrently with other passes and with
    /// [`Self::run`]; the topology manager lock serializes pool
    /// mutations.
    pub async fn reconcile_once(&self) -> ReconcileReport {
        let (discovered, discovered_ids) = self.aggregate();
        let mut report = ReconcileReport {
            discovered: discovered.len(),
            ..ReconcileReport::default()
        };

        let mut topology = self.topology.lock().await;

        // Deregister workers this orchestrator registered that vanished.
        let gone = {
            let mut known = self.known.lock().expect("discovery known set poisoned");
            let gone: Vec<String> = known
                .iter()
                .filter(|id| !discovered_ids.contains(*id))
                .cloned()
                .collect();
            for id in &gone {
                known.remove(id);
            }
            gone
        };
        for id in gone {
            match topology.deregister_endpoint(&id) {
                Ok(()) => report.deregistered += 1,
                // Already removed by another subsystem: nothing to do.
                Err(TopologyError::EndpointNotFound(_)) => {}
                Err(e) => report.errors.push(e.to_string()),
            }
        }

        // Register newly discovered endpoints (already-present ids are
        // left untouched so other subsystems keep their registration).
        for endpoint in discovered {
            let id = endpoint.id.clone();
            if topology.pool().get(&id).is_some() {
                continue;
            }
            match topology.register_endpoint(endpoint) {
                Ok(()) => {
                    report.registered += 1;
                    if let Ok(mut known) = self.known.lock() {
                        known.insert(id);
                    }
                }
                Err(TopologyError::DuplicateEndpoint(_)) => {}
                Err(e) => report.errors.push(e.to_string()),
            }
        }
        report
    }

    /// Poll `discovered()` on every source at `interval`, reconciling
    /// the pool after each pass, until the returned handle is aborted.
    pub fn run(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let report = self.reconcile_once().await;
                for error in report.errors {
                    tracing::warn!(error = %error, "discovery reconcile error");
                }
                tokio::time::sleep(self.interval).await;
            }
        })
    }

    /// Aggregate every source's endpoints, deduplicated by id.
    ///
    /// When two sources report the same id, the trusted endpoint wins
    /// (a user-asserted config endpoint overrides an unauthenticated
    /// mDNS advertisement); otherwise the first source in registration
    /// order wins.
    fn aggregate(&self) -> (Vec<WorkerEndpoint>, HashSet<String>) {
        let mut by_id: HashMap<String, WorkerEndpoint> = HashMap::new();
        for source in &self.sources {
            for endpoint in source.discovered() {
                match by_id.get(&endpoint.id) {
                    None => {
                        by_id.insert(endpoint.id.clone(), endpoint);
                    }
                    Some(existing) if !existing.trusted && endpoint.trusted => {
                        by_id.insert(endpoint.id.clone(), endpoint);
                    }
                    Some(_) => {}
                }
            }
        }
        let ids: HashSet<String> = by_id.keys().cloned().collect();
        (by_id.into_values().collect(), ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_inference::{Backend, StaticBackendSelectionPolicy};
    use std::collections::HashMap;

    fn test_endpoint(id: &str, trusted: bool) -> WorkerEndpoint {
        WorkerEndpoint {
            id: id.to_owned(),
            transport_kind: TransportKind::Quic,
            address: "quic://192.168.1.10:9100".to_owned(),
            capabilities: vec!["load_bundle".to_owned()],
            device_label: "cuda:0".to_owned(),
            trusted,
            metadata: serde_json::json!({}),
        }
    }

    struct NoopBackendFactory;

    impl super::super::topology::WorkerBackendFactory for NoopBackendFactory {
        fn build_backend(
            &self,
            _endpoint: &WorkerEndpoint,
        ) -> Option<Arc<dyn reimagine_inference::InferenceBackend>> {
            None
        }

        fn backend_label(&self) -> Backend {
            Backend::new("none")
        }
    }

    fn topology_manager() -> Arc<Mutex<ConnectionTopologyManager>> {
        Arc::new(Mutex::new(ConnectionTopologyManager::new(
            super::super::pool::WorkerPool::new(),
            Arc::new(StaticBackendSelectionPolicy::new(Vec::new())),
            Arc::new(NoopBackendFactory),
        )))
    }

    /// A scriptable discovery source for tests.
    struct MockDiscovery {
        endpoints: std::sync::Mutex<Vec<WorkerEndpoint>>,
        started: std::sync::Mutex<bool>,
        start_calls: std::sync::Mutex<u32>,
        stop_calls: std::sync::Mutex<u32>,
    }

    impl MockDiscovery {
        fn new(endpoints: Vec<WorkerEndpoint>) -> Arc<Self> {
            Arc::new(Self {
                endpoints: std::sync::Mutex::new(endpoints),
                started: std::sync::Mutex::new(false),
                start_calls: std::sync::Mutex::new(0),
                stop_calls: std::sync::Mutex::new(0),
            })
        }

        fn set(&self, endpoints: Vec<WorkerEndpoint>) {
            *self.endpoints.lock().unwrap() = endpoints;
        }

        fn start_calls(&self) -> u32 {
            *self.start_calls.lock().unwrap()
        }

        fn stop_calls(&self) -> u32 {
            *self.stop_calls.lock().unwrap()
        }
    }

    impl WorkerDiscovery for MockDiscovery {
        fn start(&self) -> Result<(), String> {
            *self.started.lock().unwrap() = true;
            *self.start_calls.lock().unwrap() += 1;
            Ok(())
        }

        fn stop(&self) -> Result<(), String> {
            *self.started.lock().unwrap() = false;
            *self.stop_calls.lock().unwrap() += 1;
            Ok(())
        }

        fn discovered(&self) -> Vec<WorkerEndpoint> {
            if !*self.started.lock().unwrap() {
                return Vec::new();
            }
            self.endpoints.lock().unwrap().clone()
        }
    }

    fn discovered_worker(id: &str, fingerprint: Option<&str>) -> DiscoveredWorker {
        let mut props = HashMap::new();
        props.insert(
            "endpoint".to_owned(),
            "quic://192.168.1.100:9100".to_owned(),
        );
        props.insert("backend".to_owned(), "burn".to_owned());
        props.insert("devices".to_owned(), "cuda:0,cuda:1".to_owned());
        props.insert(
            "capabilities".to_owned(),
            "load_bundle,text_encode".to_owned(),
        );
        if let Some(fp) = fingerprint {
            props.insert("fingerprint".to_owned(), fp.to_owned());
        }
        DiscoveredWorker {
            id: format!("{id}._reimagine-worker._tcp.local."),
            addr: "192.168.1.100:9100".parse().unwrap(),
            properties: props,
        }
    }

    #[test]
    fn mdns_mapping_produces_untrusted_quic_endpoint() {
        let endpoint = mdns_endpoint(discovered_worker("worker-a", Some("aabbcc")));
        assert_eq!(endpoint.id, "worker-a._reimagine-worker._tcp.local.");
        assert_eq!(endpoint.transport_kind, TransportKind::Quic);
        assert_eq!(endpoint.address, "quic://192.168.1.100:9100");
        assert_eq!(
            endpoint.capabilities,
            vec!["load_bundle".to_owned(), "text_encode".to_owned()]
        );
        assert_eq!(endpoint.device_label, "cuda:0");
        // Trust gate: mDNS is unauthenticated — never trusted, never
        // auto-connected.
        assert!(!endpoint.trusted);
        assert_eq!(endpoint.metadata["source"], "mdns");
        assert_eq!(endpoint.metadata["fingerprint"], "aabbcc");
    }

    #[test]
    fn mdns_mapping_without_fingerprint_or_devices() {
        let mut worker = discovered_worker("worker-b", None);
        worker.properties.remove("devices");
        let endpoint = mdns_endpoint(worker);
        assert_eq!(endpoint.metadata["fingerprint"], serde_json::Value::Null);
        assert_eq!(endpoint.device_label, "remote");
        assert!(!endpoint.trusted);
    }

    #[test]
    fn mdns_discovery_reports_nothing_before_start() {
        let source = MdnsDiscovery::new();
        assert!(source.discovered().is_empty());
        // start/stop are idempotent no-ops on an unstarted source.
        assert!(source.stop().is_ok());
    }

    #[test]
    fn config_discovery_reads_static_endpoints_from_config() {
        let config = InferenceBackendConfig {
            workers: vec![
                WorkerEndpointConfig {
                    id: "worker-a".to_owned(),
                    transport: "quic".to_owned(),
                    address: "quic://192.168.1.5:9100".to_owned(),
                    capabilities: vec!["load_bundle".to_owned()],
                    device_label: "cuda:0".to_owned(),
                    fingerprint: Some("aabbccdd".to_owned()),
                },
                WorkerEndpointConfig {
                    id: "worker-b".to_owned(),
                    transport: "quic".to_owned(),
                    address: "quic://192.168.1.6:9100".to_owned(),
                    capabilities: Vec::new(),
                    device_label: String::new(),
                    fingerprint: None,
                },
            ],
            ..InferenceBackendConfig::default()
        };
        let source = ConfigDiscovery::from_config(&config).unwrap();
        let endpoints = source.discovered();
        assert_eq!(endpoints.len(), 2);

        let a = &endpoints[0];
        assert_eq!(a.id, "worker-a");
        assert_eq!(a.transport_kind, TransportKind::Quic);
        assert_eq!(a.address, "quic://192.168.1.5:9100");
        // A pre-shared fingerprint marks the endpoint trusted.
        assert!(a.trusted);
        assert_eq!(a.metadata["fingerprint"], "aabbccdd");

        // Without a fingerprint the manual endpoint stays untrusted
        // pending the T19 trust flow.
        assert!(!endpoints[1].trusted);
        assert_eq!(
            endpoints[1].metadata["fingerprint"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn config_discovery_accepts_other_transports() {
        let config = InferenceBackendConfig {
            workers: vec![WorkerEndpointConfig {
                id: "w".to_owned(),
                transport: "grpc".to_owned(),
                address: "grpc://cloud.example:50051".to_owned(),
                capabilities: Vec::new(),
                device_label: String::new(),
                fingerprint: None,
            }],
            ..InferenceBackendConfig::default()
        };
        let source = ConfigDiscovery::from_config(&config).unwrap();
        assert_eq!(source.discovered()[0].transport_kind, TransportKind::Grpc);
    }

    #[test]
    fn config_discovery_rejects_unknown_transport() {
        let config = InferenceBackendConfig {
            workers: vec![WorkerEndpointConfig {
                id: "w".to_owned(),
                transport: "carrier-pigeon".to_owned(),
                address: "quic://x".to_owned(),
                capabilities: Vec::new(),
                device_label: String::new(),
                fingerprint: None,
            }],
            ..InferenceBackendConfig::default()
        };
        assert!(ConfigDiscovery::from_config(&config).is_err());
    }

    #[tokio::test]
    async fn orchestrator_registers_new_discovered_endpoints() {
        let source =
            MockDiscovery::new(vec![test_endpoint("w1", false), test_endpoint("w2", false)]);
        source.start().unwrap();
        let topology = topology_manager();
        let orchestrator =
            DiscoveryOrchestrator::new(vec![source], topology.clone(), Duration::from_millis(1));

        let report = orchestrator.reconcile_once().await;
        assert_eq!(report.discovered, 2);
        assert_eq!(report.registered, 2);
        assert_eq!(report.deregistered, 0);
        assert!(report.errors.is_empty());

        let pool = topology.lock().await;
        assert_eq!(pool.pool().len(), 2);
        assert!(!pool.all_endpoints()[0].trusted);
    }

    #[tokio::test]
    async fn orchestrator_deregisters_vanished_endpoints() {
        let source =
            MockDiscovery::new(vec![test_endpoint("w1", false), test_endpoint("w2", false)]);
        source.start().unwrap();
        let topology = topology_manager();
        let orchestrator = DiscoveryOrchestrator::new(
            vec![source.clone()],
            topology.clone(),
            Duration::from_millis(1),
        );

        orchestrator.reconcile_once().await;
        assert_eq!(topology.lock().await.pool().len(), 2);

        // w2 disappears (missed ServiceRemoved event).
        source.set(vec![test_endpoint("w1", false)]);
        let report = orchestrator.reconcile_once().await;
        assert_eq!(report.deregistered, 1);
        let pool = topology.lock().await;
        assert_eq!(pool.pool().len(), 1);
        assert!(pool.pool().get("w1").is_some());
        assert!(pool.pool().get("w2").is_none());
    }

    #[tokio::test]
    async fn orchestrator_does_not_deregister_endpoints_it_did_not_register() {
        let source = MockDiscovery::new(Vec::new());
        source.start().unwrap();
        let topology = topology_manager();
        // Another subsystem registers a worker in the same pool.
        topology
            .lock()
            .await
            .register_endpoint(test_endpoint("external", true))
            .unwrap();

        let orchestrator =
            DiscoveryOrchestrator::new(vec![source], topology.clone(), Duration::from_millis(1));
        let report = orchestrator.reconcile_once().await;
        assert_eq!(report.deregistered, 0);
        let pool = topology.lock().await;
        assert_eq!(pool.pool().len(), 1);
        assert!(pool.pool().get("external").is_some());
    }

    #[tokio::test]
    async fn orchestrator_reconcile_is_idempotent() {
        let source = MockDiscovery::new(vec![test_endpoint("w1", false)]);
        source.start().unwrap();
        let topology = topology_manager();
        let orchestrator =
            DiscoveryOrchestrator::new(vec![source], topology.clone(), Duration::from_millis(1));

        let first = orchestrator.reconcile_once().await;
        assert_eq!(first.registered, 1);
        // Second pass must not re-register or deregister anything.
        let second = orchestrator.reconcile_once().await;
        assert_eq!(second.registered, 0);
        assert_eq!(second.deregistered, 0);
        assert_eq!(topology.lock().await.pool().len(), 1);
    }

    #[tokio::test]
    async fn orchestrator_aggregates_sources_and_prefers_trusted() {
        // The same worker id is advertised by mDNS (untrusted) and
        // declared in config with a pin (trusted): the trusted endpoint
        // must win and the pool must hold exactly one entry.
        let mdns_source = MockDiscovery::new(vec![test_endpoint("w1", false)]);
        mdns_source.start().unwrap();
        let config_source = Arc::new(ConfigDiscovery::from_endpoints(vec![test_endpoint(
            "w1", true,
        )]));
        config_source.start().unwrap();

        let topology = topology_manager();
        let orchestrator = DiscoveryOrchestrator::new(
            vec![mdns_source, config_source],
            topology.clone(),
            Duration::from_millis(1),
        );
        let report = orchestrator.reconcile_once().await;
        assert_eq!(report.discovered, 1);
        assert_eq!(report.registered, 1);

        let pool = topology.lock().await;
        assert_eq!(pool.pool().len(), 1);
        assert!(pool.pool().get("w1").unwrap().endpoint.trusted);
    }

    #[tokio::test]
    async fn orchestrator_start_stops_all_sources() {
        let a = MockDiscovery::new(Vec::new());
        let b = MockDiscovery::new(Vec::new());
        let orchestrator = DiscoveryOrchestrator::new(
            vec![a.clone(), b.clone()],
            topology_manager(),
            Duration::from_millis(1),
        );

        assert_eq!(orchestrator.start(), vec![Ok(()), Ok(())]);
        assert_eq!(a.start_calls(), 1);
        assert_eq!(b.start_calls(), 1);

        assert_eq!(orchestrator.stop(), vec![Ok(()), Ok(())]);
        assert_eq!(a.stop_calls(), 1);
        assert_eq!(b.stop_calls(), 1);
    }

    #[tokio::test]
    async fn orchestrator_run_polls_at_interval() {
        let source = MockDiscovery::new(vec![test_endpoint("w1", false)]);
        source.start().unwrap();
        let topology = topology_manager();
        let orchestrator = Arc::new(DiscoveryOrchestrator::new(
            vec![source.clone()],
            topology.clone(),
            Duration::from_millis(5),
        ));

        let handle = orchestrator.clone().run();

        // Let the poll loop register w1, then have w2 appear.
        tokio::time::sleep(Duration::from_millis(30)).await;
        source.set(vec![test_endpoint("w1", false), test_endpoint("w2", false)]);
        tokio::time::sleep(Duration::from_millis(30)).await;
        handle.abort();

        let pool = topology.lock().await;
        assert_eq!(pool.pool().len(), 2);
        assert!(pool.pool().get("w1").is_some());
        assert!(pool.pool().get("w2").is_some());
    }

    #[test]
    fn default_interval_is_ten_seconds() {
        assert_eq!(DEFAULT_DISCOVERY_INTERVAL, Duration::from_secs(10));
    }
}
