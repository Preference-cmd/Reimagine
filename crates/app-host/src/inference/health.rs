//! Periodic health monitoring of pooled workers (T11).
//!
//! The [`HealthMonitor`] sweeps the [`WorkerPool`] on a per-transport
//! interval, probes each endpoint through a pluggable
//! [`WorkerHealthProbe`], records latency/failure counts, and flips
//! workers to [`WorkerState::Offline`] after a consecutive-failure
//! threshold. Reconnection is out of scope for V1 (workers are
//! supervised by the host process; a dead network worker needs a
//! fresh connect via its candidate).
//!
//! The production probe seam (`StartedWorker::health()` for stdio,
//! QUIC connection liveness, gRPC `HealthCheck`) is wired by the
//! topology integration (T13); this module ships the state machine,
//! thresholds, and tests.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reimagine_backend_worker_protocol::transport::TransportKind;
use tokio::sync::Mutex;

use super::pool::{WorkerEndpoint, WorkerPool, WorkerState};

/// Outcome of a single health probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthProbeResult {
    pub healthy: bool,
    pub latency_ms: Option<u64>,
    pub message: Option<String>,
}

/// Pluggable transport-level probe.
#[async_trait::async_trait]
pub trait WorkerHealthProbe: Send + Sync {
    /// Probe `endpoint`; `Err` counts as an immediate failure.
    async fn probe(&self, endpoint: &WorkerEndpoint) -> Result<HealthProbeResult, String>;
}

/// Per-transport check policy (ticket: tight for stdio, generous for
/// network transports).
#[derive(Debug, Clone, Copy)]
pub struct TransportHealthPolicy {
    pub check_interval: Duration,
    pub probe_timeout: Duration,
    pub failure_threshold: u32,
}

impl Default for TransportHealthPolicy {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(5),
            probe_timeout: Duration::from_secs(2),
            failure_threshold: 2,
        }
    }
}

impl TransportHealthPolicy {
    pub fn for_kind(kind: TransportKind) -> Self {
        match kind {
            TransportKind::Stdio => Self {
                check_interval: Duration::from_secs(5),
                probe_timeout: Duration::from_secs(2),
                failure_threshold: 2,
            },
            TransportKind::Quic => Self {
                check_interval: Duration::from_secs(15),
                probe_timeout: Duration::from_secs(5),
                failure_threshold: 3,
            },
            TransportKind::Grpc => Self {
                check_interval: Duration::from_secs(30),
                probe_timeout: Duration::from_secs(10),
                failure_threshold: 3,
            },
            TransportKind::Mock => Self::default(),
        }
    }
}

/// Failure bookkeeping per worker.
#[derive(Debug, Clone)]
struct MonitorEntry {
    consecutive_failures: u32,
    last_state: WorkerState,
}

impl Default for MonitorEntry {
    fn default() -> Self {
        Self {
            consecutive_failures: 0,
            last_state: WorkerState::Connecting,
        }
    }
}

/// Periodic health monitor over a shared worker pool.
pub struct HealthMonitor {
    pool: Arc<Mutex<WorkerPool>>,
    probe: Arc<dyn WorkerHealthProbe>,
    entries: Mutex<HashMap<String, MonitorEntry>>,
    /// Called after a state flip (e.g. router rebuild hook).
    on_state_change: Option<Box<dyn Fn(&str, WorkerState) + Send + Sync>>,
}

impl std::fmt::Debug for HealthMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HealthMonitor")
            .field("pool", &"<locked>")
            .field("probe", &"<trait>")
            .finish_non_exhaustive()
    }
}

impl HealthMonitor {
    pub fn new(pool: Arc<Mutex<WorkerPool>>, probe: Arc<dyn WorkerHealthProbe>) -> Self {
        Self {
            pool,
            probe,
            entries: Mutex::new(HashMap::new()),
            on_state_change: None,
        }
    }

    /// Register a hook fired after any pool state flip.
    pub fn with_state_change_hook(
        mut self,
        hook: Box<dyn Fn(&str, WorkerState) + Send + Sync>,
    ) -> Self {
        self.on_state_change = Some(hook);
        self
    }

    /// Run a single sweep over all registered workers, returning the
    /// ids whose pool state changed during this sweep.
    pub async fn check_once(&self) -> Vec<String> {
        let mut changed = Vec::new();
        let endpoints: Vec<WorkerEndpoint> = {
            let pool = self.pool.lock().await;
            pool.all_endpoints().into_iter().cloned().collect()
        };
        let mut entries = self.entries.lock().await;
        for endpoint in endpoints {
            let policy = TransportHealthPolicy::for_kind(endpoint.transport_kind);
            let entry = entries.entry(endpoint.id.clone()).or_default();
            let outcome =
                tokio::time::timeout(policy.probe_timeout, self.probe.probe(&endpoint)).await;

            let healthy = match outcome {
                Ok(Ok(result)) => {
                    let mut pool = self.pool.lock().await;
                    if let Some(worker) = pool.get_mut(&endpoint.id) {
                        worker.health.last_check = Some(Instant::now());
                        worker.health.latency_ms = result.latency_ms;
                        if result.healthy {
                            worker.health.failure_count = 0;
                            entry.consecutive_failures = 0;
                            if worker.state != WorkerState::Ready {
                                worker.state = WorkerState::Ready;
                                entry.last_state = WorkerState::Ready;
                                changed.push(endpoint.id.clone());
                            }
                            true
                        } else {
                            false
                        }
                    } else {
                        true
                    }
                }
                _ => false,
            };

            if !healthy {
                entry.consecutive_failures += 1;
                let mut pool = self.pool.lock().await;
                if let Some(worker) = pool.get_mut(&endpoint.id) {
                    worker.health.failure_count = entry.consecutive_failures;
                    if entry.consecutive_failures >= policy.failure_threshold
                        && worker.state != WorkerState::Offline
                    {
                        worker.state = WorkerState::Offline;
                        entry.last_state = WorkerState::Offline;
                        changed.push(endpoint.id.clone());
                    }
                }
            }
        }
        drop(entries);

        for id in &changed {
            let state = {
                let pool = self.pool.lock().await;
                pool.get(id).map(|w| w.state)
            };
            if let (Some(state), Some(hook)) = (state, &self.on_state_change) {
                hook(id, state);
            }
        }
        changed
    }

    /// Run periodic sweeps until the returned handle is aborted.
    pub fn run(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let _ = self.check_once().await;
                // Per-transport scheduling is approximated with the
                // tightest policy interval; per-kind intervals are
                // enforced by callers that need them (T13).
                tokio::time::sleep(TransportHealthPolicy::default().check_interval).await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::pool::WorkerEndpoint;

    fn test_endpoint(id: &str) -> WorkerEndpoint {
        WorkerEndpoint {
            id: id.to_owned(),
            transport_kind: TransportKind::Stdio,
            address: "local".to_owned(),
            capabilities: vec!["echo".to_owned()],
            device_label: "cpu".to_owned(),
            metadata: serde_json::json!({}),
        }
    }

    struct FakeProbe {
        results: std::sync::Mutex<Vec<Result<HealthProbeResult, String>>>,
    }

    impl FakeProbe {
        fn new(results: Vec<Result<HealthProbeResult, String>>) -> Self {
            Self {
                results: std::sync::Mutex::new(results),
            }
        }
    }

    #[async_trait::async_trait]
    impl WorkerHealthProbe for FakeProbe {
        async fn probe(&self, _endpoint: &WorkerEndpoint) -> Result<HealthProbeResult, String> {
            let results = self.results.lock().expect("probe results poisoned");
            Ok(results
                .first()
                .cloned()
                .unwrap_or_else(|| {
                    Ok(HealthProbeResult {
                        healthy: true,
                        latency_ms: Some(1),
                        message: None,
                    })
                })
                .unwrap_or_else(|e| HealthProbeResult {
                    healthy: false,
                    latency_ms: None,
                    message: Some(e),
                }))
        }
    }

    fn healthy() -> Result<HealthProbeResult, String> {
        Ok(HealthProbeResult {
            healthy: true,
            latency_ms: Some(7),
            message: None,
        })
    }

    fn failing(reason: &str) -> Result<HealthProbeResult, String> {
        Ok(HealthProbeResult {
            healthy: false,
            latency_ms: None,
            message: Some(reason.to_owned()),
        })
    }

    #[tokio::test]
    async fn healthy_worker_becomes_ready_and_records_latency() {
        let pool = Arc::new(Mutex::new(WorkerPool::new()));
        pool.lock().await.register(test_endpoint("w1"));
        let monitor = HealthMonitor::new(pool.clone(), Arc::new(FakeProbe::new(vec![healthy()])));
        let changed = monitor.check_once().await;
        assert_eq!(changed, vec!["w1".to_owned()]);
        let pool = pool.lock().await;
        let worker = pool.get("w1").unwrap();
        assert_eq!(worker.state, WorkerState::Ready);
        assert_eq!(worker.health.latency_ms, Some(7));
        assert_eq!(worker.health.failure_count, 0);
    }

    #[tokio::test]
    async fn failures_below_threshold_do_not_flip_offline() {
        let pool = Arc::new(Mutex::new(WorkerPool::new()));
        pool.lock().await.register(test_endpoint("w1"));
        let monitor = HealthMonitor::new(
            pool.clone(),
            Arc::new(FakeProbe::new(vec![failing("down")])),
        );
        // One failing sweep: stdio threshold is 2.
        let changed = monitor.check_once().await;
        assert!(changed.is_empty());
        let pool = pool.lock().await;
        assert_eq!(pool.get("w1").unwrap().state, WorkerState::Connecting);
        assert_eq!(pool.get("w1").unwrap().health.failure_count, 1);
    }

    #[tokio::test]
    async fn threshold_reached_flips_offline_and_fires_hook() {
        let pool = Arc::new(Mutex::new(WorkerPool::new()));
        pool.lock().await.register(test_endpoint("w1"));
        let flipped: Arc<std::sync::Mutex<Vec<(String, WorkerState)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let hook_pool = flipped.clone();
        let monitor = HealthMonitor::new(
            pool.clone(),
            Arc::new(FakeProbe::new(vec![failing("down")])),
        )
        .with_state_change_hook(Box::new(move |id, state| {
            hook_pool.lock().unwrap().push((id.to_owned(), state));
        }));
        let _ = monitor.check_once().await;
        let _ = monitor.check_once().await;
        let pool = pool.lock().await;
        assert_eq!(pool.get("w1").unwrap().state, WorkerState::Offline);
        drop(pool);
        assert_eq!(
            *flipped.lock().unwrap(),
            vec![("w1".to_owned(), WorkerState::Offline)]
        );
    }

    #[tokio::test]
    async fn recovery_returns_worker_to_ready() {
        let pool = Arc::new(Mutex::new(WorkerPool::new()));
        pool.lock().await.register(test_endpoint("w1"));
        let monitor = HealthMonitor::new(
            pool.clone(),
            Arc::new(FakeProbe::new(vec![failing("down")])),
        );
        let _ = monitor.check_once().await;
        let _ = monitor.check_once().await;
        {
            let pool = pool.lock().await;
            assert_eq!(pool.get("w1").unwrap().state, WorkerState::Offline);
        }
        // Probe flips healthy (FakeProbe returns the first result forever,
        // so swap in a healthy-only probe by rebuilding the monitor).
        let monitor = HealthMonitor::new(pool.clone(), Arc::new(FakeProbe::new(vec![healthy()])));
        let changed = monitor.check_once().await;
        assert_eq!(changed, vec!["w1".to_owned()]);
        let pool = pool.lock().await;
        assert_eq!(pool.get("w1").unwrap().state, WorkerState::Ready);
    }

    #[tokio::test]
    async fn per_transport_policy_differs() {
        let stdio = TransportHealthPolicy::for_kind(TransportKind::Stdio);
        let quic = TransportHealthPolicy::for_kind(TransportKind::Quic);
        let grpc = TransportHealthPolicy::for_kind(TransportKind::Grpc);
        assert!(stdio.check_interval < quic.check_interval);
        assert!(quic.check_interval < grpc.check_interval);
        assert!(stdio.failure_threshold <= quic.failure_threshold);
    }
}
