//! Transfer planning and cost modeling for cross-worker tensor movement (T15).
//!
//! The planner decides *what* moves (token references + tensor metadata),
//! not *how* — serialization and execution belong to the transfer executor
//! (T17). Costs are config-driven defaults; live measurement (bandwidth,
//! cloud pricing) is a follow-up.

use reimagine_backend_worker_protocol::TensorMetadata;
use reimagine_backend_worker_protocol::transport::TransportKind;

pub mod executor;

pub use executor::{TransferChannel, TransferExecutor};

/// A worker the planner can route between.
///
/// Deliberately lightweight (no app-host types) so this crate stays
/// dependency-clean; hosts adapt their pool/endpoints into this view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerSpec {
    pub id: String,
    pub transport_kind: TransportKind,
}

/// The decided way to move a tensor from source to target.
#[derive(Debug, Clone, PartialEq)]
pub enum TransferPlan {
    /// Same worker — no movement (token stays put).
    Local { key: String },
    /// Same-machine process boundary (stdio transports).
    Ipc { key: String, target: String },
    /// Cross-machine network transfer (QUIC/gRPC transports).
    Network {
        key: String,
        target: String,
        estimated_bytes: u64,
    },
    /// External object storage hop.
    ObjectStorage { key: String, url: String },
    /// No viable route.
    Unsupported { reason: String },
}

/// Estimated cost of one hop.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TransferCost {
    /// Estimated transfer time in milliseconds.
    pub estimated_ms: f64,
    /// Estimated monetary cost in USD (0 for local/LAN defaults).
    pub estimated_usd: f64,
}

/// A route between two workers: ordered hops plus total cost.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferRoute {
    pub hops: Vec<TransferPlan>,
    pub total_cost: TransferCost,
}

/// Cost model for path selection.
pub trait CostModel: Send + Sync {
    /// Estimate the cost of moving `bytes` between `source` and `target`.
    fn estimate(&self, source: &WorkerSpec, target: &WorkerSpec, bytes: u64) -> TransferCost;

    /// Find the best route; defaults to a single direct hop.
    fn find_path(&self, source: &WorkerSpec, target: &WorkerSpec, bytes: u64) -> TransferRoute {
        let cost = self.estimate(source, target, bytes);
        TransferRoute {
            hops: vec![self.plan(source, target, bytes, cost)],
            total_cost: cost,
        }
    }

    /// Build the concrete plan for a direct hop (extensible hook).
    fn plan(
        &self,
        source: &WorkerSpec,
        target: &WorkerSpec,
        bytes: u64,
        cost: TransferCost,
    ) -> TransferPlan;
}

/// Configurable static cost model.
///
/// Defaults: local (same worker) = free; same-host IPC (stdio) = small
/// constant; LAN (QUIC) = bandwidth-based estimate; Cloud (gRPC) =
/// configurable per-byte price (default conservative 0.05 USD/GB).
#[derive(Debug, Clone)]
pub struct ConfigurableCostModel {
    /// Assumed LAN bandwidth in bytes/second (default 125 MB/s ≈ 1 Gbps).
    pub lan_bytes_per_sec: f64,
    /// Assumed per-GB cloud egress price in USD (default 0.05).
    pub cloud_usd_per_gb: f64,
    /// Fixed IPC hop cost in milliseconds (default 0.1; small-payload
    /// floor, never the dominant term).
    pub ipc_ms: f64,
}

impl Default for ConfigurableCostModel {
    fn default() -> Self {
        Self {
            lan_bytes_per_sec: 125.0 * 1024.0 * 1024.0,
            cloud_usd_per_gb: 0.05,
            ipc_ms: 0.1,
        }
    }
}

fn hop_class(source: &WorkerSpec, target: &WorkerSpec) -> HopClass {
    if source.id == target.id {
        return HopClass::Local;
    }
    match (&source.transport_kind, &target.transport_kind) {
        (TransportKind::Grpc, _) | (_, TransportKind::Grpc) => HopClass::Cloud,
        (TransportKind::Quic, _) | (_, TransportKind::Quic) => HopClass::Lan,
        (TransportKind::Stdio, TransportKind::Stdio) => HopClass::Ipc,
        // Mock transports are test-only; treat like same-host IPC.
        _ => HopClass::Ipc,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HopClass {
    Local,
    Ipc,
    Lan,
    Cloud,
}

impl ConfigurableCostModel {
    fn cost_for(&self, class: HopClass, bytes: u64) -> TransferCost {
        match class {
            HopClass::Local => TransferCost {
                estimated_ms: 0.0,
                estimated_usd: 0.0,
            },
            HopClass::Ipc => TransferCost {
                // stdio IPC is assumed 10x faster than LAN (shared-memory
                // semantics); stays strictly cheaper at every payload size.
                estimated_ms: (bytes as f64 / (self.lan_bytes_per_sec * 10.0)) * 1000.0,
                estimated_usd: 0.0,
            },
            HopClass::Lan => TransferCost {
                estimated_ms: (bytes as f64 / self.lan_bytes_per_sec) * 1000.0,
                estimated_usd: 0.0,
            },
            HopClass::Cloud => TransferCost {
                estimated_ms: (bytes as f64 / self.lan_bytes_per_sec) * 1000.0 * 3.0,
                estimated_usd: (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) * self.cloud_usd_per_gb,
            },
        }
    }
}

impl CostModel for ConfigurableCostModel {
    fn estimate(&self, source: &WorkerSpec, target: &WorkerSpec, bytes: u64) -> TransferCost {
        self.cost_for(hop_class(source, target), bytes)
    }

    fn plan(
        &self,
        source: &WorkerSpec,
        target: &WorkerSpec,
        bytes: u64,
        _cost: TransferCost,
    ) -> TransferPlan {
        let key = format!("{}:{}", source.id, target.id);
        match hop_class(source, target) {
            HopClass::Local => TransferPlan::Local { key },
            HopClass::Ipc => TransferPlan::Ipc {
                key,
                target: target.id.clone(),
            },
            HopClass::Lan | HopClass::Cloud => TransferPlan::Network {
                key,
                target: target.id.clone(),
                estimated_bytes: bytes,
            },
        }
    }
}

/// Plans tensor movement between workers.
#[derive(Debug, Clone)]
pub struct TransferPlanner<M: CostModel> {
    workers: Vec<WorkerSpec>,
    cost_model: M,
}

impl<M: CostModel> TransferPlanner<M> {
    /// Create a planner over a known worker set.
    pub fn new(workers: Vec<WorkerSpec>, cost_model: M) -> Self {
        Self {
            workers,
            cost_model,
        }
    }

    /// All known workers.
    pub fn workers(&self) -> &[WorkerSpec] {
        &self.workers
    }

    /// Resolve a worker id into its spec.
    pub fn worker(&self, id: &str) -> Option<&WorkerSpec> {
        self.workers.iter().find(|w| w.id == id)
    }

    /// Plan the cheapest transfer of `metadata` from `source_id` to
    /// `target_id`.
    pub fn plan_transfer(
        &self,
        source_id: &str,
        target_id: &str,
        metadata: &TensorMetadata,
    ) -> TransferPlan {
        let Some(source) = self.worker(source_id) else {
            return TransferPlan::Unsupported {
                reason: format!("unknown source worker `{source_id}`"),
            };
        };
        let Some(target) = self.worker(target_id) else {
            return TransferPlan::Unsupported {
                reason: format!("unknown target worker `{target_id}`"),
            };
        };
        if source.id == target.id {
            return TransferPlan::Local {
                key: format!("{}:{}", source.id, target.id),
            };
        }
        let cost = self
            .cost_model
            .estimate(source, target, metadata.size_bytes);
        self.cost_model
            .plan(source, target, metadata.size_bytes, cost)
    }

    /// Full route (single-hop default) including the estimated cost.
    pub fn route(
        &self,
        source_id: &str,
        target_id: &str,
        metadata: &TensorMetadata,
    ) -> TransferRoute {
        let Some(source) = self.worker(source_id) else {
            return TransferRoute {
                hops: vec![TransferPlan::Unsupported {
                    reason: format!("unknown source worker `{source_id}`"),
                }],
                total_cost: TransferCost {
                    estimated_ms: f64::INFINITY,
                    estimated_usd: f64::INFINITY,
                },
            };
        };
        let Some(target) = self.worker(target_id) else {
            return TransferRoute {
                hops: vec![TransferPlan::Unsupported {
                    reason: format!("unknown target worker `{target_id}`"),
                }],
                total_cost: TransferCost {
                    estimated_ms: f64::INFINITY,
                    estimated_usd: f64::INFINITY,
                },
            };
        };
        self.cost_model
            .find_path(source, target, metadata.size_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str, kind: TransportKind) -> WorkerSpec {
        WorkerSpec {
            id: id.to_owned(),
            transport_kind: kind,
        }
    }

    fn meta(bytes: u64) -> TensorMetadata {
        TensorMetadata {
            dtype: "f32".to_owned(),
            shape: vec![1, 4, 64, 64],
            size_bytes: bytes,
            backend_format: "burn::nchw".to_owned(),
        }
    }

    fn planner() -> TransferPlanner<ConfigurableCostModel> {
        TransferPlanner::new(
            vec![
                spec("local-1", TransportKind::Stdio),
                spec("lan-1", TransportKind::Quic),
                spec("cloud-1", TransportKind::Grpc),
            ],
            ConfigurableCostModel::default(),
        )
    }

    #[test]
    fn same_worker_is_local_and_free() {
        let p = planner();
        let plan = p.plan_transfer("local-1", "local-1", &meta(1024));
        assert_eq!(
            plan,
            TransferPlan::Local {
                key: "local-1:local-1".to_owned()
            }
        );
        let route = p.route("local-1", "local-1", &meta(1024));
        assert_eq!(route.total_cost.estimated_usd, 0.0);
        assert_eq!(route.total_cost.estimated_ms, 0.0);
    }

    #[test]
    fn stdio_to_stdio_is_ipc() {
        let p = planner();
        let plan = p.plan_transfer("local-1", "local-1", &meta(1024));
        let _ = plan;
        // second stdio worker
        let p2 = TransferPlanner::new(
            vec![
                spec("a", TransportKind::Stdio),
                spec("b", TransportKind::Stdio),
            ],
            ConfigurableCostModel::default(),
        );
        match p2.plan_transfer("a", "b", &meta(4096)) {
            TransferPlan::Ipc { key, target } => {
                assert_eq!(key, "a:b");
                assert_eq!(target, "b");
            }
            other => panic!("expected Ipc, got {other:?}"),
        }
    }

    #[test]
    fn lan_transfer_is_network_with_bytes() {
        let p = planner();
        match p.plan_transfer("local-1", "lan-1", &meta(8 * 1024 * 1024)) {
            TransferPlan::Network {
                key,
                target,
                estimated_bytes,
            } => {
                assert_eq!(key, "local-1:lan-1");
                assert_eq!(target, "lan-1");
                assert_eq!(estimated_bytes, 8 * 1024 * 1024);
            }
            other => panic!("expected Network, got {other:?}"),
        }
    }

    #[test]
    fn cloud_cost_is_billed() {
        let p = planner();
        let route = p.route("lan-1", "cloud-1", &meta(1024 * 1024 * 1024));
        assert!(route.total_cost.estimated_usd > 0.0);
        assert!(route.total_cost.estimated_usd < 0.06);
    }

    #[test]
    fn cost_ordering_local_lt_ipc_lt_lan_lt_cloud() {
        let model = ConfigurableCostModel::default();
        let local = model.estimate(
            &spec("a", TransportKind::Stdio),
            &spec("a", TransportKind::Stdio),
            1024,
        );
        let ipc = model.estimate(
            &spec("a", TransportKind::Stdio),
            &spec("b", TransportKind::Stdio),
            1024,
        );
        let lan = model.estimate(
            &spec("a", TransportKind::Stdio),
            &spec("c", TransportKind::Quic),
            1024,
        );
        let cloud = model.estimate(
            &spec("a", TransportKind::Stdio),
            &spec("d", TransportKind::Grpc),
            1024,
        );
        assert!(local.estimated_ms <= ipc.estimated_ms);
        assert!(ipc.estimated_ms <= lan.estimated_ms);
        assert!(lan.estimated_ms <= cloud.estimated_ms);
    }

    #[test]
    fn unknown_worker_is_unsupported() {
        let p = planner();
        match p.plan_transfer("nope", "local-1", &meta(1)) {
            TransferPlan::Unsupported { reason } => {
                assert!(reason.contains("nope"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
