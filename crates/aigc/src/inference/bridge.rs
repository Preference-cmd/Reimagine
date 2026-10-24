//! Topology-aware bridge policy (T16).
//!
//! [`TopologyAwareBridgePolicy`] extends the V1
//! [`reimagine_inference::RejectAllBridgePolicy`] behavior with cross-worker
//! tensor-transfer planning. It is installed by the
//! [`ConnectionTopologyManager`](super::topology::ConnectionTopologyManager)
//! when a worker pool exists; everywhere else the inference router
//! keeps the reject-all default, so single-worker mode is
//! byte-identical to V1.

use std::collections::HashMap;

use reimagine_backend_worker_protocol::TensorMetadata;
use reimagine_backend_worker_transfer::{
    ConfigurableCostModel, TransferCost, TransferPlan, TransferPlanner, WorkerSpec,
};
use reimagine_inference::{Backend, BackendBridgePolicy, BridgePlan, InferenceCapability};

use super::pool::WorkerPool;

/// Bridge policy that plans cross-worker tensor movement through the
/// transfer planner when two backends resolve to *different* workers.
///
/// # Backend -> worker mapping
///
/// The policy receives [`Backend`] labels only, never worker ids, so
/// the label -> worker-id relationship is **constructor-injected**
/// via `backend_to_worker`. Builders must only register labels that
/// are unambiguous: [`TopologyAwareBridgePolicy::from_pool`] maps a
/// backend label to a worker id only when exactly one ready worker
/// carries that label; ambiguous labels are omitted and resolve to
/// `Unsupported` rather than guessing.
///
/// # Behavior
///
/// - Same backend -> [`BridgePlan::Direct`] (never consults the
///   planner).
/// - No planner (no topology manager / empty worker set) -> the same
///   `Direct` / `Unsupported` classification as
///   [`reimagine_inference::RejectAllBridgePolicy`].
/// - Different backends -> map both to worker ids and ask the
///   planner for a route. Any non-`Local` route is priced as a
///   [`BridgePlan::CrossWorker`] plan; `Local` and `Unsupported`
///   routes become `Unsupported`.
///
/// `plan_transfer` is synchronous (trait contract) and must not block
/// request dispatch: the planner's `route` is a pure,
/// allocation-only function over the constructor-provided worker
/// set.
#[derive(Debug, Clone)]
pub struct TopologyAwareBridgePolicy {
    planner: Option<TransferPlanner<ConfigurableCostModel>>,
    backend_to_worker: HashMap<Backend, String>,
}

impl TopologyAwareBridgePolicy {
    /// Policy with no planner and no mapping: identical to
    /// [`reimagine_inference::RejectAllBridgePolicy`] (`Direct` same backend /
    /// `Unsupported` otherwise).
    pub fn empty() -> Self {
        Self {
            planner: None,
            backend_to_worker: HashMap::new(),
        }
    }

    /// Construct a policy with an explicit planner and
    /// backend-label -> worker-id mapping.
    ///
    /// Pass `None` for the planner to keep reject-all behavior even
    /// when a mapping exists (e.g. workers are known but no planner
    /// is available).
    pub fn new(
        planner: Option<TransferPlanner<ConfigurableCostModel>>,
        backend_to_worker: HashMap<Backend, String>,
    ) -> Self {
        Self {
            planner,
            backend_to_worker,
        }
    }

    /// Build a policy from a worker pool.
    ///
    /// The planner is populated from the pool's *ready* workers; an
    /// empty pool yields `None` and reject-all behavior. The
    /// `backend_label` is mapped to a worker id only when exactly one
    /// ready worker carries it — multiple workers behind one label
    /// (the V1 shared-label topology router) are indistinguishable at
    /// the [`Backend`] level and must not be guessed.
    pub fn from_pool(pool: &WorkerPool, backend_label: &Backend) -> Self {
        let ready = pool.ready_workers();
        if ready.is_empty() {
            return Self::empty();
        }
        let workers = ready
            .iter()
            .map(|worker| WorkerSpec {
                id: worker.endpoint.id.clone(),
                transport_kind: worker.endpoint.transport_kind,
            })
            .collect();
        let planner = TransferPlanner::new(workers, ConfigurableCostModel::default());
        let mut backend_to_worker = HashMap::new();
        if let [worker] = ready.as_slice() {
            backend_to_worker.insert(backend_label.clone(), worker.endpoint.id.clone());
        }
        Self::new(Some(planner), backend_to_worker)
    }
}

impl BackendBridgePolicy for TopologyAwareBridgePolicy {
    fn plan_transfer(
        &self,
        source_backend: &Backend,
        target_backend: &Backend,
        capability: InferenceCapability,
    ) -> BridgePlan {
        if source_backend == target_backend {
            return BridgePlan::Direct;
        }
        let Some(planner) = &self.planner else {
            return BridgePlan::Unsupported {
                reason: "no transfer planner (topology manager not configured)".to_owned(),
            };
        };
        let Some(source_worker) = self.backend_to_worker.get(source_backend) else {
            return BridgePlan::Unsupported {
                reason: format!("backend `{source_backend}` is not mapped to a worker"),
            };
        };
        let Some(target_worker) = self.backend_to_worker.get(target_backend) else {
            return BridgePlan::Unsupported {
                reason: format!("backend `{target_backend}` is not mapped to a worker"),
            };
        };
        let route = planner.route(source_worker, target_worker, &payload_metadata(capability));
        match route.hops.first() {
            Some(TransferPlan::Network { .. })
            | Some(TransferPlan::Ipc { .. })
            | Some(TransferPlan::ObjectStorage { .. }) => BridgePlan::CrossWorker {
                source_worker: source_worker.clone(),
                target_worker: target_worker.clone(),
                estimated_cost: format_cost(&route.total_cost),
            },
            Some(TransferPlan::Local { .. }) => BridgePlan::Unsupported {
                reason: format!(
                    "planner priced a local route for distinct backends `{source_backend}` (worker `{source_worker}`) and `{target_backend}` (worker `{target_worker}`)"
                ),
            },
            Some(TransferPlan::Unsupported { reason }) => BridgePlan::Unsupported {
                reason: reason.clone(),
            },
            None => BridgePlan::Unsupported {
                reason: "transfer planner returned an empty route".to_owned(),
            },
        }
    }
}

/// Cost-estimation payload metadata.
///
/// The bridge policy sees capabilities, not tensor sizes, so planning
/// uses documented per-capability size defaults. These are
/// *estimates* for path/cost selection only — the real sizes land
/// with the T17 transfer executor.
fn payload_metadata(capability: InferenceCapability) -> TensorMetadata {
    let size_bytes = match capability {
        // Model weights are the heaviest payload; conservative 1 GiB.
        InferenceCapability::LoadBundle => 1024 * 1024 * 1024,
        // SDXL 4x128x128 f32 latents (~256 KiB) — used by every
        // latent-shaped capability.
        InferenceCapability::CreateEmptyLatent
        | InferenceCapability::DiffusionSample
        | InferenceCapability::LatentDecode
        | InferenceCapability::LatentEncode => 256 * 1024,
        // CLIP text-embedding output (~128 KiB).
        InferenceCapability::TextEncode => 128 * 1024,
        // Encoded images (~3 MiB).
        InferenceCapability::ImageImport
        | InferenceCapability::ImageSave
        | InferenceCapability::ImagePreview => 3 * 1024 * 1024,
    };
    TensorMetadata {
        dtype: "f32".to_owned(),
        shape: Vec::new(),
        size_bytes,
        backend_format: "cross-worker".to_owned(),
    }
}

/// Human-readable cost string, e.g. `"~2.4ms"` or `"~7.4ms / ~$0.0500"`.
fn format_cost(cost: &TransferCost) -> String {
    let mut text = format!("~{:.1}ms", cost.estimated_ms);
    if cost.estimated_usd > 0.0 {
        text.push_str(&format!(" / ~${:.6}", cost.estimated_usd));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_backend_worker_protocol::TransportKind;

    fn endpoint(id: &str, transport_kind: TransportKind) -> super::super::pool::WorkerEndpoint {
        super::super::pool::WorkerEndpoint {
            id: id.to_owned(),
            transport_kind,
            address: "local".to_owned(),
            capabilities: Vec::new(),
            device_label: "cpu".to_owned(),
            trusted: true,
            metadata: serde_json::json!({}),
        }
    }

    fn mark_ready(pool: &mut WorkerPool, id: &str) {
        pool.get_mut(id).unwrap().state = super::super::pool::WorkerState::Ready;
    }

    fn policy_with_workers() -> TopologyAwareBridgePolicy {
        let planner = TransferPlanner::new(
            vec![
                WorkerSpec {
                    id: "local-1".to_owned(),
                    transport_kind: TransportKind::Stdio,
                },
                WorkerSpec {
                    id: "lan-1".to_owned(),
                    transport_kind: TransportKind::Quic,
                },
                WorkerSpec {
                    id: "cloud-1".to_owned(),
                    transport_kind: TransportKind::Grpc,
                },
            ],
            ConfigurableCostModel::default(),
        );
        let mut mapping = HashMap::new();
        mapping.insert(Backend::new("local"), "local-1".to_owned());
        mapping.insert(Backend::new("lan"), "lan-1".to_owned());
        mapping.insert(Backend::new("cloud"), "cloud-1".to_owned());
        TopologyAwareBridgePolicy::new(Some(planner), mapping)
    }

    #[test]
    fn same_backend_is_direct_even_without_planner() {
        let policy = TopologyAwareBridgePolicy::empty();
        let plan = policy.plan_transfer(
            &Backend::new("local"),
            &Backend::new("local"),
            InferenceCapability::CreateEmptyLatent,
        );
        assert_eq!(plan, BridgePlan::Direct);
    }

    #[test]
    fn without_planner_cross_backend_matches_reject_all() {
        let policy = TopologyAwareBridgePolicy::empty();
        let plan = policy.plan_transfer(
            &Backend::new("local"),
            &Backend::new("lan"),
            InferenceCapability::CreateEmptyLatent,
        );
        match plan {
            BridgePlan::Unsupported { reason } => {
                assert!(reason.contains("no transfer planner"), "{reason}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn lan_different_workers_produce_cross_worker_plan() {
        let policy = policy_with_workers();
        let plan = policy.plan_transfer(
            &Backend::new("local"),
            &Backend::new("lan"),
            InferenceCapability::CreateEmptyLatent,
        );
        match plan {
            BridgePlan::CrossWorker {
                source_worker,
                target_worker,
                estimated_cost,
            } => {
                assert_eq!(source_worker, "local-1");
                assert_eq!(target_worker, "lan-1");
                assert!(estimated_cost.contains("~"), "{estimated_cost}");
                assert!(estimated_cost.contains("ms"), "{estimated_cost}");
                // LAN is free in the default cost model.
                assert!(!estimated_cost.contains('$'), "{estimated_cost}");
            }
            other => panic!("expected CrossWorker, got {other:?}"),
        }
    }

    #[test]
    fn cloud_different_workers_produce_cross_worker_plan_with_cost() {
        let policy = policy_with_workers();
        let plan = policy.plan_transfer(
            &Backend::new("lan"),
            &Backend::new("cloud"),
            InferenceCapability::CreateEmptyLatent,
        );
        match plan {
            BridgePlan::CrossWorker {
                source_worker,
                target_worker,
                estimated_cost,
            } => {
                assert_eq!(source_worker, "lan-1");
                assert_eq!(target_worker, "cloud-1");
                assert!(estimated_cost.contains("ms"), "{estimated_cost}");
                // Cloud egress is billed in the default cost model.
                assert!(estimated_cost.contains('$'), "{estimated_cost}");
            }
            other => panic!("expected CrossWorker, got {other:?}"),
        }
    }

    #[test]
    fn unmapped_backend_is_unsupported() {
        let policy = policy_with_workers();
        let plan = policy.plan_transfer(
            &Backend::new("local"),
            &Backend::new("unknown"),
            InferenceCapability::CreateEmptyLatent,
        );
        match plan {
            BridgePlan::Unsupported { reason } => {
                assert!(reason.contains("unknown"), "{reason}");
                assert!(reason.contains("mapped"), "{reason}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn local_route_across_distinct_backends_is_unsupported() {
        let planner = TransferPlanner::new(
            vec![WorkerSpec {
                id: "w1".to_owned(),
                transport_kind: TransportKind::Stdio,
            }],
            ConfigurableCostModel::default(),
        );
        let mut mapping = HashMap::new();
        mapping.insert(Backend::new("a"), "w1".to_owned());
        mapping.insert(Backend::new("b"), "w1".to_owned());
        let policy = TopologyAwareBridgePolicy::new(Some(planner), mapping);
        let plan = policy.plan_transfer(
            &Backend::new("a"),
            &Backend::new("b"),
            InferenceCapability::CreateEmptyLatent,
        );
        match plan {
            BridgePlan::Unsupported { reason } => {
                assert!(reason.contains("local route"), "{reason}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn unknown_worker_in_planner_is_unsupported() {
        let planner = TransferPlanner::new(
            vec![WorkerSpec {
                id: "w1".to_owned(),
                transport_kind: TransportKind::Stdio,
            }],
            ConfigurableCostModel::default(),
        );
        let mut mapping = HashMap::new();
        mapping.insert(Backend::new("a"), "w1".to_owned());
        mapping.insert(Backend::new("b"), "ghost".to_owned());
        let policy = TopologyAwareBridgePolicy::new(Some(planner), mapping);
        let plan = policy.plan_transfer(
            &Backend::new("a"),
            &Backend::new("b"),
            InferenceCapability::CreateEmptyLatent,
        );
        match plan {
            BridgePlan::Unsupported { reason } => {
                assert!(reason.contains("ghost"), "{reason}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn from_pool_with_empty_pool_is_reject_all_equivalent() {
        let policy =
            TopologyAwareBridgePolicy::from_pool(&WorkerPool::new(), &Backend::new("fake"));
        assert!(policy.planner.is_none());
        assert!(policy.backend_to_worker.is_empty());
        let plan = policy.plan_transfer(
            &Backend::new("fake"),
            &Backend::new("burn"),
            InferenceCapability::DiffusionSample,
        );
        assert!(matches!(plan, BridgePlan::Unsupported { .. }));
    }

    #[test]
    fn from_pool_maps_unique_label_and_builds_planner() {
        let mut pool = WorkerPool::new();
        pool.register(endpoint("w1", TransportKind::Quic));
        mark_ready(&mut pool, "w1");
        let policy = TopologyAwareBridgePolicy::from_pool(&pool, &Backend::new("fake"));
        assert!(policy.planner.is_some());
        assert_eq!(
            policy
                .backend_to_worker
                .get(&Backend::new("fake"))
                .map(String::as_str),
            Some("w1")
        );
        assert_eq!(
            policy.plan_transfer(
                &Backend::new("fake"),
                &Backend::new("fake"),
                InferenceCapability::DiffusionSample,
            ),
            BridgePlan::Direct
        );
    }

    #[test]
    fn from_pool_omits_ambiguous_shared_label() {
        let mut pool = WorkerPool::new();
        pool.register(endpoint("w1", TransportKind::Quic));
        pool.register(endpoint("w2", TransportKind::Quic));
        mark_ready(&mut pool, "w1");
        mark_ready(&mut pool, "w2");
        let policy = TopologyAwareBridgePolicy::from_pool(&pool, &Backend::new("fake"));
        assert!(policy.planner.is_some());
        // Two workers behind one label are indistinguishable at the
        // Backend level; the label must not be mapped to either.
        assert!(policy.backend_to_worker.is_empty());
        let plan = policy.plan_transfer(
            &Backend::new("fake"),
            &Backend::new("burn"),
            InferenceCapability::DiffusionSample,
        );
        assert!(matches!(plan, BridgePlan::Unsupported { .. }));
    }
}
