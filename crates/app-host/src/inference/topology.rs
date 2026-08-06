//! Connection topology manager for dynamic worker pool management.
//!
//! [`ConnectionTopologyManager`] owns a [`WorkerPool`], a selection
//! policy, and a [`RouterRef`] and produces an [`InferenceRouter`] for
//! the runtime. It supports dynamic worker registration and atomic
//! router swaps via [`arc_swap`].
//!
//! Landed with T10 ahead of the T13 `WorkspaceHost` integration that
//! constructs it; no call site builds one yet.
#![allow(dead_code)]

use std::sync::Arc;

use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceDescriptor, InferenceBackend,
    InferenceBackendRegistry, InferenceRouter, RouterRef,
};

use super::bridge::TopologyAwareBridgePolicy;
use super::pool::{WorkerEndpoint, WorkerPool, WorkerState};

/// Factory that creates an [`InferenceBackend`] from a [`WorkerEndpoint`].
///
/// Implementations map a worker endpoint to a concrete backend
/// suitable for registration in the [`InferenceBackendRegistry`].
/// The topology manager calls this when building a new router.
pub trait WorkerBackendFactory: Send + Sync {
    /// Attempt to build a backend for the given endpoint.
    ///
    /// Returns `None` if the endpoint cannot be served (e.g. the
    /// backend is not available on this host).
    fn build_backend(&self, endpoint: &WorkerEndpoint) -> Option<Arc<dyn InferenceBackend>>;

    /// Return the open [`Backend`] label for the factory's backends.
    fn backend_label(&self) -> Backend;
}

/// Error type for topology manager operations.
#[derive(Debug)]
pub enum TopologyError {
    /// The endpoint id is already registered.
    DuplicateEndpoint(String),
    /// The endpoint id was not found.
    EndpointNotFound(String),
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateEndpoint(id) => write!(f, "endpoint `{id}` is already registered"),
            Self::EndpointNotFound(id) => write!(f, "endpoint `{id}` not found"),
        }
    }
}

impl std::error::Error for TopologyError {}

/// Central topology manager that bridges the dynamic [`WorkerPool`]
/// and the executor-facing [`InferenceRouter`].
///
/// When workers join or leave the pool, the manager rebuilds the
/// router from the current set of ready workers and atomically swaps
/// it into the [`RouterRef`] held by executors.
///
/// # Invariants
///
/// - The [`RouterRef`] is always consistent with the pool state
///   after any successful `register_endpoint` or
///   `deregister_endpoint` call.
/// - Only workers in the [`WorkerState::Ready`] state appear in the
///   rebuilt router.
pub struct ConnectionTopologyManager {
    pool: WorkerPool,
    selection_policy: Arc<dyn reimagine_inference::BackendSelectionPolicy>,
    backend_factory: Arc<dyn WorkerBackendFactory>,
    router: RouterRef,
}

impl ConnectionTopologyManager {
    /// Create a new topology manager with the given pool, policy,
    /// and backend factory.
    ///
    /// The initial router reflects the current pool state (which may
    /// be empty). The router's bridge policy is the
    /// [`TopologyAwareBridgePolicy`] built from the pool's ready
    /// workers; with an empty pool it degrades to
    /// [`reimagine_inference::RejectAllBridgePolicy`]-identical
    /// behavior (T16).
    pub fn new(
        pool: WorkerPool,
        selection_policy: Arc<dyn reimagine_inference::BackendSelectionPolicy>,
        backend_factory: Arc<dyn WorkerBackendFactory>,
    ) -> Self {
        let router = Self::build_router_from_pool(
            &pool,
            Arc::clone(&selection_policy),
            Arc::clone(&backend_factory),
        );
        let router = Arc::new(arc_swap::ArcSwap::from_pointee(router));
        Self {
            pool,
            selection_policy,
            backend_factory,
            router,
        }
    }

    /// Access the [`RouterRef`] for executor use.
    pub fn router_ref(&self) -> &RouterRef {
        &self.router
    }

    /// Register a new worker endpoint in the pool and hot-swap the
    /// router.
    pub fn register_endpoint(&mut self, endpoint: WorkerEndpoint) -> Result<(), TopologyError> {
        let id = endpoint.id.clone();
        if self.pool.get(&id).is_some() {
            return Err(TopologyError::DuplicateEndpoint(id));
        }
        self.pool.register(endpoint);
        self.rebuild_and_swap();
        Ok(())
    }

    /// Deregister a worker endpoint by id and hot-swap the router.
    pub fn deregister_endpoint(&mut self, id: &str) -> Result<(), TopologyError> {
        if self.pool.deregister(id).is_none() {
            return Err(TopologyError::EndpointNotFound(id.to_owned()));
        }
        self.rebuild_and_swap();
        Ok(())
    }

    /// Transition a worker to the ready state and hot-swap the
    /// router so the worker becomes available for inference.
    pub fn mark_ready(&mut self, id: &str) -> Result<(), TopologyError> {
        let worker = self
            .pool
            .get_mut(id)
            .ok_or_else(|| TopologyError::EndpointNotFound(id.to_owned()))?;
        worker.state = WorkerState::Ready;
        self.rebuild_and_swap();
        Ok(())
    }

    /// Return the number of workers in the ready state.
    pub fn active_workers(&self) -> usize {
        self.pool.ready_workers().len()
    }

    /// Return all worker endpoints in the pool.
    pub fn all_endpoints(&self) -> Vec<&WorkerEndpoint> {
        self.pool.all_endpoints()
    }

    /// Return a reference to the underlying worker pool.
    pub fn pool(&self) -> &WorkerPool {
        &self.pool
    }

    /// Build a new [`InferenceRouter`] from the current pool state
    /// and atomically swap it into the [`RouterRef`].
    fn rebuild_and_swap(&self) {
        let new_router = Self::build_router_from_pool(
            &self.pool,
            Arc::clone(&self.selection_policy),
            Arc::clone(&self.backend_factory),
        );
        self.router.store(Arc::new(new_router));
    }

    /// Build an [`InferenceRouter`] from the given pool's ready
    /// workers without modifying the manager's state.
    fn build_router_from_pool(
        pool: &WorkerPool,
        selection_policy: Arc<dyn reimagine_inference::BackendSelectionPolicy>,
        backend_factory: Arc<dyn WorkerBackendFactory>,
    ) -> InferenceRouter {
        let mut registry = InferenceBackendRegistry::new();
        let backend_label = backend_factory.backend_label();

        for worker in pool.ready_workers() {
            if let Some(backend) = backend_factory.build_backend(&worker.endpoint) {
                let instance_id = BackendInstance::new(format!(
                    "{}:{}",
                    backend_label.as_str(),
                    worker.endpoint.id
                ));
                let descriptor = BackendInstanceDescriptor::new(instance_id, backend_label.clone());
                registry.register(descriptor, backend);
            }
        }

        InferenceRouter::with_policy(
            Arc::new(registry),
            selection_policy,
            Arc::new(TopologyAwareBridgePolicy::from_pool(pool, &backend_label)),
        )
    }
}

impl std::fmt::Debug for ConnectionTopologyManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionTopologyManager")
            .field("pool_size", &self.pool.len())
            .field("active_workers", &self.active_workers())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_backend_worker_protocol::TransportKind;
    use reimagine_inference::{
        Backend, BackendInstance, BridgePlan, CannedCapabilityResponse, CreateEmptyLatentRequest,
        CreateEmptyLatentResponse, FakeBackend, InferenceBackend, InferenceCapability,
        LatentContent, LatentSpaceMetadata, RuntimeLatent, StaticBackendSelectionPolicy,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn test_endpoint(id: &str) -> WorkerEndpoint {
        WorkerEndpoint {
            id: id.to_owned(),
            transport_kind: TransportKind::Stdio,
            address: "local".to_owned(),
            capabilities: vec!["echo".to_owned()],
            device_label: "cpu".to_owned(),
            trusted: true,
            metadata: serde_json::json!({}),
        }
    }

    /// A fake backend factory that creates [`FakeBackend`] instances
    /// keyed by worker id.
    struct FakeWorkerBackendFactory {
        backends: Mutex<HashMap<String, Arc<FakeBackend>>>,
    }

    impl FakeWorkerBackendFactory {
        fn new() -> Self {
            Self {
                backends: Mutex::new(HashMap::new()),
            }
        }

        fn register_fake(&self, id: &str, backend: FakeBackend) {
            self.backends
                .lock()
                .unwrap()
                .insert(id.to_owned(), Arc::new(backend));
        }
    }

    impl WorkerBackendFactory for FakeWorkerBackendFactory {
        fn build_backend(&self, endpoint: &WorkerEndpoint) -> Option<Arc<dyn InferenceBackend>> {
            let backends = self.backends.lock().unwrap();
            backends
                .get(&endpoint.id)
                .cloned()
                .map(|b| b as Arc<dyn InferenceBackend>)
        }

        fn backend_label(&self) -> Backend {
            Backend::new("fake")
        }
    }

    fn noop_selection_policy() -> Arc<dyn reimagine_inference::BackendSelectionPolicy> {
        Arc::new(StaticBackendSelectionPolicy::new(Vec::new()))
    }

    fn make_fake_backend_for(id: &str) -> FakeBackend {
        let id = id.to_owned();
        FakeBackend::new("fake").create_empty_latent(CannedCapabilityResponse::from_request(
            move |request: CreateEmptyLatentRequest| {
                Ok(CreateEmptyLatentResponse::new(RuntimeLatent::new(
                    reimagine_inference::BackendTensorHandle::new(
                        Backend::new("fake"),
                        reimagine_inference::BackendPayloadKey::new(format!("{id}-latent")),
                        reimagine_core::model::TensorDType::F32,
                        reimagine_core::model::TensorShape::new(vec![1, 4, 8, 8]),
                        "cpu",
                    ),
                    request.width(),
                    request.height(),
                    request.batch_size(),
                    4,
                    LatentSpaceMetadata::sdxl_base(),
                    LatentContent::EmptyGeometry,
                )))
            },
        ))
    }

    #[test]
    fn new_manager_builds_initial_router_from_empty_pool() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        let manager =
            ConnectionTopologyManager::new(WorkerPool::new(), noop_selection_policy(), factory);
        assert_eq!(manager.active_workers(), 0);
        assert!(manager.all_endpoints().is_empty());
        // The router should exist and have zero registered backends.
        let router = manager.router_ref().load();
        assert_eq!(router.registry().len(), 0);
    }

    #[test]
    fn register_endpoint_adds_worker_and_hot_swaps_router() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        factory.register_fake("w1", make_fake_backend_for("w1"));

        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        // Initially empty.
        assert_eq!(manager.router_ref().load().registry().len(), 0);

        manager.register_endpoint(test_endpoint("w1")).unwrap();

        // Worker is in pool but in Connecting state, so no backends
        // in the router yet.
        assert_eq!(manager.pool.len(), 1);
        assert_eq!(manager.active_workers(), 0);
        assert_eq!(manager.router_ref().load().registry().len(), 0);
    }

    #[test]
    fn mark_ready_makes_worker_appear_in_router() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        factory.register_fake("w1", make_fake_backend_for("w1"));

        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        manager.register_endpoint(test_endpoint("w1")).unwrap();
        manager.mark_ready("w1").unwrap();

        assert_eq!(manager.active_workers(), 1);
        let router = manager.router_ref().load();
        assert_eq!(router.registry().len(), 1);
    }

    #[test]
    fn register_duplicate_endpoint_returns_error() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        factory.register_fake("w1", make_fake_backend_for("w1"));

        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        manager.register_endpoint(test_endpoint("w1")).unwrap();
        let err = manager.register_endpoint(test_endpoint("w1"));
        assert!(err.is_err());
        assert!(matches!(err, Err(TopologyError::DuplicateEndpoint(_))));
    }

    #[test]
    fn deregister_endpoint_removes_worker_and_hot_swaps_router() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        factory.register_fake("w1", make_fake_backend_for("w1"));
        factory.register_fake("w2", make_fake_backend_for("w2"));

        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        manager.register_endpoint(test_endpoint("w1")).unwrap();
        manager.register_endpoint(test_endpoint("w2")).unwrap();
        manager.mark_ready("w1").unwrap();
        manager.mark_ready("w2").unwrap();

        assert_eq!(manager.active_workers(), 2);
        assert_eq!(manager.router_ref().load().registry().len(), 2);

        manager.deregister_endpoint("w1").unwrap();

        assert_eq!(manager.pool.len(), 1);
        assert_eq!(manager.active_workers(), 1);
        let router = manager.router_ref().load();
        assert_eq!(router.registry().len(), 1);
        let instances = router.registry().instances();
        assert_eq!(instances[0], BackendInstance::new("fake:w2"));
    }

    #[test]
    fn deregister_nonexistent_endpoint_returns_error() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        let err = manager.deregister_endpoint("ghost");
        assert!(err.is_err());
        assert!(matches!(err, Err(TopologyError::EndpointNotFound(_))));
    }

    #[test]
    fn build_router_appears_in_router_ref_for_executors() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        factory.register_fake("w1", make_fake_backend_for("w1"));
        factory.register_fake("w2", make_fake_backend_for("w2"));

        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        manager.register_endpoint(test_endpoint("w1")).unwrap();
        manager.register_endpoint(test_endpoint("w2")).unwrap();
        manager.mark_ready("w1").unwrap();
        manager.mark_ready("w2").unwrap();

        // Load the router via the RouterRef as executors would.
        let router = manager.router_ref().load();
        let instances = router.registry().instances();
        assert_eq!(instances.len(), 2);
        assert!(instances.contains(&BackendInstance::new("fake:w1")));
        assert!(instances.contains(&BackendInstance::new("fake:w2")));
    }

    #[test]
    fn active_workers_counts_only_ready_workers() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        factory.register_fake("w1", make_fake_backend_for("w1"));
        factory.register_fake("w2", make_fake_backend_for("w2"));
        factory.register_fake("w3", make_fake_backend_for("w3"));

        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        manager.register_endpoint(test_endpoint("w1")).unwrap();
        manager.register_endpoint(test_endpoint("w2")).unwrap();
        manager.register_endpoint(test_endpoint("w3")).unwrap();

        assert_eq!(manager.active_workers(), 0);

        manager.mark_ready("w2").unwrap();
        assert_eq!(manager.active_workers(), 1);
    }

    #[test]
    fn all_endpoints_returns_every_registered_endpoint() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());

        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        manager.register_endpoint(test_endpoint("a")).unwrap();
        manager.register_endpoint(test_endpoint("b")).unwrap();

        let endpoints = manager.all_endpoints();
        assert_eq!(endpoints.len(), 2);
        let ids: Vec<&str> = endpoints.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn router_ref_is_stable_across_rebuilds() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        factory.register_fake("w1", make_fake_backend_for("w1"));
        factory.register_fake("w2", make_fake_backend_for("w2"));

        let manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );

        // The RouterRef Arc itself stays the same; only the inner
        // InferenceRouter is swapped.
        let ref_ptr = Arc::as_ptr(&manager.router) as *const ();
        let mut manager = manager;
        manager.register_endpoint(test_endpoint("w1")).unwrap();
        manager.mark_ready("w1").unwrap();
        let ref_ptr_after = Arc::as_ptr(&manager.router) as *const ();
        assert_eq!(ref_ptr, ref_ptr_after, "RouterRef Arc should be stable");
    }

    #[test]
    fn router_bridge_policy_degrades_to_reject_all_without_ready_workers() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        let manager =
            ConnectionTopologyManager::new(WorkerPool::new(), noop_selection_policy(), factory);
        let router = manager.router_ref().load();
        assert_eq!(
            router.bridge_policy().plan_transfer(
                &Backend::new("fake"),
                &Backend::new("fake"),
                InferenceCapability::DiffusionSample,
            ),
            BridgePlan::Direct
        );
        assert!(matches!(
            router.bridge_policy().plan_transfer(
                &Backend::new("fake"),
                &Backend::new("burn"),
                InferenceCapability::DiffusionSample,
            ),
            BridgePlan::Unsupported { .. }
        ));
    }

    #[test]
    fn router_bridge_policy_is_topology_aware_after_worker_ready() {
        let factory = Arc::new(FakeWorkerBackendFactory::new());
        factory.register_fake("w1", make_fake_backend_for("w1"));

        let mut manager = ConnectionTopologyManager::new(
            WorkerPool::new(),
            noop_selection_policy(),
            Arc::clone(&factory) as Arc<dyn WorkerBackendFactory>,
        );
        manager.register_endpoint(test_endpoint("w1")).unwrap();
        manager.mark_ready("w1").unwrap();

        let router = manager.router_ref().load();
        // A single ready worker makes the label mapping unambiguous;
        // the installed policy is the topology-aware one, so an
        // unmapped cross-backend plan fails with a mapping diagnostic
        // rather than the reject-all message.
        let plan = router.bridge_policy().plan_transfer(
            &Backend::new("fake"),
            &Backend::new("burn"),
            InferenceCapability::DiffusionSample,
        );
        match plan {
            BridgePlan::Unsupported { reason } => {
                assert!(reason.contains("mapped"), "{reason}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
