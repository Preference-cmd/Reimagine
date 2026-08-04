use std::collections::HashMap;

use reimagine_backend_worker_protocol::TransportKind;

/// A connectable worker endpoint.
#[derive(Debug, Clone)]
pub struct WorkerEndpoint {
    pub id: String,
    pub transport_kind: TransportKind,
    pub address: String,
    pub capabilities: Vec<String>,
    pub device_label: String,
    pub metadata: serde_json::Value,
}

/// The state of a pooled worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is being connected.
    Connecting,
    /// Worker is ready to accept requests.
    Ready,
    /// Worker is draining in-flight requests before shutdown.
    Draining,
    /// Worker has failed and is not accepting requests.
    Failed,
    /// Worker is offline and not reachable.
    Offline,
}

/// Health status of a worker.
#[derive(Debug, Clone)]
pub struct WorkerHealth {
    pub state: WorkerState,
    pub latency_ms: Option<u64>,
    pub last_check: Option<std::time::Instant>,
    pub failure_count: u32,
}

impl Default for WorkerHealth {
    fn default() -> Self {
        Self {
            state: WorkerState::Connecting,
            latency_ms: None,
            last_check: None,
            failure_count: 0,
        }
    }
}

/// A worker tracked in the pool.
#[derive(Debug, Clone)]
pub struct PooledWorker {
    pub endpoint: WorkerEndpoint,
    pub state: WorkerState,
    pub health: WorkerHealth,
}

/// A pool of registered workers.
#[derive(Debug, Default)]
pub struct WorkerPool {
    workers: HashMap<String, PooledWorker>,
}

impl WorkerPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new worker.
    pub fn register(&mut self, endpoint: WorkerEndpoint) {
        let id = endpoint.id.clone();
        self.workers.insert(
            id,
            PooledWorker {
                endpoint,
                state: WorkerState::Connecting,
                health: WorkerHealth::default(),
            },
        );
    }

    /// Deregister a worker by id.
    pub fn deregister(&mut self, id: &str) -> Option<PooledWorker> {
        self.workers.remove(id)
    }

    /// Get a worker by id.
    pub fn get(&self, id: &str) -> Option<&PooledWorker> {
        self.workers.get(id)
    }

    /// Get a mutable reference to a worker by id.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut PooledWorker> {
        self.workers.get_mut(id)
    }

    /// Get all workers in the ready state.
    pub fn ready_workers(&self) -> Vec<&PooledWorker> {
        self.workers
            .values()
            .filter(|w| w.state == WorkerState::Ready)
            .collect()
    }

    /// Get all worker endpoints.
    pub fn all_endpoints(&self) -> Vec<&WorkerEndpoint> {
        self.workers.values().map(|w| &w.endpoint).collect()
    }

    /// Get the number of workers in the pool.
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Check if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn register_and_get() {
        let mut pool = WorkerPool::new();
        pool.register(test_endpoint("worker-1"));
        assert!(pool.get("worker-1").is_some());
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn deregister() {
        let mut pool = WorkerPool::new();
        pool.register(test_endpoint("worker-1"));
        let removed = pool.deregister("worker-1");
        assert!(removed.is_some());
        assert!(pool.get("worker-1").is_none());
        assert!(pool.is_empty());
    }

    #[test]
    fn ready_workers_filter() {
        let mut pool = WorkerPool::new();
        pool.register(test_endpoint("worker-1"));
        pool.register(test_endpoint("worker-2"));
        pool.register(test_endpoint("worker-3"));

        // Mark worker-2 as ready
        pool.get_mut("worker-2").unwrap().state = WorkerState::Ready;

        let ready = pool.ready_workers();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].endpoint.id, "worker-2");
    }

    #[test]
    fn all_endpoints() {
        let mut pool = WorkerPool::new();
        pool.register(test_endpoint("a"));
        pool.register(test_endpoint("b"));

        let endpoints = pool.all_endpoints();
        assert_eq!(endpoints.len(), 2);
        let ids: Vec<&str> = endpoints.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }
}
