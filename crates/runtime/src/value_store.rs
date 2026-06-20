//! Internal store of intermediate node outputs for a single run.

use std::collections::HashMap;
use std::sync::Arc;

use reimagine_core::model::{NodeId, SlotId};
use reimagine_inference_core::ExecutionValueRetention;

use crate::value::ExecutionValue;

/// Key for a value produced by a node and stored in the run value store.
///
/// Combines the producing node and the output slot id so the same node can
/// publish multiple typed outputs (e.g. positive/negative conditioning).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OutputKey {
    node_id: NodeId,
    slot_id: SlotId,
}

impl OutputKey {
    pub fn new(node_id: impl Into<NodeId>, slot_id: impl Into<SlotId>) -> Self {
        Self {
            node_id: node_id.into(),
            slot_id: slot_id.into(),
        }
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn slot_id(&self) -> &SlotId {
        &self.slot_id
    }
}

/// Per-run value store keyed by [`OutputKey`].
///
/// Stores lightweight `Arc<ExecutionValue>` handles, not large tensor or model
/// payloads — those remain in backend-owned stores and are referenced by
/// handles carried inside [`ExecutionValue`].
///
/// The store also records the producer-declared
/// [`ExecutionValueRetention`] policy for each output. Issue 02 stores the
/// policy without acting on it; issue 05 will use it to drive early
/// release of single-use and run-scoped values.
#[derive(Debug, Default)]
pub struct RunValueStore {
    values: HashMap<OutputKey, Arc<ExecutionValue>>,
    retention: HashMap<OutputKey, ExecutionValueRetention>,
}

impl RunValueStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a value for the given output key with the default
    /// [`ExecutionValueRetention::RunScoped`] retention.
    ///
    /// Existing callers that pre-date the retention contract continue to
    /// compile and behave identically — the runtime holds the value for
    /// the entire run.
    pub fn insert(&mut self, key: OutputKey, value: Arc<ExecutionValue>) {
        self.values.insert(key.clone(), value);
        self.retention
            .insert(key, ExecutionValueRetention::RunScoped);
    }

    /// Insert a value with an explicit retention policy declared by the
    /// producer (an [`ExecutionOutput`](reimagine_inference_core::ExecutionOutput)).
    pub fn insert_with_retention(
        &mut self,
        key: OutputKey,
        value: Arc<ExecutionValue>,
        retention: ExecutionValueRetention,
    ) {
        self.values.insert(key.clone(), value);
        self.retention.insert(key, retention);
    }

    /// Get a value for the given key.
    pub fn get(&self, key: &OutputKey) -> Option<Arc<ExecutionValue>> {
        self.values.get(key).cloned()
    }

    /// Returns the producer-declared retention policy for the given key,
    /// or `None` if the key is not present.
    pub fn retention(&self, key: &OutputKey) -> Option<ExecutionValueRetention> {
        self.retention.get(key).copied()
    }

    /// Returns `true` if the store contains a value for the given key.
    pub fn contains(&self, key: &OutputKey) -> bool {
        self.values.contains_key(key)
    }

    /// Number of stored values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate all stored `(key, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&OutputKey, &Arc<ExecutionValue>)> {
        self.values.iter()
    }

    /// Iterate all stored `(key, retention)` pairs.
    pub fn retention_iter(&self) -> impl Iterator<Item = (&OutputKey, &ExecutionValueRetention)> {
        self.retention.iter()
    }

    /// Remove and return the value for the given key.
    pub fn remove(&mut self, key: &OutputKey) -> Option<Arc<ExecutionValue>> {
        self.retention.remove(key);
        self.values.remove(key)
    }
}
