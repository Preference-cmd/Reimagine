//! Internal store of intermediate node outputs for a single run.

use std::collections::HashMap;
use std::sync::Arc;

use reimagine_core::model::{NodeId, SlotId};
use reimagine_inference::ExecutionValueRetention;

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

/// Single record for a stored intermediate value, pairing the value with
/// the producer-declared [`ExecutionValueRetention`].
///
/// Issue 05 collapses the previous `values` / `retention` parallel maps
/// into a single record map so retention and value lifetimes are kept
/// together for the lifetime of the record.
#[derive(Debug, Clone)]
pub struct RuntimeValueRecord {
    value: Arc<ExecutionValue>,
    retention: ExecutionValueRetention,
}

impl RuntimeValueRecord {
    pub fn new(value: Arc<ExecutionValue>, retention: ExecutionValueRetention) -> Self {
        Self { value, retention }
    }

    pub fn value(&self) -> &Arc<ExecutionValue> {
        &self.value
    }

    pub fn retention(&self) -> ExecutionValueRetention {
        self.retention
    }

    /// Consume the record and return the inner `Arc<ExecutionValue>`.
    pub fn into_value(self) -> Arc<ExecutionValue> {
        self.value
    }
}

/// Per-run value store keyed by [`OutputKey`].
///
/// Stores lightweight `Arc<ExecutionValue>` handles, not large tensor or model
/// payloads — those remain in backend-owned stores and are referenced by
/// handles carried inside [`ExecutionValue`].
///
/// Each stored record carries the producer-declared
/// [`ExecutionValueRetention`] policy. The runtime uses the policy to
/// decide when to drop its run-scoped `Arc<ExecutionValue>` references:
/// `SingleUse` values are dropped after their unique consumer
/// completes, `RunScoped` values live until terminal cleanup, and
/// `WorkspaceScoped` values are treated as opaque run-owned handles
/// whose drop is detached from any backend cache eviction.
#[derive(Debug, Default)]
pub struct RunValueStore {
    records: HashMap<OutputKey, RuntimeValueRecord>,
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
        self.records.insert(
            key,
            RuntimeValueRecord::new(value, ExecutionValueRetention::RunScoped),
        );
    }

    /// Insert a value with an explicit retention policy declared by the
    /// producer (an [`ExecutionOutput`](reimagine_inference::ExecutionOutput)).
    pub fn insert_with_retention(
        &mut self,
        key: OutputKey,
        value: Arc<ExecutionValue>,
        retention: ExecutionValueRetention,
    ) {
        self.records
            .insert(key, RuntimeValueRecord::new(value, retention));
    }

    /// Get a value for the given key.
    pub fn get(&self, key: &OutputKey) -> Option<Arc<ExecutionValue>> {
        self.records.get(key).map(|r| r.value().clone())
    }

    /// Returns the producer-declared retention policy for the given key,
    /// or `None` if the key is not present.
    pub fn retention(&self, key: &OutputKey) -> Option<ExecutionValueRetention> {
        self.records.get(key).map(|r| r.retention())
    }

    /// Returns `true` if the store contains a value for the given key.
    pub fn contains(&self, key: &OutputKey) -> bool {
        self.records.contains_key(key)
    }

    /// Number of stored values.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Iterate all stored `(key, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&OutputKey, &Arc<ExecutionValue>)> {
        self.records.iter().map(|(k, r)| (k, r.value()))
    }

    /// Iterate all stored `(key, record)` pairs.
    pub fn record_iter(&self) -> impl Iterator<Item = (&OutputKey, &RuntimeValueRecord)> {
        self.records.iter()
    }

    /// Remove and return the value for the given key.
    pub fn remove(&mut self, key: &OutputKey) -> Option<Arc<ExecutionValue>> {
        self.records.remove(key).map(|r| r.into_value())
    }

    /// Drop every stored record, releasing the runtime's run-scoped
    /// `Arc<ExecutionValue>` references. Backend-owned payloads that
    /// outlive the run must hold their own handles.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}
