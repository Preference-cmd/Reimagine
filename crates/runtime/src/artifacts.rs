//! Per-run artifact store and the per-node artifact capability exposed to
//! executors during execution.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reimagine_core::event::{RunEvent, RunEventId, RunEventKind};
use reimagine_core::model::{
    ArtifactId, ArtifactRef, NodeId, RunId, SlotId, WorkflowId, WorkflowVersion,
};

use crate::cancellation::CancellationToken;
use crate::clock::Clock;
use crate::events::RunEventSink;

/// One artifact recorded during a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRecord {
    pub id: ArtifactId,
    pub node_id: NodeId,
    pub slot_id: SlotId,
    pub reference: ArtifactRef,
}

impl ArtifactRecord {
    pub fn new(
        id: impl Into<ArtifactId>,
        node_id: NodeId,
        slot_id: SlotId,
        reference: ArtifactRef,
    ) -> Self {
        Self {
            id: id.into(),
            node_id,
            slot_id,
            reference,
        }
    }
}

/// Per-run artifact store. Internal to the runner task.
#[derive(Debug, Default)]
pub struct ArtifactStore {
    records: HashMap<ArtifactId, ArtifactRecord>,
    order: Vec<ArtifactId>,
}

impl ArtifactStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, record: ArtifactRecord) {
        let id = record.id.clone();
        if !self.records.contains_key(&id) {
            self.order.push(id.clone());
        }
        self.records.insert(id, record);
    }

    pub fn get(&self, id: &ArtifactId) -> Option<&ArtifactRecord> {
        self.records.get(id)
    }

    pub fn iter_ordered(&self) -> impl Iterator<Item = &ArtifactRecord> {
        self.order.iter().filter_map(|id| self.records.get(id))
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Which host-facing event to emit alongside an artifact record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactEventKind {
    Saved,
    Preview,
}

/// Capability given to node executors so they can publish artifacts.
pub struct NodeArtifactCapability {
    node_id: NodeId,
    store: Arc<tokio::sync::Mutex<ArtifactStore>>,
    sink: Arc<dyn RunEventSink>,
    next_artifact_id: Arc<AtomicU64>,
    next_event_id: Arc<AtomicU64>,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    cancellation: CancellationToken,
    clock: Arc<dyn Clock>,
}

impl NodeArtifactCapability {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
        store: Arc<tokio::sync::Mutex<ArtifactStore>>,
        sink: Arc<dyn RunEventSink>,
        clock: Arc<dyn Clock>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            node_id,
            store,
            sink,
            next_artifact_id: Arc::new(AtomicU64::new(0)),
            next_event_id: Arc::new(AtomicU64::new(0)),
            run_id,
            workflow_id,
            workflow_version,
            cancellation,
            clock,
        }
    }

    /// Record an artifact and emit the corresponding event.
    ///
    /// Returns `None` when the run has been cancelled. Callers that
    /// receive `None` should treat the artifact as dropped and not surface
    /// a phantom id to the host.
    pub async fn record(
        &self,
        slot_id: impl Into<SlotId>,
        reference: ArtifactRef,
        kind: ArtifactEventKind,
    ) -> Option<ArtifactId> {
        if self.cancellation.is_cancelled() {
            return None;
        }
        let slot_id = slot_id.into();
        let id_index = self.next_artifact_id.fetch_add(1, Ordering::Relaxed);
        let id = ArtifactId::new(format!(
            "{}-{}-{}",
            self.run_id.as_str(),
            self.node_id.as_str(),
            id_index
        ));
        let record = ArtifactRecord::new(id.clone(), self.node_id.clone(), slot_id, reference);
        {
            let mut store = self.store.lock().await;
            store.record(record);
        }

        let event_kind = match kind {
            ArtifactEventKind::Saved => RunEventKind::ArtifactCreated,
            ArtifactEventKind::Preview => RunEventKind::PreviewUpdated,
        };
        let event_id_index = self.next_event_id.fetch_add(1, Ordering::Relaxed);
        let event = RunEvent::new(
            RunEventId::new(format!(
                "{}-{}-{}",
                self.run_id.as_str(),
                self.node_id.as_str(),
                event_id_index
            )),
            self.run_id.clone(),
            self.workflow_id.clone(),
            self.workflow_version,
            event_kind,
            self.clock.now(),
        )
        .with_node_id(self.node_id.clone())
        .with_artifact(id.clone());
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.sink.emit(event))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(
                    target: "reimagine_runtime",
                    run_id = %self.run_id.as_str(),
                    node_id = %self.node_id.as_str(),
                    error = %error,
                    "run event sink failed while emitting artifact event"
                );
            }
            Err(_) => {
                tracing::warn!(
                    target: "reimagine_runtime",
                    run_id = %self.run_id.as_str(),
                    node_id = %self.node_id.as_str(),
                    "run event sink panicked while emitting artifact event"
                );
            }
        }
        Some(id)
    }
}
