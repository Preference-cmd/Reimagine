//! Per-node artifact publisher abstraction.
//!
//! The executor contract owns the *shape* of the publisher (this trait)
//! so that node executors can record artifacts without depending on the
//! runtime's concrete [`ArtifactStore`](reimagine_runtime::ArtifactStore)
//! or [`RunEventSink`](reimagine_runtime::RunEventSink). The runtime
//! provides a concrete implementation
//! ([`RuntimeNodeArtifactCapability`](reimagine_runtime::RuntimeNodeArtifactCapability))
//! at context construction time and wraps it in an `Arc<dyn
//! ArtifactPublisher>` before handing it to the executor.

use reimagine_core::model::{ArtifactId, ArtifactRef, SlotId};

/// Which host-facing event to emit alongside an artifact record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactEventKind {
    Saved,
    Preview,
}

/// Abstract publisher handed to node executors so they can record
/// artifacts during execution.
///
/// Runtime provides a concrete impl; executors only see this trait.
/// Returns `None` when the run has been cancelled — callers should
/// treat the artifact as dropped and not surface a phantom id to the
/// host.
///
/// The slot id is taken as `SlotId` (not `impl Into<SlotId>`) so the
/// trait stays object-safe — call sites that hold an `&str` or
/// `String` can call `.into()` themselves before invoking `record`.
#[async_trait::async_trait]
pub trait ArtifactPublisher: Send + Sync + 'static {
    async fn record(
        &self,
        slot_id: SlotId,
        reference: ArtifactRef,
        kind: ArtifactEventKind,
    ) -> Option<ArtifactId>;
}
