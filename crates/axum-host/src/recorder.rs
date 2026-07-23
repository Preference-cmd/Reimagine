//! In-memory `RunEventSink` used by the Axum host to expose runtime
//! events over HTTP.
//!
//! The recorder owns a `HashMap<RunId, Vec<RunEvent>>` so
//! `GET /runs/:id/events` can stream the timeline of a single run back
//! to the client. Hosts install the recorder as the runtime's
//! `RunEventSink`; the runtime emits events into the recorder and the
//! recorder fans them out into per-run buckets.
//!
//! V1 is intentionally simple: append-only, in-memory, and not
//! persisted. SSE / WebSocket delivery is a later refinement; the
//! polling endpoint reads the same recorder.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use reimagine_core::event::RunEvent;
use reimagine_core::model::RunId;
use reimagine_runtime::RunEventSink;
use reimagine_runtime::RuntimeError;

/// Append-only recorder that captures every runtime event for
/// retrieval over HTTP. Cloned `Arc`s share the same underlying
/// storage; cloning is the same as cloning a `VecRunEventSink`.
#[derive(Debug, Default, Clone)]
pub struct RunEventRecorder {
    inner: Arc<RecorderInner>,
}

#[derive(Debug, Default)]
struct RecorderInner {
    /// Per-run event log. We keep a separate `Vec` per run id so the
    /// `GET /runs/:id/events` route can answer without filtering the
    /// full event stream.
    events: Mutex<HashMap<RunId, Vec<RunEvent>>>,
}

impl RunEventRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read the full event log for a run.
    pub fn events_for(&self, run_id: &RunId) -> Vec<RunEvent> {
        let guard = self.inner.events.lock().expect("recorder poisoned");
        guard.get(run_id).cloned().unwrap_or_default()
    }

    /// Number of distinct runs that have at least one event recorded.
    pub fn active_run_count(&self) -> usize {
        let guard = self.inner.events.lock().expect("recorder poisoned");
        guard.len()
    }

    /// Drop the recorder's view of a run. Used by tests and by future
    /// cleanup paths; not invoked by HTTP handlers in V1.
    pub fn clear_run(&self, run_id: &RunId) {
        let mut guard = self.inner.events.lock().expect("recorder poisoned");
        guard.remove(run_id);
    }
}

impl RunEventSink for RunEventRecorder {
    fn emit(&self, event: RunEvent) -> Result<(), RuntimeError> {
        let run_id = event.run_id().clone();
        let mut guard = self.inner.events.lock().expect("recorder poisoned");
        let bucket = guard.entry(run_id).or_default();
        bucket.push(event);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::event::RunEventKind;
    use reimagine_core::model::{WorkflowId, WorkflowVersion};

    fn sample_event(run_id: &RunId, kind: RunEventKind) -> RunEvent {
        RunEvent::new(
            format!("evt-{}", kind_label(kind)),
            run_id.clone(),
            WorkflowId::new("wf"),
            WorkflowVersion::new(1),
            kind,
            reimagine_core::event::Timestamp::new("2026-06-13T00:00:00Z"),
        )
    }

    fn kind_label(kind: RunEventKind) -> &'static str {
        match kind {
            RunEventKind::RunQueued => "queued",
            RunEventKind::RunStarted => "started",
            RunEventKind::RunCompleted => "completed",
            _ => "other",
        }
    }

    #[test]
    fn records_events_per_run() {
        let recorder = RunEventRecorder::new();
        let run_a = RunId::new("run-a");
        let run_b = RunId::new("run-b");
        recorder
            .emit(sample_event(&run_a, RunEventKind::RunQueued))
            .unwrap();
        recorder
            .emit(sample_event(&run_a, RunEventKind::RunStarted))
            .unwrap();
        recorder
            .emit(sample_event(&run_b, RunEventKind::RunQueued))
            .unwrap();

        assert_eq!(recorder.events_for(&run_a).len(), 2);
        assert_eq!(recorder.events_for(&run_b).len(), 1);
        assert_eq!(recorder.active_run_count(), 2);
    }

    #[test]
    fn unknown_run_returns_empty() {
        let recorder = RunEventRecorder::new();
        assert!(recorder.events_for(&RunId::new("nope")).is_empty());
    }

    #[test]
    fn clear_run_removes_events() {
        let recorder = RunEventRecorder::new();
        let run = RunId::new("run-c");
        recorder
            .emit(sample_event(&run, RunEventKind::RunQueued))
            .unwrap();
        assert_eq!(recorder.active_run_count(), 1);
        recorder.clear_run(&run);
        assert_eq!(recorder.active_run_count(), 0);
    }
}
