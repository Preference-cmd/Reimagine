use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use reimagine_core::event::RunEvent;
use reimagine_core::model::RunId;
use reimagine_runtime::{BoxedRunEventSink, RunEventSink, RuntimeError};
use serde::Serialize;
use tauri::ipc::Channel;

/// Tauri-owned runtime event hub.
///
/// Recorded events are stored per-run for replay; subscribed channels
/// receive events live. Channel send failures are silently dropped.
#[derive(Clone)]
pub struct TauriRunEventHub {
    inner: Arc<Mutex<HubInner>>,
}

#[derive(Default)]
struct HubInner {
    events: HashMap<RunId, Vec<RunEvent>>,
    subscribers: HashMap<RunId, Channel<RunEventPayload>>,
}

/// Lightweight projection of a `RunEvent` for IPC transport.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunEventPayload {
    pub id: String,
    pub run_id: String,
    pub kind: String,
    pub node_id: Option<String>,
    pub artifact_id: Option<String>,
    pub created_at: String,
}

impl TauriRunEventHub {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HubInner::default())),
        }
    }

    /// Wrap in a `BoxedRunEventSink` for `WorkspaceHost` bootstrap.
    pub fn boxed() -> BoxedRunEventSink {
        Arc::new(Self::new()) as BoxedRunEventSink
    }

    /// Register a channel to receive events for a run.
    pub fn subscribe(&self, run_id: &RunId, channel: Channel<RunEventPayload>) {
        let mut guard = self.inner.lock().expect("hub poisoned");
        guard.subscribers.insert(run_id.clone(), channel);
    }

    /// Return all recorded events for a run (for replay).
    pub fn events_for(&self, run_id: &RunId) -> Vec<RunEvent> {
        let guard = self.inner.lock().expect("hub poisoned");
        guard.events.get(run_id).cloned().unwrap_or_default()
    }

    /// Remove a subscriber and its events from the hub (cleanup).
    pub fn unsubscribe(&self, run_id: &RunId) {
        let mut guard = self.inner.lock().expect("hub poisoned");
        guard.subscribers.remove(run_id);
        guard.events.remove(run_id);
    }
}

impl RunEventSink for TauriRunEventHub {
    fn emit(&self, event: RunEvent) -> Result<(), RuntimeError> {
        let run_id = event.run_id().clone();
        let payload = RunEventPayload::from(&event);

        let mut guard = self.inner.lock().expect("hub poisoned");

        // Record the event
        guard.events.entry(run_id.clone()).or_default().push(event);

        // Best-effort send to subscriber
        if let Some(channel) = guard.subscribers.get(&run_id) {
            if channel.send(payload).is_err() {
                // Dead subscriber — remove it silently
                guard.subscribers.remove(&run_id);
            }
        }

        Ok(())
    }
}

impl From<&RunEvent> for RunEventPayload {
    fn from(event: &RunEvent) -> Self {
        Self {
            id: event.id().as_str().to_string(),
            run_id: event.run_id().to_string(),
            kind: format!("{:?}", event.kind()),
            node_id: event.node_id().map(|n| n.to_string()),
            artifact_id: event.artifact().map(|a| a.to_string()),
            created_at: event.created_at().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::event::{RunEventKind, Timestamp};
    use reimagine_core::model::{RunId, WorkflowId, WorkflowVersion};

    fn make_event(run_id: &RunId, kind: RunEventKind) -> RunEvent {
        RunEvent::new(
            format!("evt-{}-{:?}", run_id, kind),
            run_id.clone(),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            kind,
            Timestamp::new("2026-07-04T00:00:00Z"),
        )
    }

    #[test]
    fn records_events_per_run() {
        let hub = TauriRunEventHub::new();
        let run_a = RunId::new("run-a");
        let run_b = RunId::new("run-b");

        hub.emit(make_event(&run_a, RunEventKind::RunQueued)).unwrap();
        hub.emit(make_event(&run_a, RunEventKind::RunStarted)).unwrap();
        hub.emit(make_event(&run_b, RunEventKind::RunQueued)).unwrap();

        assert_eq!(hub.events_for(&run_a).len(), 2);
        assert_eq!(hub.events_for(&run_b).len(), 1);
    }

    #[test]
    fn unknown_run_returns_empty() {
        let hub = TauriRunEventHub::new();
        assert!(hub.events_for(&RunId::new("nope")).is_empty());
    }

    #[test]
    fn unsubscribe_removes_events() {
        let hub = TauriRunEventHub::new();
        let run = RunId::new("run-c");
        hub.emit(make_event(&run, RunEventKind::RunQueued)).unwrap();
        assert_eq!(hub.events_for(&run).len(), 1);
        hub.unsubscribe(&run);
        assert!(hub.events_for(&run).is_empty());
    }

    #[test]
    fn boxed_wraps_correctly() {
        let sink = TauriRunEventHub::boxed();
        let run = RunId::new("run-d");
        sink.emit(make_event(&run, RunEventKind::RunStarted)).unwrap();
        // Can't downcast Arc<dyn RunEventSink> back, so this just tests no crash
    }
}
