use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use super::run_event::RunEvent;

/// A notification published on the [`EventBus`].
///
/// Serialized with a `type` tag: `document` (a document reached a new
/// version), `run` (workflow run timeline) or `module` (a module-owned event
/// whose `kind` starts with the module's namespace, such as `aigc.`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Event {
    #[serde(rename = "document")]
    Document {
        kind: String,
        #[serde(rename = "projectId")]
        project_id: String,
        #[serde(rename = "documentId")]
        document_id: String,
        version: u64,
    },
    #[serde(rename = "run")]
    Run(RunEvent),
    #[serde(rename = "module")]
    Module {
        kind: String,
        payload: serde_json::Value,
    },
}

impl Event {
    pub fn document(
        kind: impl Into<String>,
        project_id: impl Into<String>,
        document_id: impl Into<String>,
        version: u64,
    ) -> Self {
        Self::Document {
            kind: kind.into(),
            project_id: project_id.into(),
            document_id: document_id.into(),
            version,
        }
    }

    pub fn module(kind: impl Into<String>, payload: serde_json::Value) -> Self {
        Self::Module {
            kind: kind.into(),
            payload,
        }
    }
}

/// Receiving end of an [`EventBus`] subscription.
pub type EventReceiver = broadcast::Receiver<Event>;

/// In-process broadcast of [`Event`]s.
///
/// Events are notifications, not state: they are not persisted or replayed,
/// and a subscriber that falls more than [`EventBus::CAPACITY`] events behind
/// skips the missed ones. Subscribers that need the current state read it
/// from the owning service.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: broadcast::Sender<Event>,
}

impl EventBus {
    pub const CAPACITY: usize = 1024;

    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(Self::CAPACITY);
        Self { sender }
    }

    pub fn subscribe(&self) -> EventReceiver {
        self.sender.subscribe()
    }

    /// Publish to the current subscribers; with none, the event is dropped.
    pub fn publish(&self, event: Event) {
        let _ = self.sender.send(event);
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{RunEventKind, Timestamp};
    use crate::model::{DocumentVersion, RunId, WorkflowId};

    #[test]
    fn bus_publish_without_subscribers_does_not_fail() {
        let bus = EventBus::new();
        bus.publish(Event::document("test", "proj", "doc", 1));
    }

    #[tokio::test]
    async fn bus_broadcasts_events_to_subscribers() {
        let bus = EventBus::new();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        let event = Event::document("board.changed", "p1", "b1", 1);
        bus.publish(event.clone());

        assert_eq!(rx1.recv().await.unwrap(), event);
        assert_eq!(rx2.recv().await.unwrap(), event);
    }

    #[test]
    fn event_serde_document_shape() {
        let event = Event::document("board.changed", "proj-1", "doc-1", 42);
        let val = serde_json::to_value(&event).expect("serialize document event");
        assert_eq!(
            val,
            serde_json::json!({
                "type": "document",
                "kind": "board.changed",
                "projectId": "proj-1",
                "documentId": "doc-1",
                "version": 42
            })
        );

        let roundtrip: Event = serde_json::from_value(val).expect("deserialize document event");
        assert_eq!(roundtrip, event);
    }

    #[test]
    fn event_serde_run_shape() {
        let run_event = RunEvent::new(
            "evt-1",
            RunId::new("run-1"),
            WorkflowId::new("wf-1"),
            DocumentVersion::new(1),
            RunEventKind::RunStarted,
            Timestamp::new("2026-10-25T00:00:00Z"),
        );
        let event = Event::Run(run_event.clone());
        let val = serde_json::to_value(&event).expect("serialize run event");
        assert_eq!(val["type"], "run");
        assert_eq!(val["id"], "evt-1");
        assert_eq!(val["runId"], "run-1");
        assert_eq!(val["workflowId"], "wf-1");
        assert_eq!(val["workflowVersion"], 1);
        assert_eq!(val["kind"], "RunStarted");

        let roundtrip: Event = serde_json::from_value(val).expect("deserialize run event");
        assert_eq!(roundtrip, event);
    }

    #[test]
    fn event_serde_module_shape() {
        let payload = serde_json::json!({
            "downloadId": "dl-1",
            "status": "in_progress",
            "bytesDownloaded": 1024
        });
        let event = Event::module("aigc.download.progress", payload.clone());
        let val = serde_json::to_value(&event).expect("serialize module event");
        assert_eq!(
            val,
            serde_json::json!({
                "type": "module",
                "kind": "aigc.download.progress",
                "payload": payload
            })
        );

        let roundtrip: Event = serde_json::from_value(val).expect("deserialize module event");
        assert_eq!(roundtrip, event);
    }
}
