use tokio::sync::broadcast;

/// Stable host-level notification emitted after a project-owned document is
/// durably changed. Host adapters may forward this value directly to UI
/// transports; the domain services remain unaware of Tauri or Axum.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentChangedEvent {
    pub kind: String,
    pub project_id: String,
    pub document_id: String,
    pub version: u64,
}

/// Broadcast source shared by the project, board, and workflow services in a
/// single WorkspaceHost. A lagging adapter can resubscribe without affecting
/// document mutation semantics.
#[derive(Debug, Clone)]
pub struct DocumentEventBus {
    sender: broadcast::Sender<DocumentChangedEvent>,
}

impl DocumentEventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(128);
        Self { sender }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DocumentChangedEvent> {
        self.sender.subscribe()
    }

    pub fn publish(&self, event: DocumentChangedEvent) {
        let _ = self.sender.send(event);
    }
}

impl Default for DocumentEventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn publishes_stable_camel_case_document_event() {
        let bus = DocumentEventBus::new();
        let mut events = bus.subscribe();
        bus.publish(DocumentChangedEvent {
            kind: "workflow.changed".to_owned(),
            project_id: "project-a".to_owned(),
            document_id: "workflow-a".to_owned(),
            version: 3,
        });
        let event = events.recv().await.expect("event published");
        assert_eq!(event.project_id, "project-a");
        assert_eq!(event.document_id, "workflow-a");
        let json = serde_json::to_value(event).expect("event serializes");
        assert_eq!(json["projectId"], "project-a");
        assert_eq!(json["documentId"], "workflow-a");
        assert_eq!(json["version"], 3);
    }
}
