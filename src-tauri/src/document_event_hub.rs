use std::fmt;
use std::sync::{Arc, Mutex};

use reimagine_app_host::DocumentChangedEvent;
use tauri::ipc::Channel;

/// Bridges the app-host broadcast stream to one long-lived UI Channel.
#[derive(Clone)]
pub struct TauriDocumentEventHub {
    inner: Arc<Mutex<Option<Channel<DocumentChangedEvent>>>>,
}

impl fmt::Debug for TauriDocumentEventHub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let subscribed = self
            .inner
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false);
        f.debug_struct("TauriDocumentEventHub")
            .field("subscribed", &subscribed)
            .finish()
    }
}

impl TauriDocumentEventHub {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    pub fn subscribe(&self, channel: Channel<DocumentChangedEvent>) {
        *self.inner.lock().expect("document event hub poisoned") = Some(channel);
    }

    pub fn send(&self, event: DocumentChangedEvent) {
        let mut guard = self.inner.lock().expect("document event hub poisoned");
        if let Some(channel) = guard.as_ref()
            && channel.send(event).is_err()
        {
            guard.take();
        }
    }

    #[allow(dead_code)]
    pub fn unsubscribe(&self) {
        self.inner
            .lock()
            .expect("document event hub poisoned")
            .take();
    }
}

impl Default for TauriDocumentEventHub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn forwards_document_events_to_channel() {
        let hub = TauriDocumentEventHub::new();
        let (tx, rx) = mpsc::channel();
        let channel = Channel::<DocumentChangedEvent>::new(move |_event| {
            let _ = tx.send(());
            Ok(())
        });
        hub.subscribe(channel);
        hub.send(DocumentChangedEvent {
            kind: "board.changed".to_owned(),
            project_id: "project-a".to_owned(),
            document_id: "board-a".to_owned(),
            version: 2,
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_ok(),
            "document event should be forwarded",
        );
    }
}
