use reimagine_core::event::Event;
use std::sync::{Arc, Mutex};
use tauri::ipc::Channel;

/// Bridges the workspace `EventBus` to connected Tauri IPC channels.
#[derive(Clone, Default)]
pub struct TauriEventBridge {
    subscribers: Arc<Mutex<Vec<Channel<Event>>>>,
}

impl std::fmt::Debug for TauriEventBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.subscribers.lock().map(|s| s.len()).unwrap_or_default();
        f.debug_struct("TauriEventBridge")
            .field("subscribers", &count)
            .finish()
    }
}

impl TauriEventBridge {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Register a channel to receive all workspace events.
    pub fn subscribe(&self, channel: Channel<Event>) {
        let mut guard = self.subscribers.lock().expect("event bridge poisoned");
        guard.push(channel);
    }

    /// Forward an event to all subscribed channels. Dead channels are removed.
    pub fn broadcast(&self, event: &Event) {
        let mut guard = self.subscribers.lock().expect("event bridge poisoned");
        guard.retain(|channel| channel.send(event.clone()).is_ok());
    }
}
