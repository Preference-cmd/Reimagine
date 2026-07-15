use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use reimagine_agent::AgentEvent;
use reimagine_agent::AgentEventSink;
use reimagine_app_host::dto::AgentEventPayload;
use tauri::ipc::Channel;

/// Tauri-owned agent event hub.
///
/// Subscribed channels receive live `AgentEvent` payloads per session.
/// Channel send failures are silently dropped. No replay is needed
/// because agent events are generated during a single turn.
#[derive(Debug, Clone)]
pub struct TauriAgentEventHub {
    inner: Arc<Mutex<HubInner>>,
}

#[derive(Default)]
struct HubInner {
    subscribers: HashMap<String, Channel<AgentEventPayload>>,
}

impl fmt::Debug for HubInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HubInner")
            .field(
                "subscribers",
                &format_args!("{} channels", self.subscribers.len()),
            )
            .finish()
    }
}

impl TauriAgentEventHub {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HubInner::default())),
        }
    }

    /// Register a channel to receive events for a session.
    pub fn subscribe(&self, session_id: &str, channel: Channel<AgentEventPayload>) {
        let mut guard = self.inner.lock().expect("agent hub poisoned");
        guard.subscribers.insert(session_id.to_string(), channel);
    }

    /// Remove a subscriber and clean up.
    #[allow(dead_code)]
    pub fn unsubscribe(&self, session_id: &str) {
        let mut guard = self.inner.lock().expect("agent hub poisoned");
        guard.subscribers.remove(session_id);
    }
}

impl AgentEventSink for TauriAgentEventHub {
    fn handle(&self, event: &AgentEvent) {
        let session_id = event.session_id().to_string();
        let payload = AgentEventPayload::from(event);

        let mut guard = self.inner.lock().expect("agent hub poisoned");

        // Best-effort send to subscriber; remove dead channels silently
        if let Some(channel) = guard.subscribers.get(&session_id)
            && channel.send(payload).is_err()
        {
            guard.subscribers.remove(&session_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent::{AgentEvent, AgentMode, AgentSessionId, ProviderName, ToolName};

    fn make_session_event(kind: &str, session_id: &AgentSessionId) -> AgentEvent {
        match kind {
            "invoked" => AgentEvent::ToolInvoked {
                session_id: session_id.clone(),
                tool: ToolName::new("workflow.get"),
                id: None,
            },
            "completed" => AgentEvent::ToolCompleted {
                session_id: session_id.clone(),
                tool: ToolName::new("workflow.get"),
                id: None,
            },
            _ => AgentEvent::SessionStarted {
                session_id: session_id.clone(),
                provider: ProviderName::new("openai"),
                mode: AgentMode::Agent,
            },
        }
    }

    #[test]
    fn delivers_events_to_subscribed_channel() {
        use std::sync::mpsc;
        let hub = TauriAgentEventHub::new();
        let session_id = AgentSessionId::new("sess-1");

        let (tx, rx) = mpsc::channel();
        let channel = Channel::<AgentEventPayload>::new(move |payload| {
            tx.send(payload).ok();
            Ok(())
        });

        hub.subscribe(session_id.as_str(), channel);
        hub.handle(&make_session_event("invoked", &session_id));

        let received = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert!(received.is_ok(), "should deliver event to channel");
    }

    #[test]
    fn unknown_session_silently_ignored() {
        let hub = TauriAgentEventHub::new();
        let session_id = AgentSessionId::new("sess-unknown");

        // Should not panic
        hub.handle(&make_session_event("invoked", &session_id));
    }
}
