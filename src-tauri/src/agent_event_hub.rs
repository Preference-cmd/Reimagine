//! Tauri-owned agent event hub.
//!
//! Streams embedded AgentEvents to a subscribed Channel<AgentEventPayload>
//! per session, replacing the frozen daemon JSON-RPC notification path
//! (AR-03). The hub is injected into the workspace bootstrap as the
//! AgentEventSink, so every event the harness emits (content_delta,
//! tool_*, provider_error, ...) is forwarded to the subscribing channel.
//! Sends are best-effort: dead channels are silently removed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use reimagine_agent_harness::{AgentEvent, AgentEventSink, AgentSessionId};
use reimagine_app_host::dto::AgentEventPayload;
use tauri::ipc::Channel;

#[derive(Debug, Clone)]
pub struct TauriAgentEventHub {
    inner: Arc<Mutex<HubInner>>,
}

#[derive(Default)]
struct HubInner {
    subscribers: HashMap<AgentSessionId, Channel<AgentEventPayload>>,
}

impl std::fmt::Debug for HubInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    /// Register a channel to receive agent events for a session.
    ///
    /// Only one channel per session is retained; a re-subscribe replaces
    /// the previous one (matching the run/download event hubs).
    pub fn subscribe(&self, session_id: &AgentSessionId, channel: Channel<AgentEventPayload>) {
        self.inner
            .lock()
            .expect("agent event hub poisoned")
            .subscribers
            .insert(session_id.clone(), channel);
    }

    /// Remove a subscriber. Call when the turn completes so the backend
    /// drops its sender and no further events flow.
    pub fn unsubscribe(&self, session_id: &AgentSessionId) {
        self.inner
            .lock()
            .expect("agent event hub poisoned")
            .subscribers
            .remove(session_id);
    }

    /// Send a payload to the session subscriber, dropping dead channels.
    pub fn send(&self, session_id: &AgentSessionId, payload: AgentEventPayload) {
        let mut guard = self.inner.lock().expect("agent event hub poisoned");
        if let Some(channel) = guard.subscribers.get(session_id)
            && channel.send(payload).is_err()
        {
            guard.subscribers.remove(session_id);
        }
    }

    /// Terminal turn_completed payload for a finished turn.
    ///
    /// AgentEvent has no dedicated completed variant; the desktop host
    /// sends this synthetic event after run_turn resolves so the UI sees
    /// an explicit end-of-stream marker (mirrors the frozen daemon
    /// agent.turn_completed notification).
    pub fn send_turn_completed(&self, session_id: &AgentSessionId, message: String) {
        self.send(
            session_id,
            AgentEventPayload {
                session_id: session_id.to_string(),
                kind: "turn_completed".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(message),
            },
        );
    }
    /// Terminal error payload for a failed turn.
    ///
    /// AR-11 canonicalises failure as kind "error"; the harness itself
    /// emits in-flight "provider_error" events, and tools surface
    /// "tool_failed". The host sends this synthetic event after a
    /// failed turn so the UI sees an explicit end-of-stream error
    /// marker (the counterpart of send_turn_completed).
    #[allow(dead_code)] // wired by hosts that surface end-of-turn failures (AR-11)
    pub fn send_error(&self, session_id: &AgentSessionId, message: String) {
        self.send(
            session_id,
            AgentEventPayload {
                session_id: session_id.to_string(),
                kind: "error".to_string(),
                tool_name: None,
                tool_call_id: None,
                code: None,
                message: Some(message),
            },
        );
    }
}

impl AgentEventSink for TauriAgentEventHub {
    fn handle(&self, event: &AgentEvent) {
        let session_id = event.session_id().clone();
        let payload = AgentEventPayload::from(event);
        self.send(&session_id, payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent_harness::{
        AgentEvent, AgentMode, AgentSessionId, ProviderName, ToolCallId, ToolName,
    };
    use std::sync::mpsc;

    fn session() -> AgentSessionId {
        AgentSessionId::new("sess-1")
    }

    /// In Tauri test mode the Channel callback receives InvokeResponseBody,
    /// so delivery tests only assert the send fired (download hub pattern).
    fn fake_channel() -> (Channel<AgentEventPayload>, mpsc::Receiver<()>) {
        let (tx, rx) = mpsc::channel();
        let channel = Channel::<AgentEventPayload>::new(move |_payload| {
            tx.send(()).ok();
            Ok(())
        });
        (channel, rx)
    }

    #[test]
    fn delivers_content_delta_to_subscribed_channel() {
        let hub = TauriAgentEventHub::new();
        let sid = session();
        let (channel, rx) = fake_channel();
        hub.subscribe(&sid, channel);

        hub.handle(&AgentEvent::ContentDelta {
            session_id: sid.clone(),
            text: "hello".to_string(),
        });

        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_ok(),
            "should deliver content_delta"
        );
    }

    #[test]
    fn delivers_tool_events_to_subscribed_channel() {
        let hub = TauriAgentEventHub::new();
        let sid = session();
        let (channel, rx) = fake_channel();
        hub.subscribe(&sid, channel);

        hub.handle(&AgentEvent::ToolInvoked {
            session_id: sid.clone(),
            tool: ToolName::new("workflow.read"),
            id: Some(ToolCallId::new("call-1")),
        });
        hub.handle(&AgentEvent::ToolCompleted {
            session_id: sid.clone(),
            tool: ToolName::new("workflow.read"),
            id: Some(ToolCallId::new("call-1")),
        });

        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_ok(),
            "should deliver tool_invoked"
        );
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_ok(),
            "should deliver tool_completed"
        );
    }

    #[test]
    fn delivers_provider_error_event() {
        let hub = TauriAgentEventHub::new();
        let sid = session();
        let (channel, rx) = fake_channel();
        hub.subscribe(&sid, channel);

        hub.handle(&AgentEvent::ProviderError {
            session_id: sid.clone(),
            provider: ProviderName::new("openai"),
            code: "rate_limit".to_string(),
            message: "upstream 429".to_string(),
        });

        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_ok(),
            "should deliver provider_error"
        );
    }

    #[test]
    fn sends_turn_completed_marker() {
        let hub = TauriAgentEventHub::new();
        let sid = session();
        let (channel, rx) = fake_channel();
        hub.subscribe(&sid, channel);

        hub.send_turn_completed(&sid, "done".to_string());

        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_ok(),
            "should deliver turn_completed"
        );
    }

    #[test]
    fn unknown_session_is_ignored_without_panic() {
        let hub = TauriAgentEventHub::new();
        hub.handle(&AgentEvent::ContentDelta {
            session_id: session(),
            text: "hello".to_string(),
        });
        hub.send_turn_completed(&session(), "done".to_string());
    }

    #[test]
    fn unsubscribe_stops_delivery() {
        let hub = TauriAgentEventHub::new();
        let sid = session();
        let (channel, rx) = fake_channel();
        hub.subscribe(&sid, channel);
        hub.unsubscribe(&sid);

        hub.handle(&AgentEvent::SessionStarted {
            session_id: sid.clone(),
            provider: ProviderName::new("openai"),
            mode: AgentMode::Agent,
        });

        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "no delivery after unsubscribe"
        );
    }

    // The payload projection is behavior we own on the Tauri boundary too:
    // assert kinds and fields map correctly for the events the UI consumes.
    #[test]
    fn payload_mapping_matches_ui_contract() {
        let sid = session();
        let delta = AgentEventPayload::from(&AgentEvent::ContentDelta {
            session_id: sid.clone(),
            text: "hello".to_string(),
        });
        assert_eq!(delta.kind, "content_delta");
        assert_eq!(delta.message.as_deref(), Some("hello"));

        let invoked = AgentEventPayload::from(&AgentEvent::ToolInvoked {
            session_id: sid.clone(),
            tool: ToolName::new("workflow.read"),
            id: Some(ToolCallId::new("call-1")),
        });
        assert_eq!(invoked.kind, "tool_invoked");
        assert_eq!(invoked.tool_name.as_deref(), Some("workflow.read"));
        assert_eq!(invoked.tool_call_id.as_deref(), Some("call-1"));

        let error = AgentEventPayload::from(&AgentEvent::ProviderError {
            session_id: sid.clone(),
            provider: ProviderName::new("openai"),
            code: "rate_limit".to_string(),
            message: "upstream 429".to_string(),
        });
        assert_eq!(error.kind, "provider_error");
        assert_eq!(error.code.as_deref(), Some("rate_limit"));
        assert_eq!(error.message.as_deref(), Some("upstream 429"));
    }
    // AR-11: the backend canonicalises failure as "error" while the
    // UI keeps the short-term "provider_error" spelling; both group
    // under error semantics so clients handle them uniformly.
    #[test]
    fn error_semantics_group_provider_and_tool_failures() {
        let sid = session();
        let provider_err = AgentEventPayload::from(&AgentEvent::ProviderError {
            session_id: sid.clone(),
            provider: ProviderName::new("openai"),
            code: "timeout".to_string(),
            message: "upstream timed out".to_string(),
        });
        assert!(
            provider_err.is_error(),
            "provider_error carries error semantics"
        );

        let tool_err = AgentEventPayload {
            session_id: sid.to_string(),
            kind: "tool_failed".to_string(),
            tool_name: Some("workflow.read".to_string()),
            tool_call_id: Some("call-1".to_string()),
            code: None,
            message: Some("boom".to_string()),
        };
        assert!(tool_err.is_error(), "tool_failed carries error semantics");

        let delta = AgentEventPayload::from(&AgentEvent::ContentDelta {
            session_id: sid.clone(),
            text: "hi".to_string(),
        });
        assert!(!delta.is_error(), "content is not an error");
    }

    // The terminal error marker is the failure counterpart of the
    // turn_completed marker (AR-11): a failed turn still emits an
    // explicit end-of-stream event so the UI never hangs waiting.
    #[test]
    fn send_error_emits_terminal_error_marker() {
        let hub = TauriAgentEventHub::new();
        let sid = session();
        let (channel, rx) = fake_channel();
        hub.subscribe(&sid, channel);

        hub.send_error(&sid, "upstream refused".to_string());

        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_ok(),
            "should deliver the terminal error marker"
        );
    }
}
