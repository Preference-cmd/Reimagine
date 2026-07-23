//! Host-neutral run event sink. Runtime owns the trait; core owns the event
//! payload.

use std::sync::{Arc, Mutex};

use reimagine_core::event::RunEvent;

use crate::RuntimeError;

/// Destination for run timeline events.
///
/// Implementations are provided by hosts (Tauri, future Axum, tests). The
/// runtime invokes this for each event; failures are reported/logged and do
/// not fail the run itself.
pub trait RunEventSink: Send + Sync + 'static {
    /// Emit a single event. Implementations must not panic; failures are
    /// returned to the runtime as a [`RuntimeError`]-shaped report.
    fn emit(&self, event: RunEvent) -> Result<(), RuntimeError>;

    /// Emit a batch of events in one call. The default simply forwards to
    /// [`RunEventSink::emit`] for each event.
    fn emit_many(&self, events: Vec<RunEvent>) {
        for event in events {
            let _ = self.emit(event);
        }
    }
}

/// Convenience type alias for an `Arc<dyn RunEventSink>`.
pub type BoxedRunEventSink = Arc<dyn RunEventSink>;

/// In-memory sink that appends to a `Vec`. Primarily useful in tests.
#[derive(Debug, Default, Clone)]
pub struct VecRunEventSink {
    events: Arc<Mutex<Vec<RunEvent>>>,
}

impl VecRunEventSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<RunEvent> {
        self.events.lock().expect("vec sink poisoned").clone()
    }

    pub fn len(&self) -> usize {
        self.events.lock().expect("vec sink poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.lock().expect("vec sink poisoned").is_empty()
    }
}

impl RunEventSink for VecRunEventSink {
    fn emit(&self, event: RunEvent) -> Result<(), RuntimeError> {
        self.events.lock().expect("vec sink poisoned").push(event);
        Ok(())
    }
}
