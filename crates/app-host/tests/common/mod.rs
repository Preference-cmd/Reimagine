pub mod sdxl_workflows;

use reimagine_core::event::{Event, EventBus, EventReceiver, RunEvent};
use std::sync::Mutex;

pub struct TestRunEventCollector {
    rx: Mutex<EventReceiver>,
    events: Mutex<Vec<RunEvent>>,
}

impl TestRunEventCollector {
    pub fn new(bus: &EventBus) -> Self {
        Self {
            rx: Mutex::new(bus.subscribe()),
            events: Mutex::new(Vec::new()),
        }
    }

    pub fn events(&self) -> Vec<RunEvent> {
        let mut rx = self.rx.lock().unwrap();
        let mut events = self.events.lock().unwrap();
        while let Ok(event) = rx.try_recv() {
            if let Event::Run(run_event) = event {
                events.push(run_event);
            }
        }
        events.clone()
    }
}
