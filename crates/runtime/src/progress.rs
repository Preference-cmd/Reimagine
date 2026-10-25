use std::sync::{Arc, Mutex};

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::event::{Event, EventBus, RunEvent, RunEventId, RunEventKind};
use reimagine_core::model::{DocumentVersion, NodeId, RunId, WorkflowId};
use reimagine_core::task::{Progress, ProgressSink};

use crate::clock::Clock;

pub(crate) struct RuntimeInferenceProgressSink {
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: DocumentVersion,
    node_id: NodeId,
    correlation_id: Option<CorrelationId>,
    event_bus: EventBus,
    clock: Arc<dyn Clock>,
    last_sequence: Mutex<Option<u64>>,
}

impl RuntimeInferenceProgressSink {
    // MB05 deliberately uses lossless progress recording: every strictly
    // increasing sequence is emitted to the host bus. There is no pressure
    // coalescing in this adapter.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: DocumentVersion,
        node_id: NodeId,
        correlation_id: Option<CorrelationId>,
        event_bus: EventBus,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            workflow_version,
            node_id,
            correlation_id,
            event_bus,
            clock,
            last_sequence: Mutex::new(None),
        }
    }
}

impl ProgressSink for RuntimeInferenceProgressSink {
    fn report(&self, progress: Progress) {
        let mut last = self.last_sequence.lock().expect("progress sink poisoned");
        if last.is_some_and(|sequence| progress.sequence <= sequence) {
            return;
        }

        let mut event = RunEvent::new(
            RunEventId::new(format!(
                "{}-{}-progress-{}",
                self.run_id.as_str(),
                self.node_id.as_str(),
                progress.sequence
            )),
            self.run_id.clone(),
            self.workflow_id.clone(),
            self.workflow_version,
            RunEventKind::NodeProgress,
            self.clock.now(),
        )
        .with_node_id(self.node_id.clone())
        .with_progress(progress.clone());
        if let Some(correlation_id) = &self.correlation_id {
            event = event.with_correlation_id(correlation_id.clone());
        }
        self.event_bus.publish(Event::Run(event));
        *last = Some(progress.sequence);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reimagine_core::event::{Event, EventBus};
    use reimagine_core::model::{DocumentVersion, NodeId, RunId, WorkflowId};
    use reimagine_core::task::{Progress, ProgressSink};

    use crate::SystemClock;

    use super::RuntimeInferenceProgressSink;

    #[test]
    fn runtime_progress_sink_emits_only_monotonic_events() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        let sink = RuntimeInferenceProgressSink::new(
            RunId::new("run-progress"),
            WorkflowId::new("workflow-progress"),
            DocumentVersion::new(1),
            NodeId::new("node-progress"),
            None,
            bus.clone(),
            Arc::new(SystemClock),
        );

        for sequence in [1, 1, 0, 2] {
            sink.report(Progress {
                sequence,
                completed: sequence,
                total: Some(2),
                message: Some(format!("step {sequence}")),
            });
        }

        let mut recorded = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let Event::Run(run_event) = event {
                recorded.push(run_event);
            }
        }
        let sequences = recorded
            .iter()
            .map(|event| event.progress().expect("structured progress").sequence())
            .collect::<Vec<_>>();
        assert_eq!(sequences, vec![1, 2]);
    }
}
