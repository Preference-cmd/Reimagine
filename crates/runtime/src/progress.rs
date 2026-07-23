use std::sync::{Arc, Mutex};

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::event::{NodeProgress, RunEvent, RunEventId, RunEventKind};
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use reimagine_inference::{InferenceProgress, InferenceProgressSink};

use crate::clock::Clock;
use crate::events::RunEventSink;

pub(crate) struct RuntimeInferenceProgressSink {
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    node_id: NodeId,
    correlation_id: Option<CorrelationId>,
    sink: Arc<dyn RunEventSink>,
    clock: Arc<dyn Clock>,
    last_sequence: Mutex<Option<u64>>,
}

impl RuntimeInferenceProgressSink {
    // MB05 deliberately uses lossless progress recording: every strictly
    // increasing sequence is emitted to the host sink. There is no pressure
    // coalescing in this adapter, so replay observes the same accepted sequence
    // that live consumers saw.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
        correlation_id: Option<CorrelationId>,
        sink: Arc<dyn RunEventSink>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            run_id,
            workflow_id,
            workflow_version,
            node_id,
            correlation_id,
            sink,
            clock,
            last_sequence: Mutex::new(None),
        }
    }
}

impl InferenceProgressSink for RuntimeInferenceProgressSink {
    fn report(&self, progress: InferenceProgress) {
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
        .with_progress(NodeProgress::new(
            progress.sequence,
            progress.completed,
            progress.total,
            progress.message,
        ));
        if let Some(correlation_id) = &self.correlation_id {
            event = event.with_correlation_id(correlation_id.clone());
        }
        if matches!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.sink.emit(event))),
            Ok(Ok(()))
        ) {
            *last = Some(progress.sequence);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use reimagine_core::event::RunEvent;
    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
    use reimagine_inference::{InferenceProgress, InferenceProgressSink};

    use crate::{RunEventSink, RuntimeError, SystemClock, VecRunEventSink};

    use super::RuntimeInferenceProgressSink;

    #[test]
    fn runtime_progress_sink_emits_only_monotonic_replayable_events() {
        let events = Arc::new(VecRunEventSink::new());
        let sink = RuntimeInferenceProgressSink::new(
            RunId::new("run-progress"),
            WorkflowId::new("workflow-progress"),
            WorkflowVersion::new(1),
            NodeId::new("node-progress"),
            None,
            events.clone(),
            Arc::new(SystemClock),
        );

        for sequence in [1, 1, 0, 2] {
            sink.report(InferenceProgress {
                sequence,
                completed: sequence,
                total: Some(2),
                message: Some(format!("step {sequence}")),
            });
        }

        let recorded = events.events();
        let sequences = recorded
            .iter()
            .map(|event| event.progress().expect("structured progress").sequence())
            .collect::<Vec<_>>();
        assert_eq!(sequences, vec![1, 2]);
    }

    #[derive(Default)]
    struct FailOnceSink {
        calls: AtomicUsize,
        events: Mutex<Vec<RunEvent>>,
    }

    impl RunEventSink for FailOnceSink {
        fn emit(&self, event: RunEvent) -> Result<(), RuntimeError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(RuntimeError::EventSink {
                    message: "transient failure".to_owned(),
                });
            }
            self.events.lock().expect("events").push(event);
            Ok(())
        }
    }

    #[test]
    fn failed_progress_delivery_does_not_consume_the_sequence() {
        let events = Arc::new(FailOnceSink::default());
        let sink = RuntimeInferenceProgressSink::new(
            RunId::new("run-progress-retry"),
            WorkflowId::new("workflow-progress-retry"),
            WorkflowVersion::new(1),
            NodeId::new("node-progress-retry"),
            None,
            events.clone(),
            Arc::new(SystemClock),
        );
        let progress = InferenceProgress {
            sequence: 1,
            completed: 1,
            total: Some(1),
            message: None,
        };

        sink.report(progress.clone());
        sink.report(progress);

        assert_eq!(events.calls.load(Ordering::SeqCst), 2);
        assert_eq!(events.events.lock().expect("events").len(), 1);
    }
}
