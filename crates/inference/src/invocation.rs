use std::sync::Arc;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId};
use reimagine_core::task::{CancelToken, ProgressSink};

#[derive(Clone)]
pub struct InferenceInvocation {
    run_id: RunId,
    node_id: NodeId,
    correlation_id: Option<CorrelationId>,
    cancellation: CancelToken,
    progress: Arc<dyn ProgressSink>,
}

impl InferenceInvocation {
    pub fn new(
        run_id: RunId,
        node_id: NodeId,
        correlation_id: Option<CorrelationId>,
        cancellation: CancelToken,
        progress: Arc<dyn ProgressSink>,
    ) -> Self {
        Self {
            run_id,
            node_id,
            correlation_id,
            cancellation,
            progress,
        }
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn cancellation(&self) -> &CancelToken {
        &self.cancellation
    }

    pub fn progress(&self) -> &Arc<dyn ProgressSink> {
        &self.progress
    }
}

impl std::fmt::Debug for InferenceInvocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InferenceInvocation")
            .field("run_id", &self.run_id)
            .field("node_id", &self.node_id)
            .field("correlation_id", &self.correlation_id)
            .finish_non_exhaustive()
    }
}
