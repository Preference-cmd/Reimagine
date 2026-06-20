//! Default no-op [`RunResourceBackend`](reimagine_inference_core::RunResourceBackend)
//! used by the runtime in tests and when no concrete backend is wired.

use std::sync::Arc;

use reimagine_core::model::RunId;
use reimagine_inference::ExecutionValue;
use reimagine_inference_core::{MemorySnapshot, RunResourceBackend};

/// Default no-op backend used in tests and when no backend is wired.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopRunResourceBackend;

#[async_trait::async_trait]
impl RunResourceBackend for NoopRunResourceBackend {
    async fn begin_run(&self, _run_id: &RunId) {}
    async fn release_runtime_value(&self, _run_id: &RunId, _value: Arc<ExecutionValue>) {}
    async fn cleanup_run(&self, _run_id: &RunId) {}
    async fn memory_snapshot(&self) -> MemorySnapshot {
        MemorySnapshot::default()
    }
}
