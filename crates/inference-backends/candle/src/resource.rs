use std::sync::Arc;

use reimagine_core::model::RunId;
use reimagine_runtime::{MemorySnapshot, RunResourceBackend, RuntimeValue};

#[derive(Debug, Default, Clone)]
pub struct CandleRunResourceBackend;

#[async_trait::async_trait]
impl RunResourceBackend for CandleRunResourceBackend {
    async fn begin_run(&self, _run_id: &RunId) {}
    async fn release_runtime_value(&self, _run_id: &RunId, _value: Arc<RuntimeValue>) {}
    async fn cleanup_run(&self, _run_id: &RunId) {}
    async fn memory_snapshot(&self) -> MemorySnapshot {
        MemorySnapshot::default()
    }
}
