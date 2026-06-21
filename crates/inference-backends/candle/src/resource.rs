use std::sync::Arc;

use reimagine_core::model::RunId;
use reimagine_inference::{MemorySnapshot, RunResourceBackend};

use crate::store::{CandleModelCache, CandleStore};

#[derive(Debug, Clone)]
pub struct CandleRunResourceBackend {
    store: Arc<CandleStore>,
    model_cache: Arc<CandleModelCache>,
}

impl CandleRunResourceBackend {
    pub fn new(store: Arc<CandleStore>, model_cache: Arc<CandleModelCache>) -> Self {
        Self { store, model_cache }
    }
}

#[async_trait::async_trait]
impl RunResourceBackend for CandleRunResourceBackend {
    async fn begin_run(&self, _run_id: &RunId) {}

    async fn cleanup_run(&self, run_id: &RunId) {
        self.store.cleanup_run(run_id);
    }

    async fn memory_snapshot(&self) -> MemorySnapshot {
        let mut observations = std::collections::HashMap::new();
        observations.insert(
            "run_payloads".to_string(),
            self.store.payload_count().to_string(),
        );
        observations.insert(
            "cached_models".to_string(),
            self.model_cache.bundle_count().to_string(),
        );
        observations.insert(
            "bytes_approximate".to_string(),
            self.store.payload_byte_size().to_string(),
        );
        MemorySnapshot { observations }
    }
}
