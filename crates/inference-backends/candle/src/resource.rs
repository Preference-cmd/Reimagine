use std::collections::BTreeMap;
use std::sync::Arc;

use reimagine_inference::{
    Backend, BackendInstance, BackendResourceObservation, BackendResourceSnapshot,
    BackendRunLifecycle, BackendRunLifecycleReport, BackendRunLifecycleRequest, DeviceProfile,
    InferenceError,
};
use reimagine_plugin::{Extension, Plugin};

use crate::store::{CandleModelCache, CandleStore};

#[derive(Debug, Clone)]
pub struct CandleResourceMechanism {
    backend_instance: BackendInstance,
    backend: Backend,
    plugin: Option<Plugin>,
    extension: Option<Extension>,
    device: Option<DeviceProfile>,
    store: Arc<CandleStore>,
    model_cache: Arc<CandleModelCache>,
}

impl CandleResourceMechanism {
    pub fn new(
        backend_instance: BackendInstance,
        backend: Backend,
        plugin: Option<Plugin>,
        extension: Option<Extension>,
        device: Option<DeviceProfile>,
        store: Arc<CandleStore>,
        model_cache: Arc<CandleModelCache>,
    ) -> Self {
        Self {
            backend_instance,
            backend,
            plugin,
            extension,
            device,
            store,
            model_cache,
        }
    }
}

#[async_trait::async_trait]
impl BackendRunLifecycle for CandleResourceMechanism {
    fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
    }

    async fn begin_run(
        &self,
        _request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError> {
        Ok(BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics: Vec::new(),
        })
    }

    async fn cleanup_run(
        &self,
        request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError> {
        self.store.cleanup_run(&request.run_id);
        Ok(BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl BackendResourceObservation for CandleResourceMechanism {
    fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
    }

    async fn resource_snapshot(&self) -> BackendResourceSnapshot {
        let mut observations: BTreeMap<String, String> = BTreeMap::new();
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
        BackendResourceSnapshot {
            backend_instance: self.backend_instance.clone(),
            backend: self.backend.clone(),
            plugin: self.plugin.clone(),
            extension: self.extension.clone(),
            device: self.device.clone(),
            observations,
            diagnostics: Vec::new(),
        }
    }
}
