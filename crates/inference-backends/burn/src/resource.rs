use std::collections::BTreeMap;
use std::sync::Arc;

use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceObservation, BackendInstanceSnapshot,
    BackendRunLifecycle, BackendRunLifecycleReport, BackendRunLifecycleRequest, DeviceProfile,
    InferenceError,
};
use reimagine_plugin::{Extension, Plugin};

use crate::store::{BurnModelCache, BurnStore};

#[derive(Debug, Clone)]
pub struct BurnBackendInstanceRuntimeHooks {
    backend_instance: BackendInstance,
    backend: Backend,
    plugin: Option<Plugin>,
    extension: Option<Extension>,
    device: Option<DeviceProfile>,
    store: Arc<BurnStore>,
    model_cache: Arc<BurnModelCache>,
}

impl BurnBackendInstanceRuntimeHooks {
    pub fn new(
        backend_instance: BackendInstance,
        backend: Backend,
        plugin: Option<Plugin>,
        extension: Option<Extension>,
        device: Option<DeviceProfile>,
        store: Arc<BurnStore>,
        model_cache: Arc<BurnModelCache>,
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
impl BackendRunLifecycle for BurnBackendInstanceRuntimeHooks {
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
        _request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError> {
        Ok(BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl BackendInstanceObservation for BurnBackendInstanceRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        let mut observations = BTreeMap::new();
        observations.insert(
            "run_payloads".to_owned(),
            self.store.payload_count().to_string(),
        );
        observations.insert(
            "cached_models".to_owned(),
            self.model_cache.bundle_count().to_string(),
        );

        BackendInstanceSnapshot {
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

#[cfg(test)]
mod tests {
    use crate::profile::{EXTENSION_LABEL, PLUGIN_LABEL};

    #[test]
    fn provenance_constants_are_valid() {
        assert_eq!(PLUGIN_LABEL, "builtin.burn");
        assert_eq!(EXTENSION_LABEL, "backend.burn");
    }
}
