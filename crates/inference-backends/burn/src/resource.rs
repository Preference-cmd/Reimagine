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
        // V1 begin_run has no backend-local state to set up yet;
        // lifecycle hooks only need to acknowledge the run. Once
        // burn/08/10/11 land per-run resources, begin_run will
        // own the bookkeeping that currently lives only in
        // `BurnStore::insert_latent`.
        Ok(BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics: Vec::new(),
        })
    }

    async fn cleanup_run(
        &self,
        request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError> {
        // Drop every payload this run pinned into the shared
        // store. Without this call, latent tensors created by
        // `latent.create_empty` would survive past the workflow
        // that produced them. Candle's analogous hook calls
        // `store.cleanup_run` for the same reason; burn mirrors
        // it here.
        self.store.cleanup_run(&request.run_id);
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
        // Mirror candle's snapshot shape so monitoring consumers
        // see the same observation keys across backends. The
        // value is approximate because latent tensors report
        // element-count * sizeof(f32) without inspecting any
        // future f16/bf16 layout.
        observations.insert(
            "bytes_approximate".to_owned(),
            self.store.payload_byte_size().to_string(),
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
    use crate::config::BurnBackendConfig;
    use crate::profile::{EXTENSION_LABEL, PLUGIN_LABEL};
    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
    use reimagine_inference::{
        BackendInstanceObservation, BackendRunLifecycle, BackendRunLifecycleRequest,
        CreateEmptyLatentRequest, InferenceBackend,
    };

    fn backend() -> crate::backend::BurnBackend {
        crate::backend::BurnBackend::new(BurnBackendConfig::new("/models", "/output"))
            .expect("burn backend")
    }

    fn build_request() -> CreateEmptyLatentRequest {
        CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            RunId::new("run-hooks"),
            WorkflowId::new("wf-hooks"),
            WorkflowVersion::new(1),
            NodeId::new("node-hooks"),
        )
    }

    #[test]
    fn provenance_constants_are_valid() {
        assert_eq!(PLUGIN_LABEL, "builtin.burn");
        assert_eq!(EXTENSION_LABEL, "backend.burn");
    }

    #[tokio::test]
    async fn runtime_hooks_cleanup_run_evicts_latent_payloads() {
        // Regression for the burn/09 review: without this hook
        // dispatching to `BurnStore::cleanup_run`, every latent
        // created by `latent.create_empty` would leak past the
        // workflow that produced it.
        let backend = backend();
        let hooks = backend.runtime_hooks(None, None, None);

        backend
            .create_empty_latent(build_request())
            .await
            .expect("create_empty_latent");
        assert_eq!(backend.store().payload_count(), 1);

        let cleanup = hooks
            .cleanup_run(BackendRunLifecycleRequest {
                run_id: RunId::new("run-hooks"),
            })
            .await
            .expect("cleanup_run");
        // burn/13: under `flex` the default instance is
        // `burn:flex:cpu`; under wgpu it's `burn:cpu`.
        let expected_instance = if cfg!(all(not(feature = "wgpu"), feature = "flex")) {
            "burn:flex:cpu"
        } else {
            "burn:cpu"
        };
        assert_eq!(cleanup.backend_instance.as_str(), expected_instance);
        assert!(cleanup.diagnostics.is_empty());
        assert_eq!(backend.store().payload_count(), 0);
    }

    #[tokio::test]
    async fn runtime_hooks_cleanup_run_is_noop_for_unknown_run() {
        let backend = backend();
        let hooks = backend.runtime_hooks(None, None, None);
        let cleanup = hooks
            .cleanup_run(BackendRunLifecycleRequest {
                run_id: RunId::new("run-unknown"),
            })
            .await
            .expect("cleanup_run");
        assert!(cleanup.diagnostics.is_empty());
        // No latent payloads existed; store stays empty.
        assert_eq!(backend.store().payload_count(), 0);
    }

    #[tokio::test]
    async fn snapshot_reports_run_payloads_cached_models_and_bytes_approximate() {
        let backend = backend();
        let hooks = backend.runtime_hooks(None, None, None);

        backend
            .create_empty_latent(build_request())
            .await
            .expect("create_empty_latent");

        let snapshot = hooks.snapshot().await;
        assert_eq!(
            snapshot.observations.get("run_payloads"),
            Some(&"1".to_owned())
        );
        assert_eq!(
            snapshot.observations.get("cached_models"),
            Some(&"0".to_owned())
        );
        // 1 × 4 × 8 × 8 f32 = 1024 bytes
        assert_eq!(
            snapshot.observations.get("bytes_approximate"),
            Some(&"1024".to_owned())
        );
    }
}
