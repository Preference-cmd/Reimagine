//! Default no-op [`BackendResourceMechanism`](reimagine_inference::BackendResourceMechanism)
//! used by the runtime in tests and when no concrete backend is wired.

use reimagine_inference::{
    Backend, BackendInstance, BackendResourceObservation, BackendResourceSnapshot,
    BackendRunLifecycle, BackendRunLifecycleReport, BackendRunLifecycleRequest, InferenceError,
};

/// Default no-op mechanism used in tests and when no backend is
/// wired.
#[derive(Debug, Clone)]
pub struct NoopResourceMechanism {
    backend_instance: BackendInstance,
    backend: Backend,
}

impl Default for NoopResourceMechanism {
    fn default() -> Self {
        Self {
            backend_instance: BackendInstance::new("noop"),
            backend: Backend::new("noop"),
        }
    }
}

#[async_trait::async_trait]
impl BackendRunLifecycle for NoopResourceMechanism {
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
impl BackendResourceObservation for NoopResourceMechanism {
    fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
    }

    async fn resource_snapshot(&self) -> BackendResourceSnapshot {
        BackendResourceSnapshot {
            backend_instance: self.backend_instance.clone(),
            backend: self.backend.clone(),
            plugin: None,
            extension: None,
            device: None,
            observations: Default::default(),
            diagnostics: Vec::new(),
        }
    }
}
