use std::collections::BTreeMap;

use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceObservation, BackendInstanceSnapshot,
    BackendRunLifecycle, BackendRunLifecycleReport, BackendRunLifecycleRequest, DeviceKind,
    DeviceProfile, InferenceError,
};
use reimagine_plugin::{Extension, Plugin};

use crate::profile::BurnProfileProvider;

#[derive(Debug, Clone)]
pub struct BurnBackendInstanceRuntimeHooks {
    backend_instance: BackendInstance,
    backend: Backend,
    plugin: Option<Plugin>,
    extension: Option<Extension>,
    device: Option<DeviceProfile>,
}

impl BurnBackendInstanceRuntimeHooks {
    pub fn new(backend_instance: BackendInstance, device_label: Option<String>) -> Self {
        let (plugin, extension) = BurnProfileProvider::plugin_provenance();
        let device = device_label.map(|label| {
            let kind = if label == "cpu" {
                DeviceKind::Cpu
            } else {
                DeviceKind::Unknown
            };
            DeviceProfile::new(label).with_kind(kind)
        });

        Self {
            backend_instance,
            backend: BurnProfileProvider::backend_kind(),
            plugin: Some(plugin),
            extension: Some(extension),
            device,
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
        BackendInstanceSnapshot {
            backend_instance: self.backend_instance.clone(),
            backend: self.backend.clone(),
            plugin: self.plugin.clone(),
            extension: self.extension.clone(),
            device: self.device.clone(),
            observations: BTreeMap::new(),
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
