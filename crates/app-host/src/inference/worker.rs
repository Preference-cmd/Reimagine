use std::collections::BTreeMap;
use std::sync::Arc;

use reimagine_backend_worker_host::{
    ProcessInferenceBackend, StartedWorker, WorkerHostError, WorkerLaunchSpec, WorkerSupervisor,
};
use reimagine_backend_worker_protocol::WorkerInstanceProfile;
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceDescriptor, BackendInstanceObservation,
    BackendInstanceProfile, BackendInstanceRuntimeHooks, BackendInstanceSnapshot,
    BackendInstanceStatus, BackendProfile, BackendRunLifecycle, BackendRunLifecycleReport,
    BackendRunLifecycleRequest, DeviceProfile, InferenceBackend, InferenceCapability,
};
use reimagine_plugin::{Extension, Plugin};

use super::candidate::BuiltBackendInstance;

#[derive(Clone, Debug, Default)]
pub struct WorkerInventorySnapshot {
    candidates: Vec<WorkerBackendCandidate>,
}

impl WorkerInventorySnapshot {
    pub fn new(candidates: Vec<WorkerBackendCandidate>) -> Self {
        Self { candidates }
    }

    pub fn candidates(&self) -> &[WorkerBackendCandidate] {
        &self.candidates
    }
}

#[derive(Clone, Debug)]
pub struct WorkerBackendCandidate {
    launch: WorkerLaunchSpec,
    manifest_profile: WorkerInstanceProfile,
}

impl WorkerBackendCandidate {
    pub fn try_new(
        launch: WorkerLaunchSpec,
        manifest_profile: WorkerInstanceProfile,
    ) -> Result<Self, WorkerActivationError> {
        if launch.expected.backend_instance_id != manifest_profile.backend_instance_id {
            return Err(WorkerActivationError::InventoryProfileMismatch {
                expected: launch.expected.backend_instance_id.0.clone(),
                actual: manifest_profile.backend_instance_id.0,
            });
        }
        Ok(Self {
            launch,
            manifest_profile,
        })
    }

    pub fn launch(&self) -> &WorkerLaunchSpec {
        &self.launch
    }

    pub fn manifest_profile(&self) -> &WorkerInstanceProfile {
        &self.manifest_profile
    }

    pub fn backend_instance(&self) -> BackendInstance {
        BackendInstance::new(self.manifest_profile.backend_instance_id.0.clone())
    }

    fn backend(&self) -> Backend {
        Backend::new(self.launch.expected.backend_kind.clone())
    }

    fn projected_profile(&self) -> BackendInstanceProfile {
        project_worker_profile(self.backend(), &self.manifest_profile)
    }
}

pub trait WorkerInventoryProvider: Send + Sync + 'static {
    fn snapshot(&self) -> WorkerInventorySnapshot;
}

#[derive(Debug, Default)]
pub struct EmptyWorkerInventoryProvider;

impl WorkerInventoryProvider for EmptyWorkerInventoryProvider {
    fn snapshot(&self) -> WorkerInventorySnapshot {
        WorkerInventorySnapshot::default()
    }
}

#[derive(Clone, Debug)]
pub struct StaticWorkerInventoryProvider {
    snapshot: WorkerInventorySnapshot,
}

impl StaticWorkerInventoryProvider {
    pub fn new(snapshot: WorkerInventorySnapshot) -> Self {
        Self { snapshot }
    }
}

impl WorkerInventoryProvider for StaticWorkerInventoryProvider {
    fn snapshot(&self) -> WorkerInventorySnapshot {
        self.snapshot.clone()
    }
}

#[derive(Debug)]
pub enum WorkerActivationError {
    InventoryProfileMismatch { expected: String, actual: String },
    Startup(WorkerHostError),
    HandshakeProfileMismatch { instance: String },
}

impl std::fmt::Display for WorkerActivationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InventoryProfileMismatch { expected, actual } => write!(
                formatter,
                "worker inventory instance `{actual}` does not match launch instance `{expected}`"
            ),
            Self::Startup(error) => write!(formatter, "worker activation failed: {error}"),
            Self::HandshakeProfileMismatch { instance } => write!(
                formatter,
                "worker `{instance}` live hello does not match its inventory manifest profile"
            ),
        }
    }
}

impl std::error::Error for WorkerActivationError {}

pub(crate) struct WorkerControlService;

impl WorkerControlService {
    pub(crate) async fn activate(
        candidate: &WorkerBackendCandidate,
    ) -> Result<(BuiltBackendInstance, BackendInstanceProfile), WorkerActivationError> {
        let worker = Arc::new(
            WorkerSupervisor::new(candidate.launch.clone())
                .start()
                .await
                .map_err(WorkerActivationError::Startup)?,
        );
        let live = worker
            .hello
            .profile
            .instances
            .iter()
            .find(|profile| {
                profile.backend_instance_id == candidate.manifest_profile.backend_instance_id
            })
            .ok_or_else(|| WorkerActivationError::HandshakeProfileMismatch {
                instance: candidate.backend_instance().to_string(),
            })?;
        if live != &candidate.manifest_profile {
            let _ = worker.shutdown().await;
            return Err(WorkerActivationError::HandshakeProfileMismatch {
                instance: candidate.backend_instance().to_string(),
            });
        }

        let live_profile = project_worker_profile(candidate.backend(), live);
        let backend = Arc::new(ProcessInferenceBackend::new(Arc::clone(&worker)));
        let (plugin, extension) = burn_provenance();
        let descriptor = BackendInstanceDescriptor::new(
            live_profile.instance.clone(),
            live_profile.backend.clone(),
        )
        .with_device(live_profile.device.clone())
        .with_plugin(plugin.clone(), extension.clone());
        let hooks = Arc::new(WorkerRuntimeHooks {
            worker,
            instance: live_profile.instance.clone(),
            backend: live_profile.backend.clone(),
            device: live_profile.device.clone(),
            plugin,
            extension,
        });
        let backend: Arc<dyn InferenceBackend> = backend;
        let hooks: Arc<dyn BackendInstanceRuntimeHooks> = hooks;
        Ok((
            BuiltBackendInstance {
                descriptor,
                backend,
                runtime_hooks: hooks,
            },
            live_profile,
        ))
    }
}

pub(crate) fn worker_backend_profile(snapshot: &WorkerInventorySnapshot) -> BackendProfile {
    let backend = Backend::new("burn");
    let (plugin, extension) = burn_provenance();
    let mut profile = BackendProfile::new(backend).with_plugin(plugin, extension);
    for candidate in snapshot
        .candidates
        .iter()
        .filter(|candidate| candidate.backend().as_str() == "burn")
    {
        profile = profile.with_instance(candidate.projected_profile());
    }
    if profile.instances.is_empty() {
        profile = profile.with_diagnostic(no_worker_diagnostic());
    }
    profile
}

pub(crate) fn activation_diagnostic(
    instance: &BackendInstance,
    error: &WorkerActivationError,
) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!("app-host-worker-activation-{instance}")),
        DiagnosticCode::new("APP_HOST/LOCAL_WORKER_UNAVAILABLE"),
        DiagnosticSeverity::Error,
        DiagnosticSourceName::new("app-host"),
        format!("local worker `{instance}` is unavailable: {error}"),
        DiagnosticTarget::new(DiagnosticTargetDomain::new("app-host.compute_profile"))
            .with_path(format!("backend_instance/{instance}")),
    )
}

fn no_worker_diagnostic() -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new("app-host-local-worker-not-installed"),
        DiagnosticCode::new("APP_HOST/LOCAL_WORKER_NOT_INSTALLED"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("app-host"),
        "no local Burn worker is installed; install a compatible worker before selecting local Burn execution",
        DiagnosticTarget::new(DiagnosticTargetDomain::new("app-host.compute_profile"))
            .with_path("backend/burn"),
    )
}

fn project_worker_profile(
    backend: Backend,
    profile: &WorkerInstanceProfile,
) -> BackendInstanceProfile {
    let mut projected = BackendInstanceProfile::new(
        BackendInstance::new(profile.backend_instance_id.0.clone()),
        backend,
        DeviceProfile::new(profile.device_label.clone()),
        BackendInstanceStatus::Available,
    );
    for capability in &profile.capabilities {
        if let Some(capability) = InferenceCapability::from_label(capability) {
            projected = projected.with_capability(capability);
        }
    }
    projected
}

fn burn_provenance() -> (Plugin, Extension) {
    (
        Plugin::try_from("builtin.burn").expect("valid Burn worker plugin id"),
        Extension::try_from("backend.burn").expect("valid Burn worker extension id"),
    )
}

struct WorkerRuntimeHooks {
    worker: Arc<StartedWorker>,
    instance: BackendInstance,
    backend: Backend,
    device: DeviceProfile,
    plugin: Plugin,
    extension: Extension,
}

#[async_trait::async_trait]
impl BackendRunLifecycle for WorkerRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.instance
    }

    async fn begin_run(
        &self,
        _request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, reimagine_inference::InferenceError> {
        Ok(BackendRunLifecycleReport {
            backend_instance: self.instance.clone(),
            diagnostics: Vec::new(),
        })
    }

    async fn cleanup_run(
        &self,
        request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, reimagine_inference::InferenceError> {
        self.worker
            .cleanup(Some(request.run_id.to_string()), Vec::new())
            .await
            .map_err(
                |error| reimagine_inference::InferenceError::BackendExecutionFailed {
                    message: error.to_string(),
                },
            )?;
        Ok(BackendRunLifecycleReport {
            backend_instance: self.instance.clone(),
            diagnostics: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl BackendInstanceObservation for WorkerRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.instance
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        let mut observations = BTreeMap::new();
        observations.insert("process_backed".to_owned(), "true".to_owned());
        BackendInstanceSnapshot {
            backend_instance: self.instance.clone(),
            backend: self.backend.clone(),
            plugin: Some(self.plugin.clone()),
            extension: Some(self.extension.clone()),
            device: Some(self.device.clone()),
            observations,
            diagnostics: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_inventory_projects_actionable_no_worker_profile() {
        let profile = worker_backend_profile(&WorkerInventorySnapshot::default());
        assert_eq!(profile.backend.as_str(), "burn");
        assert!(profile.instances.is_empty());
        assert_eq!(
            profile.diagnostics[0].code().as_str(),
            "APP_HOST/LOCAL_WORKER_NOT_INSTALLED"
        );
        assert!(profile.diagnostics[0].message().contains("install"));
    }
}
