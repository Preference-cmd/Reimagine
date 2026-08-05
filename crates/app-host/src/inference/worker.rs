use std::path::{Path, PathBuf};
use std::sync::Arc;

use reimagine_backend_worker_host::{
    InstallationRecord, InventoryStore, ProcessInferenceBackend, WorkerHostError, WorkerLaunchSpec,
    WorkerLimits, WorkerStorePaths, WorkerSupervisor,
};
use reimagine_backend_worker_protocol::{BackendInstanceId, ProtocolRange, WorkerInstanceProfile};
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceDescriptor, BackendInstanceProfile,
    BackendInstanceRuntimeHooks, BackendInstanceStatus, BackendProfile, DeviceProfile,
    InferenceBackend, InferenceCapability,
};
use reimagine_plugin::{Extension, Plugin};

use super::candidate::BuiltBackendInstance;
use super::switch::{
    ProcessSwitchableWorker, RunCancellation, SwitchableWorker, SwitchingWorkerRuntimeHooks,
    WorkerSwitchError, WorkerSwitchService,
};

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

/// Inventory provider that reads installed worker packages from the
/// durable worker store.
///
/// Records that cannot be mapped (unreadable store, missing executable,
/// non-Burn records, profile mismatch) are skipped with a log. A store
/// with no records yields an empty snapshot, so a host with nothing
/// installed stays in worker-less mode.
#[derive(Clone, Debug)]
pub struct InstalledWorkerInventoryProvider {
    store_paths: WorkerStorePaths,
}

impl InstalledWorkerInventoryProvider {
    pub fn new(store_paths: WorkerStorePaths) -> Self {
        Self { store_paths }
    }

    /// Derive the worker store for a workspace base path, mirroring the
    /// desktop host's `{workspace}.app-data` application-data convention.
    pub fn for_base_path(base_path: impl AsRef<Path>) -> Self {
        Self::new(WorkerStorePaths::new(
            base_path.as_ref().with_extension("app-data"),
        ))
    }
}

impl WorkerInventoryProvider for InstalledWorkerInventoryProvider {
    fn snapshot(&self) -> WorkerInventorySnapshot {
        let store = InventoryStore::new(self.store_paths.clone());
        match store.list() {
            Ok(snapshot) => WorkerInventorySnapshot::new(
                snapshot
                    .records
                    .iter()
                    .filter_map(candidate_from_record)
                    .collect(),
            ),
            Err(error) => {
                tracing::warn!("worker inventory store unreadable: {error}");
                WorkerInventorySnapshot::default()
            }
        }
    }
}

/// Map a durable installation record to a runnable worker candidate.
///
/// The installed package layout is not recorded, so the executable is
/// resolved by scanning the install directory for the worker binary
/// (preferring `burn-worker` entries, matching package extraction).
fn candidate_from_record(record: &InstallationRecord) -> Option<WorkerBackendCandidate> {
    if record.identity.backend_kind != "burn" {
        tracing::debug!(
            "skipping installed worker `{}`: unsupported backend kind `{}`",
            record.installation_id.0,
            record.identity.backend_kind
        );
        return None;
    }
    let Some(executable) = worker_executable(Path::new(&record.install_path)) else {
        tracing::warn!(
            "skipping installed worker `{}`: no executable found in `{}`",
            record.installation_id.0,
            record.install_path
        );
        return None;
    };
    let manifest_profile = record.manifest_profile.clone().unwrap_or_else(|| {
        default_burn_manifest_profile(record.identity.backend_instance_id.0.as_str())
    });
    match WorkerBackendCandidate::try_new(
        WorkerLaunchSpec {
            executable,
            expected: record.identity.clone(),
            supported_protocols: ProtocolRange::new(1, 1),
            limits: WorkerLimits::default(),
            environment: Vec::new(),
            transport: Default::default(),
        },
        manifest_profile,
    ) {
        Ok(candidate) => Some(candidate),
        Err(error) => {
            tracing::warn!(
                "skipping installed worker `{}`: {error}",
                record.installation_id.0
            );
            None
        }
    }
}

fn worker_executable(install_dir: &Path) -> Option<PathBuf> {
    let files = std::fs::read_dir(install_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    let preferred = files.iter().find(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("burn-worker"))
    });
    preferred.cloned().or_else(|| files.into_iter().next())
}

fn default_burn_manifest_profile(instance_id: &str) -> WorkerInstanceProfile {
    let device_label = instance_id
        .strip_prefix("burn:")
        .filter(|label| !label.is_empty())
        .unwrap_or("default")
        .to_owned();
    WorkerInstanceProfile {
        backend_instance_id: BackendInstanceId(instance_id.to_owned()),
        device_label,
        capabilities: vec![
            "latent.create_empty".to_owned(),
            "diffusion.sample".to_owned(),
            "latent.decode".to_owned(),
            "text.encode".to_owned(),
        ],
        operation_options: serde_json::json!({}),
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
    ) -> Result<
        (
            BuiltBackendInstance,
            BackendInstanceProfile,
            Arc<WorkerSwitchService>,
        ),
        WorkerActivationError,
    > {
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
        let active = Arc::new(ProcessSwitchableWorker::new(
            live_profile.instance.clone(),
            Arc::clone(&worker),
            Arc::clone(&backend),
        ));
        let active_worker: Arc<dyn SwitchableWorker> = active;
        let workers = Arc::new(WorkerSwitchService::new(
            active_worker,
            Arc::new(UnavailableRunCancellation),
        ));
        let (plugin, extension) = burn_provenance();
        let descriptor = BackendInstanceDescriptor::new(
            live_profile.instance.clone(),
            live_profile.backend.clone(),
        )
        .with_device(live_profile.device.clone())
        .with_plugin(plugin.clone(), extension.clone());
        let hooks = Arc::new(SwitchingWorkerRuntimeHooks::new(Arc::clone(&workers)));
        let backend: Arc<dyn InferenceBackend> = backend;
        let hooks: Arc<dyn BackendInstanceRuntimeHooks> = hooks;
        Ok((
            BuiltBackendInstance {
                descriptor,
                backend,
                runtime_hooks: hooks,
            },
            live_profile,
            workers,
        ))
    }
}

struct UnavailableRunCancellation;

impl RunCancellation for UnavailableRunCancellation {
    fn cancel(&self, run_id: &reimagine_core::model::RunId) -> Result<(), WorkerSwitchError> {
        Err(WorkerSwitchError::Cancellation {
            run_id: run_id.clone(),
            message: "runtime cancellation is not bound".to_owned(),
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_backend_worker_host::ExpectedWorkerIdentity;
    use reimagine_backend_worker_protocol::WorkerInstallationId;

    fn test_record(id: &str, install_dir: &Path) -> InstallationRecord {
        InstallationRecord {
            installation_id: WorkerInstallationId(id.to_owned()),
            version: "1.0.0".to_owned(),
            identity: ExpectedWorkerIdentity {
                backend_instance_id: BackendInstanceId("burn:wgpu:default".to_owned()),
                installation_id: WorkerInstallationId(id.to_owned()),
                backend_kind: "burn".to_owned(),
                target: "aarch64-apple-darwin".to_owned(),
                manifest_digest: "abc123".to_owned(),
            },
            installed_at: chrono::Utc::now(),
            install_path: install_dir.to_string_lossy().into_owned(),
            manifest_profile: None,
        }
    }

    fn installed_dir(temp: &tempfile::TempDir, id: &str) -> std::path::PathBuf {
        let dir = temp.path().join("installed").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("reimagine-inference-burn-worker"), b"binary").unwrap();
        dir
    }

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

    #[test]
    fn installed_provider_maps_records_to_candidates() {
        let temp = tempfile::tempdir().unwrap();
        let install_dir = installed_dir(&temp, "burn-wgpu-v1");
        let store = InventoryStore::new(WorkerStorePaths::new(temp.path()));
        store
            .add(&test_record("burn-wgpu-v1", &install_dir))
            .unwrap();

        let provider = InstalledWorkerInventoryProvider::new(store.store_paths().clone());
        let snapshot = provider.snapshot();
        assert_eq!(snapshot.candidates().len(), 1);
        let candidate = &snapshot.candidates()[0];
        assert_eq!(
            candidate.launch().expected.installation_id.0,
            "burn-wgpu-v1"
        );
        assert_eq!(
            candidate.launch().executable,
            install_dir.join("reimagine-inference-burn-worker")
        );
        assert_eq!(candidate.manifest_profile().device_label, "wgpu:default");
        assert_eq!(candidate.backend_instance().as_str(), "burn:wgpu:default");
    }

    #[test]
    fn installed_provider_uses_recorded_manifest_profile_when_present() {
        let temp = tempfile::tempdir().unwrap();
        let install_dir = installed_dir(&temp, "burn-wgpu-v1");
        let mut record = test_record("burn-wgpu-v1", &install_dir);
        record.manifest_profile = Some(WorkerInstanceProfile {
            backend_instance_id: BackendInstanceId("burn:wgpu:default".to_owned()),
            device_label: "wgpu:cuda".to_owned(),
            capabilities: vec!["latent.create_empty".to_owned()],
            operation_options: serde_json::json!({}),
        });
        let store = InventoryStore::new(WorkerStorePaths::new(temp.path()));
        store.add(&record).unwrap();

        let provider = InstalledWorkerInventoryProvider::new(store.store_paths().clone());
        let snapshot = provider.snapshot();
        assert_eq!(snapshot.candidates().len(), 1);
        assert_eq!(
            snapshot.candidates()[0].manifest_profile().device_label,
            "wgpu:cuda"
        );
    }

    #[test]
    fn installed_provider_returns_empty_snapshot_for_missing_store() {
        let temp = tempfile::tempdir().unwrap();
        let provider = InstalledWorkerInventoryProvider::new(WorkerStorePaths::new(temp.path()));
        assert!(provider.snapshot().candidates().is_empty());
    }

    #[test]
    fn installed_provider_returns_empty_snapshot_for_empty_store() {
        let temp = tempfile::tempdir().unwrap();
        let store = InventoryStore::new(WorkerStorePaths::new(temp.path()));
        let provider = InstalledWorkerInventoryProvider::new(store.store_paths().clone());
        assert!(provider.snapshot().candidates().is_empty());
    }

    #[test]
    fn installed_provider_skips_record_without_executable() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("never-installed");
        let store = InventoryStore::new(WorkerStorePaths::new(temp.path()));
        store.add(&test_record("burn-stale", &missing)).unwrap();

        let provider = InstalledWorkerInventoryProvider::new(store.store_paths().clone());
        assert!(provider.snapshot().candidates().is_empty());
    }

    #[test]
    fn installed_provider_skips_non_burn_records() {
        let temp = tempfile::tempdir().unwrap();
        let install_dir = installed_dir(&temp, "candle-cpu-v1");
        let mut record = test_record("candle-cpu-v1", &install_dir);
        record.identity.backend_kind = "candle".to_owned();
        let store = InventoryStore::new(WorkerStorePaths::new(temp.path()));
        store.add(&record).unwrap();

        let provider = InstalledWorkerInventoryProvider::new(store.store_paths().clone());
        assert!(provider.snapshot().candidates().is_empty());
    }

    #[test]
    fn for_base_path_derives_app_data_store() {
        let provider = InstalledWorkerInventoryProvider::for_base_path("/tmp/ws");
        assert_eq!(
            provider.store_paths.app_data_root(),
            Path::new("/tmp/ws.app-data")
        );
        assert_eq!(
            provider.store_paths.inventory_dir(),
            Path::new("/tmp/ws.app-data/workers/inventory")
        );
    }
}
