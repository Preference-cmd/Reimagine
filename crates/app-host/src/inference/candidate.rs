use std::sync::Arc;

use reimagine_backend_worker_host::{
    ExpectedWorkerIdentity, ProcessInferenceBackend, WorkerHostError, WorkerLaunchSpec,
    WorkerSupervisor,
};
use reimagine_backend_worker_protocol::{BackendInstanceId, ProtocolRange, WorkerInstallationId};
use reimagine_config::AppConfig;
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
#[cfg(feature = "candle")]
use reimagine_inference_candle::{
    CandleBackend, CandleBackendConfig, CandleDevice, CandleProfileProvider,
};
use reimagine_plugin::{Extension, Plugin};

#[derive(Debug)]
pub(crate) enum BackendCandidateError {
    #[cfg(feature = "candle")]
    Candle(reimagine_inference_candle::CandleBackendError),
    BurnWorker(WorkerHostError),
    ExecutorRegistration(reimagine_inference::NodeExecutorRegistryError),
    /// Topology/discovery bootstrap failure (T13).
    Topology(String),
}

impl std::fmt::Display for BackendCandidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "candle")]
            Self::Candle(error) => write!(f, "{error}"),
            Self::BurnWorker(error) => write!(f, "Burn worker error: {error}"),
            Self::ExecutorRegistration(error) => write!(f, "executor registration failed: {error}"),
            Self::Topology(message) => write!(f, "topology bootstrap failed: {message}"),
        }
    }
}

impl std::error::Error for BackendCandidateError {}

#[cfg(feature = "candle")]
impl From<reimagine_inference_candle::CandleBackendError> for BackendCandidateError {
    fn from(value: reimagine_inference_candle::CandleBackendError) -> Self {
        Self::Candle(value)
    }
}

impl From<WorkerHostError> for BackendCandidateError {
    fn from(value: WorkerHostError) -> Self {
        Self::BurnWorker(value)
    }
}

pub(crate) struct BuiltBackendInstance {
    pub(crate) descriptor: BackendInstanceDescriptor,
    pub(crate) backend: Arc<dyn InferenceBackend>,
    pub(crate) runtime_hooks: Arc<dyn BackendInstanceRuntimeHooks>,
}

pub(crate) trait BackendCandidate: Send + Sync {
    fn backend(&self) -> Backend;

    fn profile(&self) -> BackendProfile;

    fn build(
        &self,
        config: &AppConfig,
        instance: &BackendInstance,
        device: Option<DeviceProfile>,
    ) -> Result<BuiltBackendInstance, BackendCandidateError>;
}

#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CandleBackendCandidate;

#[cfg(feature = "candle")]
impl CandleBackendCandidate {
    pub(crate) const fn new() -> Self {
        Self
    }
}

#[cfg(feature = "candle")]
impl BackendCandidate for CandleBackendCandidate {
    fn backend(&self) -> Backend {
        Backend::new("candle")
    }

    fn profile(&self) -> BackendProfile {
        CandleProfileProvider::new().probe()
    }

    fn build(
        &self,
        config: &AppConfig,
        instance: &BackendInstance,
        device: Option<DeviceProfile>,
    ) -> Result<BuiltBackendInstance, BackendCandidateError> {
        let device_label = instance
            .as_str()
            .split_once(':')
            .map(|(_, label)| label)
            .unwrap_or("cpu");
        let candle_config = CandleBackendConfig::new(
            config.paths().models_dir().to_path_buf(),
            config.paths().output_dir().to_path_buf(),
        )
        .with_device(CandleDevice::new(device_label));
        let backend = Arc::new(CandleBackend::new(candle_config)?);
        let plugin = Plugin::try_from("builtin.candle").expect("valid built-in plugin id");
        let extension = Extension::try_from("backend.candle").expect("valid built-in extension id");
        let descriptor =
            BackendInstanceDescriptor::new(instance.clone(), backend.backend_kind().clone())
                .with_plugin(plugin.clone(), extension.clone());
        let descriptor = if let Some(device) = device.clone() {
            descriptor.with_device(device)
        } else {
            descriptor
        };
        let runtime_hooks = backend.runtime_hooks(Some(plugin), Some(extension), device);
        let backend: Arc<dyn InferenceBackend> = backend;
        Ok(BuiltBackendInstance {
            descriptor,
            backend,
            runtime_hooks: Arc::new(runtime_hooks),
        })
    }
}

/// Backend candidate for Burn inference workers.
///
/// Burn backends run as separate worker processes managed by `WorkerSupervisor`.
/// This candidate discovers installed Burn workers and provides profiles for them.
/// When `build()` is called, it starts the worker process and returns the built backend.
///
/// Burn workers are discovered from:
/// 1. The system PATH (searching for `burn-worker` binary)
/// 2. The worker inventory store (installed workers)
#[derive(Debug, Clone, Copy)]
pub(crate) struct BurnBackendCandidate;

impl BurnBackendCandidate {
    pub(crate) const fn new() -> Self {
        Self
    }

    /// Find burn-worker in PATH.
    fn find_worker_executable() -> Option<std::path::PathBuf> {
        let path_var = std::env::var_os("PATH")?;
        let paths = std::env::split_paths(&path_var);

        for dir in paths {
            let candidate = if cfg!(target_os = "windows") {
                dir.join("burn-worker.exe")
            } else {
                dir.join("burn-worker")
            };
            if candidate.exists() {
                return Some(candidate);
            }
        }

        None
    }
}

impl BackendCandidate for BurnBackendCandidate {
    fn backend(&self) -> Backend {
        Backend::new("burn")
    }

    fn profile(&self) -> BackendProfile {
        let (plugin, extension) = burn_provenance();
        let mut profile = BackendProfile::new(Backend::new("burn")).with_plugin(plugin, extension);

        // Check if burn-worker is available in PATH
        if Self::find_worker_executable().is_some() {
            // Add a default profile for the discovered worker
            let instance_id = BackendInstance::new("burn:wgpu:default");
            let instance_profile = BackendInstanceProfile::new(
                instance_id,
                Backend::new("burn"),
                DeviceProfile::new("wgpu:default"),
                BackendInstanceStatus::Available,
            )
            .with_capability(InferenceCapability::CreateEmptyLatent)
            .with_capability(InferenceCapability::DiffusionSample)
            .with_capability(InferenceCapability::LatentDecode)
            .with_capability(InferenceCapability::TextEncode);

            profile = profile.with_instance(instance_profile);
        } else {
            // No worker found, add a diagnostic
            profile = profile.with_diagnostic(no_worker_diagnostic());
        }

        profile
    }

    fn build(
        &self,
        _config: &AppConfig,
        instance: &BackendInstance,
        device: Option<DeviceProfile>,
    ) -> Result<BuiltBackendInstance, BackendCandidateError> {
        // Find the worker executable
        let worker_binary = Self::find_worker_executable().ok_or_else(|| {
            BackendCandidateError::BurnWorker(WorkerHostError::Spawn {
                path: std::path::PathBuf::from("burn-worker"),
                message: "Burn worker binary not found in PATH; install a Burn worker first"
                    .to_string(),
            })
        })?;

        // Extract device label from instance (e.g., "burn:wgpu:default" -> "wgpu:default")
        let device_label = instance
            .as_str()
            .strip_prefix("burn:")
            .unwrap_or("wgpu:default");

        // Create worker identity
        let installation_id = WorkerInstallationId(format!("burn-{}", device_label));
        let backend_instance_id = BackendInstanceId(instance.as_str().to_string());

        // Create launch spec
        let launch_spec = WorkerLaunchSpec {
            executable: worker_binary,
            expected: ExpectedWorkerIdentity {
                backend_instance_id: backend_instance_id.clone(),
                installation_id: installation_id.clone(),
                backend_kind: "burn".to_string(),
                target: std::env::consts::ARCH.to_string(),
                manifest_digest: "local-discovery".to_string(),
            },
            supported_protocols: ProtocolRange::new(1, 1),
            limits: Default::default(),
            environment: Vec::new(),
            transport: Default::default(),
        };

        // Start the worker process
        let supervisor = WorkerSupervisor::new(launch_spec);
        let started = tokio::runtime::Handle::current().block_on(supervisor.start())?;

        // Create the inference backend
        let backend = Arc::new(ProcessInferenceBackend::new(Arc::new(started)));

        // Build descriptor
        let plugin = Plugin::try_from("builtin.burn").expect("valid Burn plugin id");
        let extension = Extension::try_from("backend.burn").expect("valid Burn extension id");
        let mut descriptor = BackendInstanceDescriptor::new(instance.clone(), Backend::new("burn"))
            .with_plugin(plugin.clone(), extension.clone());

        if let Some(device) = device {
            descriptor = descriptor.with_device(device);
        }

        // Build runtime hooks
        let runtime_hooks = Arc::new(BurnRuntimeHooks {
            instance: instance.clone(),
        });

        Ok(BuiltBackendInstance {
            descriptor,
            backend: backend as Arc<dyn InferenceBackend>,
            runtime_hooks,
        })
    }
}

/// Runtime hooks for Burn worker backends.
#[derive(Debug)]
struct BurnRuntimeHooks {
    instance: BackendInstance,
}

#[async_trait::async_trait]
impl reimagine_inference::BackendRunLifecycle for BurnRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.instance
    }

    async fn begin_run(
        &self,
        _request: reimagine_inference::BackendRunLifecycleRequest,
    ) -> Result<reimagine_inference::BackendRunLifecycleReport, reimagine_inference::InferenceError>
    {
        Ok(reimagine_inference::BackendRunLifecycleReport {
            backend_instance: self.instance.clone(),
            diagnostics: Vec::new(),
        })
    }

    async fn cleanup_run(
        &self,
        _request: reimagine_inference::BackendRunLifecycleRequest,
    ) -> Result<reimagine_inference::BackendRunLifecycleReport, reimagine_inference::InferenceError>
    {
        Ok(reimagine_inference::BackendRunLifecycleReport {
            backend_instance: self.instance.clone(),
            diagnostics: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl reimagine_inference::BackendInstanceObservation for BurnRuntimeHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.instance
    }

    async fn snapshot(&self) -> reimagine_inference::BackendInstanceSnapshot {
        use std::collections::BTreeMap;
        reimagine_inference::BackendInstanceSnapshot {
            backend_instance: self.instance.clone(),
            backend: Backend::new("burn"),
            plugin: Some(Plugin::try_from("builtin.burn").expect("valid Burn plugin id")),
            extension: Some(Extension::try_from("backend.burn").expect("valid Burn extension id")),
            device: None,
            observations: BTreeMap::new(),
            diagnostics: Vec::new(),
        }
    }
}

fn burn_provenance() -> (Plugin, Extension) {
    (
        Plugin::try_from("builtin.burn").expect("valid Burn worker plugin id"),
        Extension::try_from("backend.burn").expect("valid Burn worker extension id"),
    )
}

fn no_worker_diagnostic() -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new("app-host-burn-worker-not-installed"),
        DiagnosticCode::new("APP_HOST/LOCAL_WORKER_NOT_INSTALLED"),
        DiagnosticSeverity::Warning,
        DiagnosticSourceName::new("app-host"),
        "no local Burn worker is installed; install a compatible worker before selecting local Burn execution",
        DiagnosticTarget::new(DiagnosticTargetDomain::new("app-host.compute_profile"))
            .with_path("backend/burn"),
    )
}

#[allow(clippy::vec_init_then_push)]
pub(crate) fn builtin_backend_candidates() -> Vec<Arc<dyn BackendCandidate>> {
    let mut candidates: Vec<Arc<dyn BackendCandidate>> = Vec::new();
    candidates.push(Arc::new(BurnBackendCandidate::new()));
    #[cfg(feature = "candle")]
    candidates.push(Arc::new(CandleBackendCandidate::new()));
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burn_candidate_provides_profile_with_diagnostic_when_no_worker() {
        let candidate = BurnBackendCandidate::new();
        let profile = candidate.profile();

        assert_eq!(profile.backend.as_str(), "burn");
        // When no worker is found, we should get a diagnostic
        if BurnBackendCandidate::find_worker_executable().is_none() {
            assert!(profile.instances.is_empty());
            assert_eq!(
                profile.diagnostics[0].code().as_str(),
                "APP_HOST/LOCAL_WORKER_NOT_INSTALLED"
            );
        }
    }

    #[test]
    fn burn_candidate_backend_is_burn() {
        let candidate = BurnBackendCandidate::new();
        assert_eq!(candidate.backend().as_str(), "burn");
    }

    #[test]
    fn builtin_candidates_include_burn() {
        let candidates = builtin_backend_candidates();
        assert!(candidates.iter().any(|c| c.backend().as_str() == "burn"));
    }

    #[test]
    fn builtin_candidates_list_burn_first() {
        let candidates = builtin_backend_candidates();
        assert!(!candidates.is_empty(), "candidates should not be empty");
        assert_eq!(
            candidates[0].backend().as_str(),
            "burn",
            "burn should be the first candidate"
        );
    }

    #[test]
    fn builtin_candidates_include_candle_when_feature_enabled() {
        let candidates = builtin_backend_candidates();
        #[cfg(feature = "candle")]
        assert!(candidates.iter().any(|c| c.backend().as_str() == "candle"));
    }

    #[test]
    fn burn_candidate_build_returns_error_when_no_worker() {
        let candidate = BurnBackendCandidate::new();
        let config = AppConfig::new(reimagine_config::AppPaths::new("/tmp/test"));
        let instance = BackendInstance::new("burn:wgpu:default");

        // This should fail because burn-worker is not in PATH
        let result = candidate.build(&config, &instance, None);
        assert!(result.is_err());
        let err_str = match result {
            Ok(_) => panic!("expected error"),
            Err(e) => format!("{e}"),
        };
        assert!(
            err_str.contains("Burn worker binary not found"),
            "expected worker not found error, got: {err_str}"
        );
    }
}
