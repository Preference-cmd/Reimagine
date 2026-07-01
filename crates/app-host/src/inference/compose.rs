use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use reimagine_config::{AppConfig, InferenceBackendConfig};
use reimagine_inference::registry::register_builtin_inference_executors;
use reimagine_inference::{
    BackendInstance, BackendInstanceRuntimeHooks, BackendInstanceStatus, BackendOverrides,
    CompositeBackendInstanceRuntimeHooks, DefaultInferenceRuntime, InferenceBackendRegistry,
    RejectAllBridgePolicy, StaticBackendSelectionPolicy, WorkspaceComputeProfile,
};
use reimagine_runtime::NodeExecutorRegistry;

use crate::ModelService;
use crate::inference::candidate::{
    BackendCandidate, BackendCandidateError, BuiltBackendInstance, builtin_backend_candidates,
};
use crate::inference::image_source_resolver::InputImageSourceResolver;
use crate::inference::resolver::ModelResolverAdapter;
use crate::inference::selection::{BackendProfilesByInstance, resolve_backend_selection};

pub(crate) struct ComposedBackends {
    registry: InferenceBackendRegistry,
    runtime_hooks: Arc<dyn BackendInstanceRuntimeHooks>,
    selected_instance: BackendInstance,
    priority_order: Vec<BackendInstance>,
    allowed_instances: Vec<BackendInstance>,
    disabled_instances: Vec<BackendInstance>,
}

pub(crate) struct ComposedInferenceRuntime {
    pub(crate) executor_registry: NodeExecutorRegistry,
    #[cfg(test)]
    pub(crate) inference_runtime: Arc<DefaultInferenceRuntime>,
    pub(crate) runtime_hooks: Arc<dyn BackendInstanceRuntimeHooks>,
    pub(crate) selected_instance: BackendInstance,
}

pub(crate) struct BootstrapInference {
    pub(crate) runtime: ComposedInferenceRuntime,
    pub(crate) compute_profile: WorkspaceComputeProfile,
}

pub(crate) fn bootstrap_inference(
    config: &AppConfig,
    backend_config: &InferenceBackendConfig,
    model_service: Arc<ModelService>,
) -> Result<BootstrapInference, BackendCandidateError> {
    bootstrap_inference_with_candidates(
        config,
        backend_config,
        model_service,
        builtin_backend_candidates(),
    )
}

fn bootstrap_inference_with_candidates(
    config: &AppConfig,
    backend_config: &InferenceBackendConfig,
    model_service: Arc<ModelService>,
    candidates: Vec<Arc<dyn BackendCandidate>>,
) -> Result<BootstrapInference, BackendCandidateError> {
    let mut workspace_profile = collect_compute_profile(&candidates);
    let resolved = resolve_backend_selection(backend_config, &workspace_profile);
    for diagnostic in resolved.diagnostics {
        workspace_profile = workspace_profile.with_diagnostic(diagnostic);
    }

    let runtime = compose_inference_runtime_with_candidates(
        config,
        model_service,
        &candidates,
        &workspace_profile,
        resolved.selected_instance,
        resolved.priority_order,
        resolved.disabled_instances,
    )?;

    Ok(BootstrapInference {
        runtime,
        compute_profile: workspace_profile,
    })
}

/// Construct the inference runtime for a workspace from built-in candidates.
///
/// Production V1 registers built-in backend candidates through a backend-
/// keyed path: candidates provide profiles, selected instances, backend
/// builders, and runtime hooks. Tests inject an additional stub candidate through
/// `bootstrap_inference_with_candidates` to prove this path is no longer
/// Candle-shaped.
#[cfg(test)]
pub(crate) fn compose_inference_runtime(
    config: &AppConfig,
    selected_instance: BackendInstance,
    model_service: Arc<ModelService>,
) -> Result<ComposedInferenceRuntime, BackendCandidateError> {
    let candidates = builtin_backend_candidates();
    let profile = collect_compute_profile(&candidates);
    compose_inference_runtime_with_candidates(
        config,
        model_service,
        &candidates,
        &profile,
        selected_instance.clone(),
        vec![selected_instance],
        Vec::new(),
    )
}

fn collect_compute_profile(candidates: &[Arc<dyn BackendCandidate>]) -> WorkspaceComputeProfile {
    let mut profile = WorkspaceComputeProfile::new();
    for candidate in candidates {
        profile = profile.with_backend_profile(candidate.profile());
    }
    profile
}

fn compose_inference_runtime_with_candidates(
    config: &AppConfig,
    model_service: Arc<ModelService>,
    candidates: &[Arc<dyn BackendCandidate>],
    profile: &WorkspaceComputeProfile,
    selected_instance: BackendInstance,
    priority_order: Vec<BackendInstance>,
    disabled_instances: Vec<BackendInstance>,
) -> Result<ComposedInferenceRuntime, BackendCandidateError> {
    let composed_backends = compose_inference_backends(
        config,
        candidates,
        profile,
        selected_instance,
        priority_order,
        disabled_instances,
    )?;
    let inference_runtime = compose_runtime_router(
        composed_backends.registry,
        composed_backends.priority_order,
        composed_backends.allowed_instances,
        composed_backends.disabled_instances,
    );

    let mut executor_registry = NodeExecutorRegistry::default();
    let image_source_resolver = Arc::new(InputImageSourceResolver::new(config.paths()));
    let executor_inference_runtime: Arc<dyn reimagine_inference::InferenceRuntime> =
        inference_runtime.clone();
    register_builtin_inference_executors(
        &mut executor_registry,
        executor_inference_runtime,
        Arc::new(ModelResolverAdapter::new(
            model_service,
            config.paths().clone(),
        )),
        image_source_resolver,
    )
    .expect("register executors");

    Ok(ComposedInferenceRuntime {
        executor_registry,
        #[cfg(test)]
        inference_runtime,
        runtime_hooks: composed_backends.runtime_hooks,
        selected_instance: composed_backends.selected_instance,
    })
}

fn compose_inference_backends(
    config: &AppConfig,
    candidates: &[Arc<dyn BackendCandidate>],
    profile: &WorkspaceComputeProfile,
    selected_instance: BackendInstance,
    priority_order: Vec<BackendInstance>,
    disabled_instances: Vec<BackendInstance>,
) -> Result<ComposedBackends, BackendCandidateError> {
    let profiles = BackendProfilesByInstance::new(profile);
    let candidate_map = candidates
        .iter()
        .map(|candidate| (candidate.backend(), Arc::clone(candidate)))
        .collect::<HashMap<_, _>>();

    let disabled = disabled_instances.iter().cloned().collect::<HashSet<_>>();
    let mut registry = InferenceBackendRegistry::new();
    let mut hooks = Vec::new();
    let mut allowed_instances = Vec::new();

    for instance in priority_order.iter().cloned() {
        if disabled.contains(&instance) {
            continue;
        }
        let Some(instance_profile) = profiles.get(&instance) else {
            continue;
        };
        if instance_profile.status != BackendInstanceStatus::Available {
            continue;
        }
        let Some(candidate) = candidate_map.get(&instance_profile.backend) else {
            continue;
        };
        let built = candidate.build(config, &instance, Some(instance_profile.device.clone()))?;
        register_built_backend(&mut registry, &mut hooks, built);
        allowed_instances.push(instance);
    }

    Ok(ComposedBackends {
        registry,
        runtime_hooks: Arc::new(CompositeBackendInstanceRuntimeHooks::new(hooks)),
        selected_instance,
        priority_order,
        allowed_instances,
        disabled_instances,
    })
}

fn register_built_backend(
    registry: &mut InferenceBackendRegistry,
    hooks: &mut Vec<Arc<dyn BackendInstanceRuntimeHooks>>,
    built: BuiltBackendInstance,
) {
    registry.register(built.descriptor, built.backend);
    hooks.push(built.runtime_hooks);
}

fn compose_runtime_router(
    registry: InferenceBackendRegistry,
    priority_order: Vec<BackendInstance>,
    allowed_instances: Vec<BackendInstance>,
    disabled_instances: Vec<BackendInstance>,
) -> Arc<DefaultInferenceRuntime> {
    let policy = StaticBackendSelectionPolicy::with_overrides(
        BackendOverrides::new(),
        priority_order,
        Some(allowed_instances),
        disabled_instances,
    );
    Arc::new(DefaultInferenceRuntime::with_policy(
        Arc::new(registry),
        Arc::new(policy),
        Arc::new(RejectAllBridgePolicy),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_config::AppPaths;
    use reimagine_inference::{
        Backend, BackendInstanceDescriptor, BackendInstanceObservation, BackendInstanceProfile,
        BackendInstanceSnapshot, BackendProfile, BackendRunLifecycle, BackendRunLifecycleReport,
        BackendRunLifecycleRequest, CannedCapabilityResponse, CreateEmptyLatentRequest,
        CreateEmptyLatentResponse, DeviceKind, DeviceProfile, FakeBackend, InferenceCapability,
        InferenceError, InferenceRuntime, LatentContent, LatentSpaceMetadata, RuntimeLatent,
    };
    use reimagine_plugin::{Extension, Plugin};
    use std::collections::BTreeMap;

    use crate::inference::candidate::CandleBackendCandidate;

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-app-host-compose-{prefix}-{nonce}"))
    }

    #[test]
    fn compose_backends_registers_cpu_instance_for_resolved_cpu_instance() {
        let base = temp_dir("resolved-cpu");
        let config = AppConfig::new(AppPaths::new(&base));
        let profile = collect_compute_profile(&builtin_backend_candidates());

        let composed = compose_inference_backends(
            &config,
            &builtin_backend_candidates(),
            &profile,
            BackendInstance::new("candle:cpu"),
            vec![BackendInstance::new("candle:cpu")],
            Vec::new(),
        )
        .expect("backends");

        assert_eq!(
            composed.registry.len(),
            1,
            "selected backend should register once"
        );
        assert_eq!(
            composed.selected_instance,
            BackendInstance::new("candle:cpu")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn compose_backends_attaches_builtin_candle_plugin_provenance() {
        let base = temp_dir("plugin-provenance");
        let config = AppConfig::new(AppPaths::new(&base));
        let candidates = builtin_backend_candidates();
        let profile = collect_compute_profile(&candidates);

        let composed = compose_inference_backends(
            &config,
            &candidates,
            &profile,
            BackendInstance::new("candle:cpu"),
            vec![BackendInstance::new("candle:cpu")],
            Vec::new(),
        )
        .expect("backends");
        let descriptors = composed.registry.descriptors();
        let descriptor = descriptors.first().expect("registered descriptor");

        assert_eq!(descriptor.instance, composed.selected_instance);
        assert_eq!(
            descriptor.plugin.as_ref().map(|p| p.as_str()),
            Some("builtin.candle")
        );
        assert_eq!(
            descriptor.extension.as_ref().map(|e| e.as_str()),
            Some("backend.candle")
        );
        assert_eq!(
            descriptor.device.as_ref().map(|d| d.label.as_str()),
            Some("cpu")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn builtin_compute_profile_includes_burn_load_bundle_capability() {
        let profile = collect_compute_profile(&builtin_backend_candidates());

        let burn = profile
            .backend_profiles
            .iter()
            .find(|profile| profile.backend == Backend::new("burn"))
            .expect("burn backend profile");
        assert_eq!(
            burn.plugin.as_ref().map(|plugin| plugin.as_str()),
            Some("builtin.burn")
        );
        assert_eq!(
            burn.extension.as_ref().map(|extension| extension.as_str()),
            Some("backend.burn")
        );
        let cpu = burn
            .instances
            .iter()
            .find(|instance| instance.instance == BackendInstance::new("burn:cpu"))
            .expect("burn:cpu instance profile");
        assert_eq!(cpu.status, BackendInstanceStatus::Available);
        assert_eq!(cpu.device.kind, DeviceKind::Cpu);
        assert_eq!(
            cpu.capabilities,
            vec![reimagine_inference::InferenceCapability::LoadBundle]
        );
        assert!(cpu.operation_options.is_empty());
        assert!(cpu.diagnostics.is_empty());
    }

    #[tokio::test]
    async fn bootstrap_with_burn_selected_registers_only_burn_candidate_and_hooks() {
        let base = temp_dir("burn-bootstrap");
        let config = AppConfig::new(AppPaths::new(&base));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));
        let backend_config = InferenceBackendConfig {
            selected_instance: Some("burn:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };

        let bootstrapped = bootstrap_inference_with_candidates(
            &config,
            &backend_config,
            model_service,
            builtin_backend_candidates(),
        )
        .expect("bootstrap");

        assert_eq!(
            bootstrapped.runtime.selected_instance,
            BackendInstance::new("burn:cpu")
        );
        let snapshots = reimagine_inference::BackendInstanceObservation::snapshots(
            bootstrapped.runtime.runtime_hooks.as_ref(),
        )
        .await;
        assert_eq!(
            snapshots.len(),
            1,
            "selected Burn should not add Candle fallback hooks"
        );
        assert_eq!(
            snapshots[0].backend_instance,
            BackendInstance::new("burn:cpu")
        );
        assert_eq!(
            snapshots[0].plugin.as_ref().map(|plugin| plugin.as_str()),
            Some("builtin.burn")
        );
        assert_eq!(
            snapshots[0]
                .extension
                .as_ref()
                .map(|extension| extension.as_str()),
            Some("backend.burn")
        );
        assert_eq!(
            snapshots[0]
                .device
                .as_ref()
                .map(|device| device.label.as_str()),
            Some("cpu")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn burn_selected_create_empty_latent_fails_at_router_capability_check() {
        let base = temp_dir("burn-router");
        let config = AppConfig::new(AppPaths::new(&base));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));
        let backend_config = InferenceBackendConfig {
            selected_instance: Some("burn:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };

        let bootstrapped = bootstrap_inference_with_candidates(
            &config,
            &backend_config,
            model_service,
            builtin_backend_candidates(),
        )
        .expect("bootstrap");

        let request = CreateEmptyLatentRequest::new(
            512,
            512,
            1,
            reimagine_core::model::RunId::new("run-burn-router"),
            reimagine_core::model::WorkflowId::new("workflow-burn-router"),
            reimagine_core::model::WorkflowVersion::new(1),
            reimagine_core::model::NodeId::new("latent-burn-router"),
        );
        let err = bootstrapped
            .runtime
            .inference_runtime
            .create_empty_latent(request)
            .await
            .expect_err("burn skeleton advertises no latent capability");

        match err {
            InferenceError::CandidateBackendLacksCapability {
                instance,
                backend,
                capability,
            } => {
                assert_eq!(instance, BackendInstance::new("burn:cpu"));
                assert_eq!(backend, Backend::new("burn"));
                assert_eq!(capability, InferenceCapability::CreateEmptyLatent);
            }
            other => panic!("expected CandidateBackendLacksCapability, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn compose_uses_metal_instance_when_available_or_errors_when_unavailable() {
        let base = temp_dir("resolved-metal");
        let config = AppConfig::new(AppPaths::new(&base));
        let candidates = builtin_backend_candidates();
        let profile = collect_compute_profile(&candidates);

        let result = compose_inference_backends(
            &config,
            &candidates,
            &profile,
            BackendInstance::new("candle:metal"),
            vec![BackendInstance::new("candle:metal")],
            Vec::new(),
        );
        match result {
            Ok(composed) => {
                assert_eq!(
                    composed.selected_instance,
                    BackendInstance::new("candle:metal")
                );
            }
            Err(BackendCandidateError::Candle(
                reimagine_inference_candle::CandleBackendError::DeviceUnavailable {
                    requested, ..
                },
            )) => {
                assert_eq!(
                    requested, "metal",
                    "non-Metal hosts may reject direct resolved-metal composition"
                );
            }
            Err(other) => panic!("expected metal composition or DeviceUnavailable, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn compose_runtime_router_uses_config_projected_backend_policy() {
        let instance = BackendInstance::new("candle:cpu");
        let registry = InferenceBackendRegistry::new();
        let runtime = compose_runtime_router(
            registry,
            vec![instance.clone()],
            vec![instance.clone()],
            Vec::new(),
        );
        let request = reimagine_inference::BackendSelectionRequest {
            capability: reimagine_inference::InferenceCapability::LoadBundle,
            node_id: None,
            affinities: Vec::new(),
            registered: Vec::new(),
            explicit_override: None,
        };

        assert_eq!(
            runtime.selection_policy().candidates(&request),
            vec![instance.clone()]
        );
        assert!(
            runtime
                .selection_policy()
                .allows_explicit_override(&instance, &request)
        );
        assert!(
            !runtime
                .selection_policy()
                .allows_explicit_override(&BackendInstance::new("candle:metal"), &request)
        );
    }

    #[test]
    fn compose_runtime_accepts_selected_instance_without_backend_config() {
        let base = temp_dir("resolved-instance-smoke");
        let config = AppConfig::new(AppPaths::new(&base));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));

        let runtime =
            compose_inference_runtime(&config, BackendInstance::new("candle:cpu"), model_service)
                .expect("runtime");
        assert_eq!(
            runtime.selected_instance,
            BackendInstance::new("candle:cpu")
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn bootstrap_with_stub_collects_multiple_profiles_and_selects_stub() {
        let base = temp_dir("stub-bootstrap");
        let config = AppConfig::new(AppPaths::new(&base));
        let model_service = Arc::new(ModelService::new(config.paths().clone()));
        let backend_config = InferenceBackendConfig {
            selected_instance: Some("stub:cpu".to_string()),
            priority_order: vec!["stub:cpu".to_string(), "candle:cpu".to_string()],
            ..InferenceBackendConfig::default()
        };

        let bootstrapped = bootstrap_inference_with_candidates(
            &config,
            &backend_config,
            model_service,
            vec![
                Arc::new(CandleBackendCandidate::new()),
                Arc::new(StubBackendCandidate::new()),
            ],
        )
        .expect("bootstrap");

        assert_eq!(
            bootstrapped.runtime.selected_instance,
            BackendInstance::new("stub:cpu")
        );
        assert!(
            bootstrapped
                .compute_profile
                .backend_profiles
                .iter()
                .any(|profile| profile.backend == Backend::new("candle"))
        );
        assert!(
            bootstrapped
                .compute_profile
                .backend_profiles
                .iter()
                .any(|profile| profile.backend == Backend::new("stub"))
        );
        let snapshots = reimagine_inference::BackendInstanceObservation::snapshots(
            bootstrapped.runtime.runtime_hooks.as_ref(),
        )
        .await;
        assert!(
            snapshots
                .iter()
                .any(|snapshot| snapshot.backend_instance == BackendInstance::new("stub:cpu")),
            "composite hooks should include the selected stub backend instance"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[derive(Debug, Clone, Copy)]
    struct StubBackendCandidate;

    impl StubBackendCandidate {
        fn new() -> Self {
            Self
        }
    }

    impl BackendCandidate for StubBackendCandidate {
        fn backend(&self) -> Backend {
            Backend::new("stub")
        }

        fn profile(&self) -> BackendProfile {
            BackendProfile::new(Backend::new("stub"))
                .with_plugin(
                    Plugin::try_from("builtin.stub").expect("plugin"),
                    Extension::try_from("backend.stub").expect("extension"),
                )
                .with_instance(
                    BackendInstanceProfile::new(
                        BackendInstance::new("stub:cpu"),
                        Backend::new("stub"),
                        DeviceProfile::new("cpu").with_kind(DeviceKind::Cpu),
                        BackendInstanceStatus::Available,
                    )
                    .with_capability(reimagine_inference::InferenceCapability::CreateEmptyLatent),
                )
        }

        fn build(
            &self,
            _config: &AppConfig,
            instance: &BackendInstance,
            device: Option<DeviceProfile>,
        ) -> Result<BuiltBackendInstance, BackendCandidateError> {
            let backend = Arc::new(FakeBackend::new("stub").create_empty_latent(
                CannedCapabilityResponse::from_request(|request: CreateEmptyLatentRequest| {
                    let batch_size = request.batch_size() as usize;
                    Ok(CreateEmptyLatentResponse::new(RuntimeLatent::new(
                        reimagine_inference::BackendTensorHandle::new(
                            Backend::new("stub"),
                            reimagine_inference::BackendPayloadKey::new("stub-latent"),
                            reimagine_core::model::TensorDType::F32,
                            reimagine_core::model::TensorShape::new(vec![
                                batch_size,
                                4,
                                (request.height() / 8) as usize,
                                (request.width() / 8) as usize,
                            ]),
                            "cpu",
                        ),
                        request.width(),
                        request.height(),
                        request.batch_size(),
                        4,
                        LatentSpaceMetadata::sdxl_base(),
                        LatentContent::EmptyGeometry,
                    )))
                }),
            ));
            let plugin = Plugin::try_from("builtin.stub").expect("plugin");
            let extension = Extension::try_from("backend.stub").expect("extension");
            let descriptor = BackendInstanceDescriptor::new(instance.clone(), Backend::new("stub"))
                .with_plugin(plugin.clone(), extension.clone());
            let descriptor = if let Some(device) = device.clone() {
                descriptor.with_device(device)
            } else {
                descriptor
            };
            let backend: Arc<dyn reimagine_inference::InferenceBackend> = backend;
            Ok(BuiltBackendInstance {
                descriptor,
                backend,
                runtime_hooks: Arc::new(StubRuntimeHooks {
                    instance: instance.clone(),
                    device,
                    plugin: Some(plugin),
                    extension: Some(extension),
                }),
            })
        }
    }

    #[derive(Debug)]
    struct StubRuntimeHooks {
        instance: BackendInstance,
        device: Option<DeviceProfile>,
        plugin: Option<Plugin>,
        extension: Option<Extension>,
    }

    #[async_trait::async_trait]
    impl BackendRunLifecycle for StubRuntimeHooks {
        fn backend_instance(&self) -> &BackendInstance {
            &self.instance
        }

        async fn begin_run(
            &self,
            _request: BackendRunLifecycleRequest,
        ) -> Result<BackendRunLifecycleReport, InferenceError> {
            Ok(BackendRunLifecycleReport {
                backend_instance: self.instance.clone(),
                diagnostics: Vec::new(),
            })
        }

        async fn cleanup_run(
            &self,
            _request: BackendRunLifecycleRequest,
        ) -> Result<BackendRunLifecycleReport, InferenceError> {
            Ok(BackendRunLifecycleReport {
                backend_instance: self.instance.clone(),
                diagnostics: Vec::new(),
            })
        }
    }

    #[async_trait::async_trait]
    impl BackendInstanceObservation for StubRuntimeHooks {
        fn backend_instance(&self) -> &BackendInstance {
            &self.instance
        }

        async fn snapshot(&self) -> BackendInstanceSnapshot {
            BackendInstanceSnapshot {
                backend_instance: self.instance.clone(),
                backend: Backend::new("stub"),
                plugin: self.plugin.clone(),
                extension: self.extension.clone(),
                device: self.device.clone(),
                observations: BTreeMap::new(),
                diagnostics: Vec::new(),
            }
        }
    }
}
