use std::collections::{HashMap, HashSet};

use reimagine_config::InferenceBackendConfig;
use reimagine_core::diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
    DiagnosticTargetDomain,
};
use reimagine_core::model::DiagnosticId;
use reimagine_inference::{
    BackendInstance, BackendInstanceProfile, BackendInstanceStatus, WorkspaceComputeProfile,
};

pub(crate) const CANDLE_CPU_FALLBACK: &str = "candle:cpu";
pub(crate) const CANDLE_CPU_FALLBACK_LABEL: &str = "cpu";

#[derive(Debug)]
pub(crate) struct ResolvedBackendSelection {
    pub(crate) selected_instance: BackendInstance,
    pub(crate) priority_order: Vec<BackendInstance>,
    pub(crate) disabled_instances: Vec<BackendInstance>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

pub(crate) fn resolve_backend_selection(
    config: &InferenceBackendConfig,
    profile: &WorkspaceComputeProfile,
) -> ResolvedBackendSelection {
    let disabled_instances = config
        .disabled_instances
        .iter()
        .map(|id| BackendInstance::new(id.trim()))
        .filter(|id| !id.as_str().is_empty())
        .collect::<Vec<_>>();
    let disabled = disabled_instances.iter().cloned().collect::<HashSet<_>>();
    let mut diagnostics = Vec::new();

    let explicit = config
        .selected_instance
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty());

    let selected_instance = if let Some(instance_id) = explicit {
        match resolve_open_selected_instance(instance_id, profile, &disabled) {
            Ok(instance) => instance,
            Err(diagnostic) => {
                diagnostics.push(diagnostic);
                BackendInstance::new(CANDLE_CPU_FALLBACK)
            }
        }
    } else {
        let (instance, fallback_diagnostics) =
            resolve_legacy_candle_device(profile, config.candle_device.trim(), &disabled);
        diagnostics.extend(fallback_diagnostics);
        instance
    };

    let mut priority_order = Vec::new();
    push_unique(&mut priority_order, selected_instance.clone());
    for configured in &config.priority_order {
        let configured = configured.trim();
        if configured.is_empty() {
            continue;
        }
        push_unique(&mut priority_order, BackendInstance::new(configured));
    }
    push_unique(
        &mut priority_order,
        BackendInstance::new(CANDLE_CPU_FALLBACK),
    );

    ResolvedBackendSelection {
        selected_instance,
        priority_order,
        disabled_instances,
        diagnostics,
    }
}

pub(crate) struct BackendProfilesByInstance<'a> {
    by_instance: HashMap<BackendInstance, &'a BackendInstanceProfile>,
}

impl<'a> BackendProfilesByInstance<'a> {
    pub(crate) fn new(profile: &'a WorkspaceComputeProfile) -> Self {
        let mut by_instance = HashMap::new();
        for backend in &profile.backend_profiles {
            for instance in &backend.instances {
                by_instance.insert(instance.instance.clone(), instance);
            }
        }
        Self { by_instance }
    }

    pub(crate) fn get(&self, instance: &BackendInstance) -> Option<&'a BackendInstanceProfile> {
        self.by_instance.get(instance).copied()
    }
}

pub(crate) fn resolved_candle_device_label(instance: &BackendInstance) -> String {
    instance_label(instance).to_string()
}

fn resolve_open_selected_instance(
    instance_id: &str,
    profile: &WorkspaceComputeProfile,
    disabled: &HashSet<BackendInstance>,
) -> Result<BackendInstance, Diagnostic> {
    let instance = BackendInstance::new(instance_id);
    if disabled.contains(&instance) {
        return Err(selected_instance_unavailable(
            instance_id,
            "backend instance is disabled by config",
        ));
    }
    let Some(profile) = find_instance_profile(profile, &instance) else {
        return Err(unknown_selected_instance(instance_id));
    };
    if profile.status != BackendInstanceStatus::Available {
        let reason = profile
            .diagnostics
            .first()
            .map(|d| d.message().to_string())
            .unwrap_or_else(|| "backend instance unavailable on this host".to_string());
        return Err(selected_instance_unavailable(instance_id, &reason));
    }
    Ok(instance)
}

fn resolve_legacy_candle_device(
    profile: &WorkspaceComputeProfile,
    configured_label: &str,
    disabled: &HashSet<BackendInstance>,
) -> (BackendInstance, Vec<Diagnostic>) {
    let normalized = configured_label.trim().to_ascii_lowercase();
    let lookup_label = match normalized.as_str() {
        "mps" => "metal",
        other => other,
    };
    let instance_id = format!("candle:{lookup_label}");
    let instance = BackendInstance::new(instance_id);

    let lookup = find_instance_profile(profile, &instance);
    match lookup {
        Some(profile) if profile.status == BackendInstanceStatus::Available => {
            if disabled.contains(&instance) {
                return (
                    BackendInstance::new(CANDLE_CPU_FALLBACK),
                    vec![selected_instance_unavailable(
                        instance.as_str(),
                        "backend instance is disabled by config",
                    )],
                );
            }
            (instance, Vec::new())
        }
        Some(profile) => {
            let reason = profile
                .diagnostics
                .first()
                .map(|d| d.message().to_string())
                .unwrap_or_else(|| "device unavailable on this host".to_string());
            let diagnostic_label = if normalized == "mps" {
                configured_label.trim()
            } else {
                lookup_label
            };
            (
                BackendInstance::new(CANDLE_CPU_FALLBACK),
                vec![reimagine_inference::diagnostics::candle_device_unavailable(
                    diagnostic_label,
                    &reason,
                )],
            )
        }
        None => (
            BackendInstance::new(CANDLE_CPU_FALLBACK),
            vec![reimagine_inference::diagnostics::invalid_candle_device(
                configured_label.trim(),
            )],
        ),
    }
}

fn find_instance_profile<'a>(
    profile: &'a WorkspaceComputeProfile,
    instance: &BackendInstance,
) -> Option<&'a BackendInstanceProfile> {
    profile
        .backend_profiles
        .iter()
        .flat_map(|backend| backend.instances.iter())
        .find(|candidate| &candidate.instance == instance)
}

fn push_unique(instances: &mut Vec<BackendInstance>, instance: BackendInstance) {
    if !instances.contains(&instance) {
        instances.push(instance);
    }
}

fn source() -> DiagnosticSourceName {
    DiagnosticSourceName::new("app-host")
}

fn target(path: impl Into<String>) -> DiagnosticTarget {
    DiagnosticTarget::new(DiagnosticTargetDomain::new("app-host.compute_profile"))
        .with_path(path.into())
}

fn unknown_selected_instance(instance: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!(
            "app-host-backend-selection-unknown-selected-{instance}"
        )),
        DiagnosticCode::new("APP_HOST/BACKEND_SELECTED_INSTANCE_UNKNOWN"),
        DiagnosticSeverity::Warning,
        source(),
        format!(
            "selected backend instance `{instance}` is not reported by any backend profile; falling back to `{CANDLE_CPU_FALLBACK}`"
        ),
        target(format!("selected_instance/{instance}")),
    )
}

fn selected_instance_unavailable(instance: &str, reason: &str) -> Diagnostic {
    Diagnostic::new(
        DiagnosticId::new(format!(
            "app-host-backend-selection-unavailable-selected-{instance}"
        )),
        DiagnosticCode::new("APP_HOST/BACKEND_SELECTED_INSTANCE_UNAVAILABLE"),
        DiagnosticSeverity::Warning,
        source(),
        format!(
            "selected backend instance `{instance}` is unavailable: {reason}; falling back to `{CANDLE_CPU_FALLBACK}`"
        ),
        target(format!("selected_instance/{instance}")),
    )
}

fn instance_label(instance: &BackendInstance) -> &str {
    instance
        .as_str()
        .split_once(':')
        .map(|(_, label)| label)
        .unwrap_or(CANDLE_CPU_FALLBACK_LABEL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_inference::{
        Backend, BackendInstanceProfile, BackendProfile, DeviceKind, DeviceProfile,
    };

    #[test]
    fn resolve_selected_instance_wins_over_legacy_candle_device() {
        let profile = profile_with_stub();
        let cfg = InferenceBackendConfig {
            selected_instance: Some("stub:cpu".to_string()),
            candle_device: "tpu".to_string(),
            ..InferenceBackendConfig::default()
        };

        let resolved = resolve_backend_selection(&cfg, &profile);

        assert_eq!(resolved.selected_instance, BackendInstance::new("stub:cpu"));
        assert!(resolved.diagnostics.is_empty());
        assert_eq!(resolved.priority_order[0], BackendInstance::new("stub:cpu"));
    }

    #[test]
    fn resolve_unknown_selected_instance_falls_back_to_candle_cpu() {
        let profile = profile_with_stub();
        let cfg = InferenceBackendConfig {
            selected_instance: Some("burn:cpu".to_string()),
            ..InferenceBackendConfig::default()
        };

        let resolved = resolve_backend_selection(&cfg, &profile);

        assert_eq!(
            resolved.selected_instance,
            BackendInstance::new(CANDLE_CPU_FALLBACK)
        );
        assert_eq!(resolved.diagnostics.len(), 1);
        assert_eq!(
            resolved.diagnostics[0].code().as_str(),
            "APP_HOST/BACKEND_SELECTED_INSTANCE_UNKNOWN"
        );
    }

    fn profile_with_stub() -> WorkspaceComputeProfile {
        WorkspaceComputeProfile::new()
            .with_backend_profile(BackendProfile::new(Backend::new("candle")).with_instance(
                BackendInstanceProfile::new(
                    BackendInstance::new("candle:cpu"),
                    Backend::new("candle"),
                    DeviceProfile::new("cpu").with_kind(DeviceKind::Cpu),
                    BackendInstanceStatus::Available,
                ),
            ))
            .with_backend_profile(BackendProfile::new(Backend::new("stub")).with_instance(
                BackendInstanceProfile::new(
                    BackendInstance::new("stub:cpu"),
                    Backend::new("stub"),
                    DeviceProfile::new("cpu").with_kind(DeviceKind::Cpu),
                    BackendInstanceStatus::Available,
                ),
            ))
    }
}
