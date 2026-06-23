//! Workspace bootstrap resolution: configured label → backend instance.
//!
//! App-host owns the bridge between a user's persisted
//! [`InferenceBackendConfig`](reimagine_config::InferenceBackendConfig)
//! and the live hardware the concrete backend crates can actually
//! use. The persisted config stores a label (e.g. `"cpu"`, `"metal"`,
//! `"mps"`) and possibly an outdated device label. The
//! [`BackendProfileProvider`](reimagine_inference::BackendProfileProvider)
//! returns the host's current backend-instance candidates with
//! [`Available`](reimagine_inference::BackendInstanceStatus::Available) /
//! [`Unavailable`](reimagine_inference::BackendInstanceStatus::Unavailable)
//! status. This module reconciles the two and emits top-level
//! diagnostics on fallback so [`WorkspaceHost::compute_profile`](crate::workspace::WorkspaceHost::compute_profile)
//! reports the decision.
//!
//! V1 only knows the Candle backend. When additional backends land,
//! the resolver should move behind a small backend-keyed dispatch and
//! each backend crate should expose its own resolver helper. The
//! function shapes here deliberately take a `&BackendProfile` and a
//! `&str` so they can be unit-tested without a live `WorkspaceHost`.

use reimagine_core::diagnostic::Diagnostic;
use reimagine_inference::{
    BackendInstance, BackendInstanceStatus, BackendProfile,
    diagnostics::{candle_device_unavailable, invalid_candle_device},
};

/// Backend instance identity app-host falls back to when the
/// configured device label cannot be honored.
pub(crate) const CANDLE_CPU_FALLBACK: &str = "candle:cpu";

/// Resolve the configured Candle device label to a concrete
/// [`BackendInstance`] using the freshly probed [`BackendProfile`].
///
/// Resolution rules:
///
/// 1. The label is lowercased. `mps` is normalized to `metal` so the
///    configured `mps` selects the canonical `candle:metal` instance.
/// 2. The matching instance in `profile.instances` is located by
///    `format!("{backend}:{normalized_label}")`.
/// 3. If the instance is
///    [`Available`](reimagine_inference::BackendInstanceStatus::Available),
///    it is returned with no diagnostics.
/// 4. If the instance is
///    [`Unavailable`](reimagine_inference::BackendInstanceStatus::Unavailable),
///    fallback [`CANDLE_CPU_FALLBACK`] is returned with a
///    [`candle_device_unavailable`] diagnostic that reuses the
///    existing instance diagnostic's reason when present.
/// 5. If no instance matches the resolved label, fallback
///    [`CANDLE_CPU_FALLBACK`] is returned with an
///    [`invalid_candle_device`] diagnostic naming the original
///    configured label (so users see `mps` even though the resolver
///    normalized it for lookup).
///
/// Returned diagnostics are intended to be appended to the
/// [`WorkspaceComputeProfile::diagnostics`](reimagine_inference::WorkspaceComputeProfile::diagnostics)
/// top-level collection by the caller.
pub(crate) fn resolve_candle_instance(
    profile: &BackendProfile,
    configured_label: &str,
) -> (BackendInstance, Vec<Diagnostic>) {
    let normalized = configured_label.trim().to_ascii_lowercase();
    let lookup_label = match normalized.as_str() {
        "mps" => "metal",
        other => other,
    };
    let instance_id = format!("{}:{lookup_label}", profile.backend.as_str());

    let lookup = profile
        .instances
        .iter()
        .find(|i| i.instance.as_str() == instance_id);

    match lookup {
        Some(instance) if instance.status == BackendInstanceStatus::Available => {
            (instance.instance.clone(), Vec::new())
        }
        Some(instance) => {
            let reason = instance
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
                vec![candle_device_unavailable(diagnostic_label, &reason)],
            )
        }
        None => (
            BackendInstance::new(CANDLE_CPU_FALLBACK),
            vec![invalid_candle_device(configured_label.trim())],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_inference::{Backend, BackendInstanceProfile, DeviceKind, DeviceProfile};

    fn profile_with(cpu_available: bool, metal_available: bool) -> BackendProfile {
        let backend = Backend::new("candle");
        let cpu_status = if cpu_available {
            BackendInstanceStatus::Available
        } else {
            BackendInstanceStatus::Unavailable
        };
        let metal_status = if metal_available {
            BackendInstanceStatus::Available
        } else {
            BackendInstanceStatus::Unavailable
        };
        let mut cpu = BackendInstanceProfile::new(
            BackendInstance::new("candle:cpu"),
            backend.clone(),
            DeviceProfile::new("cpu").with_kind(DeviceKind::Cpu),
            cpu_status,
        );
        if !cpu_available {
            cpu = cpu.with_diagnostic(candle_device_unavailable("cpu", "cpu disabled"));
        }
        let mut metal = BackendInstanceProfile::new(
            BackendInstance::new("candle:metal"),
            backend.clone(),
            DeviceProfile::new("metal").with_kind(DeviceKind::Gpu),
            metal_status,
        );
        if !metal_available {
            metal = metal.with_diagnostic(candle_device_unavailable("metal", "no metal runtime"));
        }
        BackendProfile::new(backend)
            .with_instance(cpu)
            .with_instance(metal)
    }

    #[test]
    fn resolve_picks_cpu_when_cpu_available() {
        let profile = profile_with(true, false);
        let (instance, diagnostics) = resolve_candle_instance(&profile, "cpu");
        assert_eq!(instance, BackendInstance::new("candle:cpu"));
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn resolve_picks_metal_when_metal_available() {
        let profile = profile_with(true, true);
        let (instance, diagnostics) = resolve_candle_instance(&profile, "metal");
        assert_eq!(instance, BackendInstance::new("candle:metal"));
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn resolve_normalizes_mps_to_metal_when_metal_available() {
        let profile = profile_with(true, true);
        let (instance, diagnostics) = resolve_candle_instance(&profile, "mps");
        assert_eq!(instance, BackendInstance::new("candle:metal"));
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn resolve_normalizes_mps_to_metal_then_falls_back_when_metal_unavailable() {
        let profile = profile_with(true, false);
        let (instance, diagnostics) = resolve_candle_instance(&profile, "mps");
        assert_eq!(instance, BackendInstance::new("candle:cpu"));
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message().contains("mps"));
    }

    #[test]
    fn resolve_falls_back_for_unknown_label_with_invalid_diagnostic() {
        let profile = profile_with(true, true);
        let (instance, diagnostics) = resolve_candle_instance(&profile, "tpu");
        assert_eq!(instance, BackendInstance::new("candle:cpu"));
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code().as_str(),
            "INFERENCE_PROFILE/INVALID_DEVICE"
        );
        assert!(diagnostics[0].message().contains("tpu"));
    }

    #[test]
    fn resolve_falls_back_for_unknown_label_when_metal_unavailable() {
        let profile = profile_with(true, false);
        let (instance, diagnostics) = resolve_candle_instance(&profile, "tpu");
        assert_eq!(instance, BackendInstance::new("candle:cpu"));
        assert_eq!(
            diagnostics[0].code().as_str(),
            "INFERENCE_PROFILE/INVALID_DEVICE"
        );
    }

    #[test]
    fn resolve_handles_uppercase_and_whitespace_labels() {
        let profile = profile_with(true, true);
        let (instance, _) = resolve_candle_instance(&profile, "  Metal  ");
        assert_eq!(instance, BackendInstance::new("candle:metal"));
    }

    #[test]
    fn resolve_cpu_unavailable_still_falls_back_to_cpu() {
        // If cpu is somehow unavailable, the fallback target should
        // still be `candle:cpu` so compose_inference_backends gets a
        // resolvable instance; a diagnostic still emits so callers
        // can surface the unexpected state.
        let profile = profile_with(false, false);
        let (instance, diagnostics) = resolve_candle_instance(&profile, "tpu");
        assert_eq!(instance, BackendInstance::new("candle:cpu"));
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].code().as_str(),
            "INFERENCE_PROFILE/INVALID_DEVICE"
        );
    }
}
