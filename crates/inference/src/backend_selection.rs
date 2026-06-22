//! Backend selection policy and instance descriptors.
//!
//! The router owns deterministic backend selection. It consults a
//! [`BackendSelectionPolicy`] trait to derive an ordered list of
//! candidate [`BackendInstance`]s for a given request, then validates
//! the first viable candidate against the registered registry and the
//! requested capability.
//!
//! The vocabulary is:
//!
//! - [`Backend`] — open implementation label (e.g. `"candle"`).
//! - [`BackendInstance`] — configured instance of a backend (e.g.
//!   `"candle:metal"`). The instance is the unit of selection.
//! - [`DeviceProfile`] — opaque descriptor attached to a backend
//!   instance for diagnostics and future device-aware scheduling.
//! - [`BackendInstanceDescriptor`] — registry-side metadata: the
//!   instance, the open backend label, optional device, optional
//!   plugin provenance.
//! - [`BackendSelectionRequest`] — what the policy sees: capability,
//!   node id, handle affinities, registered descriptors, and any
//!   explicit override from the execution request overlay.
//! - [`BackendOverrides`] — explicit per-call overrides supplied
//!   through an execution request overlay, not persisted in workflow
//!   JSON.
//! - [`BackendSelectionOverlay`] — per-request overlay that carries
//!   an explicit override and is attached to the request DTOs
//!   (e.g. [`crate::LoadBundleRequest`]).
//!
//! Selection precedence is fixed in
//! [`crate::router::DefaultInferenceRuntime`]:
//!
//! 1. Existing backend-bound handle affinities.
//! 2. Explicit override from the request overlay (when no
//!    incompatible handles exist).
//! 3. Priority order from the policy.
//! 4. Diagnostic failure.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::capability::InferenceCapability;
use reimagine_core::model::NodeId;
use reimagine_plugin::{Extension, Plugin};

/// Open stable backend implementation label, e.g. `"candle"` or
/// `"remote"`. Distinct from a concrete configured [`BackendInstance`].
///
/// Backend labels are not a closed enum. Built-in and future external
/// backends are modeled as plugin extensions over
/// `HostSurface::InferenceBackend`; the open label is the stable
/// identity they advertise.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Backend(String);

impl Backend {
    /// Construct a `Backend` from a label. Matches the
    /// `core::ModelId` shape; trust the caller or validate at the
    /// system boundary.
    pub fn new(label: impl Into<String>) -> Self {
        Self(label.into())
    }

    /// Borrow the underlying label.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Backend {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for Backend {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Configured backend instance identity, e.g. `"candle:metal"`.
///
/// The instance is the unit of selection. A single [`Backend`]
/// implementation may register multiple instances with different
/// devices, profiles, or configurations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BackendInstance(String);

impl BackendInstance {
    /// Construct a `BackendInstance` from an identity. Matches the
    /// `core::ModelId` shape; trust the caller or validate at the
    /// system boundary.
    pub fn new(instance: impl Into<String>) -> Self {
        Self(instance.into())
    }

    /// Borrow the underlying identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BackendInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for BackendInstance {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for BackendInstance {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Opaque descriptor of a backend's device, populated by app-host or
/// the concrete backend. V1 router policy treats this as descriptive
/// metadata only; future device-aware scheduling may match on it.
///
/// `#[non_exhaustive]` allows adding structured fields (compute
/// capability, memory class, …) without breaking downstream matches.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DeviceProfile {
    /// Opaque label (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub label: String,
}

impl DeviceProfile {
    /// Construct a `DeviceProfile` from a label.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

/// Registry-side metadata for a single configured backend instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendInstanceDescriptor {
    /// Stable instance identity (the unit of selection).
    pub instance: BackendInstance,
    /// Open backend implementation label.
    pub backend: Backend,
    /// Optional device metadata.
    pub device: Option<DeviceProfile>,
    /// Optional plugin that contributed this instance. Populated when
    /// the instance was discovered through a
    /// `PluginExtension` over `HostSurface::InferenceBackend`.
    pub plugin: Option<Plugin>,
    /// Optional extension identity within the contributing plugin.
    pub extension: Option<Extension>,
}

impl BackendInstanceDescriptor {
    /// Construct a descriptor with the minimum required fields.
    pub fn new(instance: BackendInstance, backend: Backend) -> Self {
        Self {
            instance,
            backend,
            device: None,
            plugin: None,
            extension: None,
        }
    }

    /// Attach a device profile.
    pub fn with_device(mut self, device: DeviceProfile) -> Self {
        self.device = Some(device);
        self
    }

    /// Attach plugin provenance.
    pub fn with_plugin(mut self, plugin: Plugin, extension: Extension) -> Self {
        self.plugin = Some(plugin);
        self.extension = Some(extension);
        self
    }
}

/// Per-request overlay that the router consults before falling back
/// to the policy. Populated by the runtime from workspace
/// configuration; never persisted in workflow JSON.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendSelectionOverlay {
    /// Explicit per-call override. When set, the router validates the
    /// instance against the registry, allowed/disabled sets, and
    /// capability support before using it.
    pub explicit_override: Option<BackendInstance>,
}

impl BackendSelectionOverlay {
    /// Empty overlay (no explicit override).
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an overlay that pins one backend instance.
    pub fn with_explicit_override(instance: BackendInstance) -> Self {
        Self {
            explicit_override: Some(instance),
        }
    }
}

/// All the context the policy needs to produce an ordered candidate
/// list for a single router dispatch.
///
#[derive(Debug, Clone)]
pub struct BackendSelectionRequest {
    /// The capability the call needs to perform.
    pub capability: InferenceCapability,
    /// Originating workflow node, when known. Useful for diagnostics
    /// and future per-node policy.
    pub node_id: Option<NodeId>,
    /// Concrete [`BackendInstance`] affinities implied by existing
    /// backend-bound handles (e.g. an input `Model` or `Latent`).
    /// The router must not silently move payload handles between
    /// instances, even when those instances share the same open
    /// [`Backend`] label.
    pub affinities: Vec<BackendInstance>,
    /// Registered backend instance descriptors, in registration order.
    pub registered: Vec<BackendInstanceDescriptor>,
    /// Explicit override from the request overlay, if any.
    pub explicit_override: Option<BackendInstance>,
}

/// Inputs the [`StaticBackendSelectionPolicy`] applies when building
/// its candidate list. The policy is the default; concrete workspaces
/// may inject a different implementation.
///
/// V1 carries no fields. Per-call explicit overrides are supplied
/// through the request DTO's [`BackendSelectionOverlay`]; per-node
/// overrides and a richer allow/disable set are future work. The
/// type is preserved so the [`StaticBackendSelectionPolicy`]
/// surface matches the issue's suggested shape.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendOverrides {}

impl BackendOverrides {
    /// Construct an empty `BackendOverrides`.
    pub fn new() -> Self {
        Self {}
    }
}

/// Default V1 policy: a static priority order combined with
/// allow/disable filters. The router handles explicit overrides and
/// affinity validation on top of the candidate list this policy
/// returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticBackendSelectionPolicy {
    /// Explicit per-call overrides (for future per-node metadata).
    pub explicit_overrides: BackendOverrides,
    /// Ordered priority list. The first registered, allowed,
    /// non-disabled, capability-supporting candidate is chosen.
    pub priority_order: Vec<BackendInstance>,
    /// Allow-list. When `Some`, only listed instances are eligible
    /// unless promoted by an explicit override.
    pub allowed: Option<Vec<BackendInstance>>,
    /// Disable list; matching instances are always skipped.
    pub disabled: Vec<BackendInstance>,
}

impl StaticBackendSelectionPolicy {
    /// Construct a policy with the given priority order. All other
    /// fields default to permissive.
    pub fn new(priority_order: Vec<BackendInstance>) -> Self {
        Self {
            explicit_overrides: BackendOverrides::new(),
            priority_order,
            allowed: None,
            disabled: Vec::new(),
        }
    }

    /// Construct a policy with explicit overrides, priority order,
    /// allow-list, and disable list.
    pub fn with_overrides(
        explicit_overrides: BackendOverrides,
        priority_order: Vec<BackendInstance>,
        allowed: Option<Vec<BackendInstance>>,
        disabled: Vec<BackendInstance>,
    ) -> Self {
        Self {
            explicit_overrides,
            priority_order,
            allowed,
            disabled,
        }
    }
}

impl Default for StaticBackendSelectionPolicy {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl BackendSelectionPolicy for StaticBackendSelectionPolicy {
    fn candidates(&self, request: &BackendSelectionRequest) -> Vec<BackendInstance> {
        let mut out: Vec<BackendInstance> = Vec::new();

        let allowed = self.allowed.as_ref();

        // When the priority list is empty, fall back to the
        // registered list in registration order. This is the
        // "first registered + allowed + capability-supported"
        // fallback the issue's selection semantics describe.
        let source: Box<dyn Iterator<Item = &BackendInstance>> = if self.priority_order.is_empty() {
            Box::new(request.registered.iter().map(|d| &d.instance))
        } else {
            Box::new(self.priority_order.iter())
        };

        for instance in source {
            if self.disabled.contains(instance) {
                continue;
            }
            if let Some(allow) = allowed {
                if !allow.contains(instance) {
                    continue;
                }
            }
            if out.contains(instance) {
                continue;
            }
            out.push(instance.clone());
        }

        out
    }

    fn allows_explicit_override(
        &self,
        instance: &BackendInstance,
        _request: &BackendSelectionRequest,
    ) -> bool {
        if self.disabled.contains(instance) {
            return false;
        }
        if let Some(allowed) = &self.allowed {
            if !allowed.contains(instance) {
                return false;
            }
        }
        true
    }
}

/// Pluggable selection policy. Implementations produce an ordered
/// list of candidate [`BackendInstance`]s; the router handles
/// affinity validation, explicit overrides, and capability checks.
pub trait BackendSelectionPolicy: Send + Sync + 'static {
    /// Return candidates in priority order. The first viable
    /// candidate is selected by the router.
    fn candidates(&self, request: &BackendSelectionRequest) -> Vec<BackendInstance>;

    /// Return whether an explicit request overlay may select this
    /// backend instance. The router still performs registration and
    /// capability checks; this method lets policy-owned allow/disable
    /// rules apply to explicit overrides.
    fn allows_explicit_override(
        &self,
        instance: &BackendInstance,
        request: &BackendSelectionRequest,
    ) -> bool;
}

/// Convenience alias for an `Arc`-wrapped selection policy.
pub type ArcBackendSelectionPolicy = Arc<dyn BackendSelectionPolicy>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_new_holds_label() {
        assert_eq!(Backend::new("candle").as_str(), "candle");
        assert_eq!(Backend::from("candle").as_str(), "candle");
    }

    #[test]
    fn backend_instance_new_holds_identity() {
        assert_eq!(
            BackendInstance::new("candle:metal").as_str(),
            "candle:metal"
        );
        assert_eq!(
            BackendInstance::from("candle:metal").as_str(),
            "candle:metal"
        );
    }

    #[test]
    fn descriptor_accepts_optional_provenance() {
        let plugin = Plugin::try_from("builtin.candle").unwrap();
        let extension = Extension::try_from("backend.candle").unwrap();
        let desc = BackendInstanceDescriptor::new(
            BackendInstance::new("candle:metal"),
            Backend::new("candle"),
        )
        .with_device(DeviceProfile::new("metal"))
        .with_plugin(plugin.clone(), extension.clone());

        assert_eq!(desc.device.as_ref().unwrap().label, "metal");
        assert_eq!(desc.plugin.as_ref().unwrap(), &plugin);
        assert_eq!(desc.extension.as_ref().unwrap(), &extension);
    }

    #[test]
    fn static_policy_skips_disabled_and_respects_allow_list() {
        let policy = StaticBackendSelectionPolicy::with_overrides(
            BackendOverrides::new(),
            vec![
                BackendInstance::new("candle:metal"),
                BackendInstance::new("candle:cuda"),
            ],
            Some(vec![BackendInstance::new("candle:metal")]),
            vec![BackendInstance::new("candle:cuda")],
        );

        let request = BackendSelectionRequest {
            capability: InferenceCapability::CreateEmptyLatent,
            node_id: None,
            affinities: Vec::new(),
            registered: Vec::new(),
            explicit_override: None,
        };
        let candidates = policy.candidates(&request);
        // "candle:metal" is allow-listed and not disabled, so it is
        // returned. "candle:cuda" is disabled and is filtered out.
        assert_eq!(candidates, vec![BackendInstance::new("candle:metal")]);
    }

    #[test]
    fn static_policy_applies_allow_disable_to_explicit_overrides() {
        let policy = StaticBackendSelectionPolicy::with_overrides(
            BackendOverrides::new(),
            Vec::new(),
            Some(vec![BackendInstance::new("candle:metal")]),
            vec![BackendInstance::new("candle:cuda")],
        );
        let request = BackendSelectionRequest {
            capability: InferenceCapability::CreateEmptyLatent,
            node_id: None,
            affinities: Vec::new(),
            registered: vec![
                BackendInstanceDescriptor::new(
                    BackendInstance::new("candle:metal"),
                    Backend::new("candle"),
                ),
                BackendInstanceDescriptor::new(
                    BackendInstance::new("candle:cuda"),
                    Backend::new("candle"),
                ),
                BackendInstanceDescriptor::new(
                    BackendInstance::new("burn:cuda"),
                    Backend::new("burn"),
                ),
            ],
            explicit_override: None,
        };

        assert!(policy.allows_explicit_override(&BackendInstance::new("candle:metal"), &request));
        assert!(!policy.allows_explicit_override(&BackendInstance::new("candle:cuda"), &request));
        assert!(!policy.allows_explicit_override(&BackendInstance::new("burn:cuda"), &request));
        assert!(!policy.allows_explicit_override(&BackendInstance::new("missing"), &request));
    }

    #[test]
    fn static_policy_falls_back_to_registered_when_priority_empty() {
        let policy = StaticBackendSelectionPolicy::new(Vec::new());
        let request = BackendSelectionRequest {
            capability: InferenceCapability::CreateEmptyLatent,
            node_id: None,
            affinities: Vec::new(),
            registered: vec![
                BackendInstanceDescriptor::new(BackendInstance::new("a:cpu"), Backend::new("a")),
                BackendInstanceDescriptor::new(BackendInstance::new("b:cpu"), Backend::new("b")),
            ],
            explicit_override: None,
        };
        let candidates = policy.candidates(&request);
        assert_eq!(
            candidates,
            vec![BackendInstance::new("a:cpu"), BackendInstance::new("b:cpu"),]
        );
    }

    #[test]
    fn static_policy_priority_only() {
        let policy = StaticBackendSelectionPolicy::new(vec![
            BackendInstance::new("burn:cuda"),
            BackendInstance::new("candle:metal"),
        ]);
        let request = BackendSelectionRequest {
            capability: InferenceCapability::CreateEmptyLatent,
            node_id: None,
            affinities: Vec::new(),
            registered: Vec::new(),
            explicit_override: None,
        };
        let candidates = policy.candidates(&request);
        assert_eq!(
            candidates,
            vec![
                BackendInstance::new("burn:cuda"),
                BackendInstance::new("candle:metal"),
            ]
        );
    }

    #[test]
    fn policy_with_no_priority_produces_no_candidates() {
        let policy = StaticBackendSelectionPolicy::default();
        let request = BackendSelectionRequest {
            capability: InferenceCapability::CreateEmptyLatent,
            node_id: None,
            affinities: Vec::new(),
            registered: Vec::new(),
            explicit_override: None,
        };
        assert!(policy.candidates(&request).is_empty());
    }

    #[test]
    fn overlay_defaults_to_no_explicit_override() {
        let overlay = BackendSelectionOverlay::new();
        assert_eq!(overlay.explicit_override, None);

        let pinned =
            BackendSelectionOverlay::with_explicit_override(BackendInstance::new("candle:metal"));
        assert_eq!(
            pinned.explicit_override,
            Some(BackendInstance::new("candle:metal"))
        );
    }
}
