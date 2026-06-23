//! Host-neutral compute, backend, and device profile vocabulary.
//!
//! This module is the V1 contract surface for the workspace capability
//! discovery slice described in
//! `docs/architecture/real-inference-roadmap.md` ("Workspace
//! Capability Discovery") and `docs/architecture/modules/inference.md`
//! ("Compute And Device Profiles").
//!
//! The vocabulary here is observation/configuration only. These DTOs
//! must not carry backend-native device handles, tensors, loaded model
//! structs, tokenizer state, graph objects, file handles, or
//! OS-specific resource owners. Concrete backend crates implement
//! [`BackendProfileProvider`] and report the devices they can actually
//! use with their own build features, runtime libraries, device APIs,
//! dtype support, and capability constraints. `inference` does not
//! perform one global hardware probe for every backend.
//!
//! App-host collects providers and exposes the aggregate
//! [`WorkspaceComputeProfile`] to Tauri / Axum / Agent / UI. Runtime
//! and node executors never consume these DTOs.
//!
//! V1 omits `generated_at` so profile tests remain deterministic.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use reimagine_core::diagnostic::Diagnostic;
use reimagine_plugin::{Extension, Plugin};

use crate::backend_selection::{Backend, BackendInstance, DeviceProfile};
use crate::capability::InferenceCapability;

// ── DeviceKind ─────────────────────────────────────────────────────

/// Coarse device kind derived from a [`DeviceProfile`] label or
/// supplied by a concrete backend probe.
///
/// `#[non_exhaustive]` is intentionally not used: the V1 set is fixed
/// and closed. New device families should land in a later slice that
/// also revisits the serde representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum DeviceKind {
    /// CPU device (`"cpu"` and any unrecognized-but-cpu-shaped labels).
    Cpu,
    /// GPU-class device: `"metal"`, `"mps"`, `"cuda"`, `"cuda:0"`, …
    Gpu,
    /// Remote / network-attached device (`"remote"`, `"remote:foo"`, …).
    Remote,
    /// Unrecognized device label.
    Unknown,
}

/// Derive a [`DeviceKind`] from a device label.
///
/// Mapping:
///
/// - `cpu` → [`DeviceKind::Cpu`]
/// - `metal`, `mps`, `cuda`, `cuda:0`, … → [`DeviceKind::Gpu`]
/// - `remote`, `remote:foo`, … → [`DeviceKind::Remote`]
/// - anything else → [`DeviceKind::Unknown`]
///
/// Matching is case-insensitive on the bare label and on the prefix
/// before `:` for indexed device labels.
pub fn kind_from_label(label: &str) -> DeviceKind {
    let normalized = label.to_ascii_lowercase();
    match normalized.as_str() {
        "cpu" => DeviceKind::Cpu,
        "metal" | "mps" => DeviceKind::Gpu,
        l if l == "cuda" || l.starts_with("cuda:") => DeviceKind::Gpu,
        l if l == "remote" || l.starts_with("remote:") => DeviceKind::Remote,
        _ => DeviceKind::Unknown,
    }
}

// ── MemoryProfile ──────────────────────────────────────────────────

/// Optional memory summary attached to a [`DeviceProfile`].
///
/// All fields are `None` when the backend did not probe memory or the
/// device has no memory concept. Bytes are reported using SI units
/// (`1 MiB == 1_048_576 bytes`); the exact unit is the backend's
/// reporting choice but V1 callers should treat it as opaque.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Total bytes available on the device, when known.
    pub total_bytes: Option<u64>,
    /// Bytes currently free on the device, when known.
    pub free_bytes: Option<u64>,
}

impl MemoryProfile {
    /// Construct an empty memory summary (both fields `None`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a memory summary with both fields populated.
    pub fn with_total_and_free(total_bytes: u64, free_bytes: u64) -> Self {
        Self {
            total_bytes: Some(total_bytes),
            free_bytes: Some(free_bytes),
        }
    }

    /// Attach a `total_bytes` value.
    pub fn with_total(mut self, total_bytes: u64) -> Self {
        self.total_bytes = Some(total_bytes);
        self
    }

    /// Attach a `free_bytes` value.
    pub fn with_free(mut self, free_bytes: u64) -> Self {
        self.free_bytes = Some(free_bytes);
        self
    }
}

// ── DTypeProfile ───────────────────────────────────────────────────

/// Host-neutral dtype string attached to a [`DeviceProfile`] as a
/// supported dtype.
///
/// V1 stores dtype identifiers as opaque strings (e.g. `"f32"`,
/// `"f16"`, `"bf16"`, `"u8"`). Backends report what they actually
/// support; the inference crate does not validate the dtype vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DTypeProfile {
    /// Opaque dtype identifier (e.g. `"f32"`).
    pub dtype: String,
}

impl DTypeProfile {
    /// Construct a `DTypeProfile` from a dtype identifier.
    pub fn new(dtype: impl Into<String>) -> Self {
        Self {
            dtype: dtype.into(),
        }
    }

    /// Borrow the underlying dtype identifier.
    pub fn as_str(&self) -> &str {
        &self.dtype
    }
}

impl From<&str> for DTypeProfile {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for DTypeProfile {
    fn from(value: String) -> Self {
        Self { dtype: value }
    }
}

// ── BackendInstanceStatus ──────────────────────────────────────────

/// Whether a backend instance is currently usable on the host.
///
/// A backend may enumerate a candidate instance (e.g. `candle:metal`)
/// and report it as `Unavailable` when the host cannot construct the
/// device (no Metal runtime on Linux, no CUDA driver, etc.). UI / API
/// surfaces use the status to decide whether to offer the instance as
/// a selectable option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum BackendInstanceStatus {
    /// Backend probed the device and the instance is ready to use.
    Available,
    /// Backend enumerated the instance but cannot construct it on this
    /// host. The reason should be recorded as a
    /// [`Diagnostic`](reimagine_core::diagnostic::Diagnostic) on the
    /// matching [`BackendInstanceProfile::diagnostics`].
    Unavailable,
}

// ── WorkspaceComputeProfile ────────────────────────────────────────

/// Aggregate profile returned by `WorkspaceHost::compute_profile()`.
///
/// Aggregates one [`BackendProfile`] per registered backend, plus a
/// flat list of top-level diagnostics that do not belong to a single
/// backend (e.g. configuration errors detected during discovery).
///
/// This is the host-neutral DTO surfaced to Tauri / Axum / Agent / UI.
/// Host adapters should project it through app-host DTOs rather than
/// bind directly to this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceComputeProfile {
    /// Per-backend profiles, one per registered backend provider.
    pub backend_profiles: Vec<BackendProfile>,
    /// Top-level diagnostics that do not belong to a single backend.
    pub diagnostics: Vec<Diagnostic>,
}

impl WorkspaceComputeProfile {
    /// Construct an empty workspace profile.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a backend profile.
    pub fn with_backend_profile(mut self, profile: BackendProfile) -> Self {
        self.backend_profiles.push(profile);
        self
    }

    /// Add a top-level diagnostic.
    pub fn with_diagnostic(mut self, diagnostic: Diagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }
}

impl Default for WorkspaceComputeProfile {
    fn default() -> Self {
        Self {
            backend_profiles: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}

// ── BackendProfile ─────────────────────────────────────────────────

/// Per-backend profile contributed by one
/// [`BackendProfileProvider`].
///
/// Carries the open [`Backend`] label, optional plugin / extension
/// provenance, the candidate backend instances the provider can
/// construct on this host, and any diagnostics emitted during probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendProfile {
    /// Stable backend implementation label (e.g. `"candle"`,
    /// `"remote"`).
    pub backend: Backend,
    /// Optional plugin that contributed this backend.
    pub plugin: Option<Plugin>,
    /// Optional extension identity within the contributing plugin.
    pub extension: Option<Extension>,
    /// Backend instance candidates the provider can construct on this
    /// host. Order is provider-defined; UI surfaces may sort by
    /// status / kind.
    pub instances: Vec<BackendInstanceProfile>,
    /// Diagnostics emitted by this provider during profile
    /// construction.
    pub diagnostics: Vec<Diagnostic>,
}

impl BackendProfile {
    /// Construct a `BackendProfile` with no plugin provenance, no
    /// instances, and no diagnostics.
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            plugin: None,
            extension: None,
            instances: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Attach plugin provenance.
    pub fn with_plugin(mut self, plugin: Plugin, extension: Extension) -> Self {
        self.plugin = Some(plugin);
        self.extension = Some(extension);
        self
    }

    /// Append a backend instance profile.
    pub fn with_instance(mut self, instance: BackendInstanceProfile) -> Self {
        self.instances.push(instance);
        self
    }

    /// Append a diagnostic.
    pub fn with_diagnostic(mut self, diagnostic: Diagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }
}

// ── BackendInstanceProfile ────────────────────────────────────────

/// Profile for a single backend instance candidate (e.g.
/// `"candle:metal"`, `"candle:cpu"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendInstanceProfile {
    /// Stable backend instance identity.
    pub instance: BackendInstance,
    /// Open backend implementation label. Mirrors the enclosing
    /// [`BackendProfile::backend`] but is included here so each
    /// instance profile is self-contained.
    pub backend: Backend,
    /// Device descriptor for this instance.
    pub device: DeviceProfile,
    /// Capabilities this backend instance advertises as supported on
    /// its device.
    pub capabilities: Vec<InferenceCapability>,
    /// Whether this instance is currently usable on the host.
    pub status: BackendInstanceStatus,
    /// Diagnostics emitted while probing this instance. Unavailable
    /// instances typically carry a diagnostic explaining why.
    pub diagnostics: Vec<Diagnostic>,
}

impl BackendInstanceProfile {
    /// Construct a minimal instance profile with empty diagnostics
    /// and capabilities.
    pub fn new(
        instance: BackendInstance,
        backend: Backend,
        device: DeviceProfile,
        status: BackendInstanceStatus,
    ) -> Self {
        Self {
            instance,
            backend,
            device,
            capabilities: Vec::new(),
            status,
            diagnostics: Vec::new(),
        }
    }

    /// Append a capability.
    pub fn with_capability(mut self, capability: InferenceCapability) -> Self {
        self.capabilities.push(capability);
        self
    }

    /// Append a diagnostic.
    pub fn with_diagnostic(mut self, diagnostic: Diagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }
}

// ── BackendProfileProvider ─────────────────────────────────────────

/// Contract implemented by concrete backend crates to report their
/// host-neutral profile.
///
/// Providers are discovery providers or factories, not necessarily
/// already-constructed executable backend instances. A provider may
/// enumerate candidates that app-host has not yet constructed (e.g.
/// the Candle provider can advertise `candle:metal` even on a Linux
/// host as `Unavailable` with a diagnostic).
#[async_trait]
pub trait BackendProfileProvider: Send + Sync {
    /// Return the backend's host-neutral profile.
    ///
    /// Implementations should treat this as a synchronous, in-process
    /// probe. Heavy I/O (e.g. probing CUDA driver versions) should be
    /// cached by the implementation; V1 calls this once per
    /// workspace bootstrap.
    async fn backend_profile(&self) -> BackendProfile;
}

// ── Diagnostic helpers ─────────────────────────────────────────────

/// Diagnostic helpers for inference-side profile construction.
///
/// All helpers emit diagnostics whose primary target domain is
/// `app-host.compute_profile` (the app-host surface that aggregates
/// and exposes the workspace compute profile). The
/// [`DiagnosticSourceName`] is `"inference"`, matching the
/// post-fold source used by other inference-side helpers (e.g.
/// `inference::resources`).
pub mod diagnostics {
    use reimagine_core::diagnostic::{
        Diagnostic, DiagnosticCode, DiagnosticSeverity, DiagnosticSourceName, DiagnosticTarget,
        DiagnosticTargetDomain,
    };
    use reimagine_core::model::DiagnosticId;

    fn source() -> DiagnosticSourceName {
        DiagnosticSourceName::new("inference")
    }

    fn target(path: impl Into<String>) -> DiagnosticTarget {
        DiagnosticTarget::new(DiagnosticTargetDomain::new("app-host.compute_profile"))
            .with_path(path.into())
    }

    /// Diagnostic emitted when a backend provider enumerates a
    /// device label that is not recognized by the inference profile
    /// vocabulary.
    pub fn invalid_candle_device(label: &str) -> Diagnostic {
        let id = format!("inference-profile-invalid-device-{label}");
        Diagnostic::new(
            DiagnosticId::new(id),
            DiagnosticCode::new("INFERENCE_PROFILE/INVALID_DEVICE"),
            DiagnosticSeverity::Warning,
            source(),
            format!("device label `{label}` is not recognized by the inference profile vocabulary"),
            target(format!("candle.device/{label}")),
        )
    }

    /// Diagnostic emitted when a backend provider recognizes a
    /// device label but cannot construct it on the current host
    /// (e.g. `candle:metal` requested on Linux).
    pub fn candle_device_unavailable(label: &str, reason: &str) -> Diagnostic {
        let id = format!("inference-profile-device-unavailable-{label}");
        Diagnostic::new(
            DiagnosticId::new(id),
            DiagnosticCode::new("INFERENCE_PROFILE/DEVICE_UNAVAILABLE"),
            DiagnosticSeverity::Error,
            source(),
            format!("candle device `{label}` is unavailable on this host: {reason}"),
            target(format!("candle.device/{label}")),
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn diag_in_target(message: &str) -> Diagnostic {
        Diagnostic::new(
            reimagine_core::model::DiagnosticId::new("d-1"),
            reimagine_core::diagnostic::DiagnosticCode::new("TEST/CODE"),
            reimagine_core::diagnostic::DiagnosticSeverity::Info,
            reimagine_core::diagnostic::DiagnosticSourceName::new("test"),
            message.to_string(),
            reimagine_core::diagnostic::DiagnosticTarget::new(
                reimagine_core::diagnostic::DiagnosticTargetDomain::new("test"),
            ),
        )
    }

    #[test]
    fn kind_from_label_recognizes_known_labels() {
        assert_eq!(kind_from_label("cpu"), DeviceKind::Cpu);
        assert_eq!(kind_from_label("CPU"), DeviceKind::Cpu);
        assert_eq!(kind_from_label("metal"), DeviceKind::Gpu);
        assert_eq!(kind_from_label("mps"), DeviceKind::Gpu);
        assert_eq!(kind_from_label("cuda"), DeviceKind::Gpu);
        assert_eq!(kind_from_label("cuda:0"), DeviceKind::Gpu);
        assert_eq!(kind_from_label("CUDA:1"), DeviceKind::Gpu);
        assert_eq!(kind_from_label("remote"), DeviceKind::Remote);
        assert_eq!(kind_from_label("remote:foo"), DeviceKind::Remote);
        assert_eq!(kind_from_label(""), DeviceKind::Unknown);
        assert_eq!(kind_from_label("tpu"), DeviceKind::Unknown);
        assert_eq!(kind_from_label("npu:0"), DeviceKind::Unknown);
    }

    #[test]
    fn device_profile_new_derives_kind_from_label() {
        let cpu = DeviceProfile::new("cpu");
        assert_eq!(cpu.label, "cpu");
        assert_eq!(cpu.kind, DeviceKind::Cpu);
        assert!(cpu.name.is_none());
        assert!(cpu.ordinal.is_none());
        assert!(cpu.memory.is_none());
        assert!(cpu.supported_dtypes.is_empty());

        let metal = DeviceProfile::new("metal");
        assert_eq!(metal.kind, DeviceKind::Gpu);

        let cuda0 = DeviceProfile::new("cuda:0");
        assert_eq!(cuda0.kind, DeviceKind::Gpu);

        let remote = DeviceProfile::new("remote:foo");
        assert_eq!(remote.kind, DeviceKind::Remote);

        let unknown = DeviceProfile::new("tpu");
        assert_eq!(unknown.kind, DeviceKind::Unknown);
    }

    #[test]
    fn device_profile_with_kind_overrides_derived_kind() {
        let p = DeviceProfile::new("cpu").with_kind(DeviceKind::Gpu);
        assert_eq!(p.kind, DeviceKind::Gpu);
        assert_eq!(p.label, "cpu");
    }

    #[test]
    fn device_profile_with_helpers_populate_optional_fields() {
        let memory = MemoryProfile::new().with_total(16 * 1024 * 1024 * 1024);
        let p = DeviceProfile::new("cuda:0")
            .with_name("NVIDIA RTX 4090")
            .with_ordinal(0)
            .with_memory(memory.clone())
            .with_supported_dtype("f32")
            .with_supported_dtype("f16");
        assert_eq!(p.name.as_deref(), Some("NVIDIA RTX 4090"));
        assert_eq!(p.ordinal, Some(0));
        assert_eq!(p.memory, Some(memory));
        assert_eq!(p.supported_dtypes.len(), 2);
        assert_eq!(p.supported_dtypes[0].as_str(), "f32");
        assert_eq!(p.supported_dtypes[1].as_str(), "f16");
    }

    #[test]
    fn memory_profile_builder_methods_populate_fields() {
        let m = MemoryProfile::with_total_and_free(1024, 512);
        assert_eq!(m.total_bytes, Some(1024));
        assert_eq!(m.free_bytes, Some(512));

        let m = MemoryProfile::new().with_total(2048).with_free(1024);
        assert_eq!(m.total_bytes, Some(2048));
        assert_eq!(m.free_bytes, Some(1024));
    }

    #[test]
    fn dtype_profile_new_and_from_conversions() {
        let p = DTypeProfile::new("f32");
        assert_eq!(p.as_str(), "f32");

        let from_str: DTypeProfile = "bf16".into();
        assert_eq!(from_str.as_str(), "bf16");

        let from_string: DTypeProfile = String::from("u8").into();
        assert_eq!(from_string.as_str(), "u8");
    }

    #[test]
    fn workspace_compute_profile_serde_roundtrip() {
        let backend_profile = BackendProfile::new(Backend::new("candle"))
            .with_instance(
                BackendInstanceProfile::new(
                    BackendInstance::new("candle:cpu"),
                    Backend::new("candle"),
                    DeviceProfile::new("cpu"),
                    BackendInstanceStatus::Available,
                )
                .with_capability(InferenceCapability::DiffusionSample),
            )
            .with_diagnostic(diag_in_target("probe ok"));
        let original = WorkspaceComputeProfile::new()
            .with_backend_profile(backend_profile)
            .with_diagnostic(diag_in_target("top-level"));

        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: WorkspaceComputeProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
    }

    #[test]
    fn backend_profile_serde_roundtrip() {
        let plugin = Plugin::try_from("builtin.candle").unwrap();
        let extension = Extension::try_from("backend.candle").unwrap();
        let original = BackendProfile::new(Backend::new("candle"))
            .with_plugin(plugin, extension)
            .with_instance(BackendInstanceProfile::new(
                BackendInstance::new("candle:metal"),
                Backend::new("candle"),
                DeviceProfile::new("metal"),
                BackendInstanceStatus::Unavailable,
            ))
            .with_diagnostic(diag_in_target("metal unavailable"));

        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: BackendProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
    }

    #[test]
    fn backend_instance_profile_serde_roundtrip() {
        let memory = MemoryProfile::with_total_and_free(1024, 512);
        let device = DeviceProfile::new("cuda:0")
            .with_name("GPU 0")
            .with_ordinal(0)
            .with_memory(memory)
            .with_supported_dtype("f16");
        let original = BackendInstanceProfile::new(
            BackendInstance::new("candle:cuda"),
            Backend::new("candle"),
            device,
            BackendInstanceStatus::Available,
        )
        .with_capability(InferenceCapability::LoadBundle)
        .with_capability(InferenceCapability::DiffusionSample)
        .with_diagnostic(diag_in_target("ok"));

        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: BackendInstanceProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
    }

    #[test]
    fn device_profile_serde_roundtrip() {
        let original = DeviceProfile::new("cuda:0")
            .with_name("NVIDIA")
            .with_ordinal(0)
            .with_memory(MemoryProfile::with_total_and_free(2048, 1024))
            .with_supported_dtype("f32")
            .with_supported_dtype("bf16");

        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: DeviceProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
    }

    #[test]
    fn memory_profile_serde_roundtrip() {
        let original = MemoryProfile::with_total_and_free(2048, 1024);
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: MemoryProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);

        let empty = MemoryProfile::new();
        let json = serde_json::to_string(&empty).expect("serialize empty");
        let parsed: MemoryProfile = serde_json::from_str(&json).expect("deserialize empty");
        assert_eq!(parsed, empty);
    }

    #[test]
    fn dtype_profile_serde_roundtrip() {
        let original = DTypeProfile::new("f32");
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: DTypeProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
    }

    #[test]
    fn device_kind_serde_roundtrip_uses_pascal_case() {
        for (variant, expected) in [
            (DeviceKind::Cpu, "\"Cpu\""),
            (DeviceKind::Gpu, "\"Gpu\""),
            (DeviceKind::Remote, "\"Remote\""),
            (DeviceKind::Unknown, "\"Unknown\""),
        ] {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(
                json, expected,
                "kind {variant:?} should serialize as {expected}"
            );
            let parsed: DeviceKind = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn backend_instance_status_serde_roundtrip_uses_pascal_case() {
        for (variant, expected) in [
            (BackendInstanceStatus::Available, "\"Available\""),
            (BackendInstanceStatus::Unavailable, "\"Unavailable\""),
        ] {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(json, expected);
            let parsed: BackendInstanceStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn backend_profile_provider_trait_is_object_safe() {
        // Compile-time check: the trait can be used as a trait object
        // and dispatched through `&dyn BackendProfileProvider`.
        fn _accepts(_provider: &dyn BackendProfileProvider) {}

        struct Stub;

        #[async_trait]
        impl BackendProfileProvider for Stub {
            async fn backend_profile(&self) -> BackendProfile {
                BackendProfile::new(Backend::new("stub"))
            }
        }

        let stub = Stub;
        let _provider: Arc<dyn BackendProfileProvider> = Arc::new(stub);
        let _ = _accepts;
    }

    #[test]
    fn invalid_candle_device_diagnostic_targets_compute_profile() {
        let d = diagnostics::invalid_candle_device("tpu");
        assert_eq!(d.code().as_str(), "INFERENCE_PROFILE/INVALID_DEVICE");
        assert_eq!(d.source().as_str(), "inference");
        assert_eq!(
            d.primary().domain().as_str(),
            "app-host.compute_profile",
            "diagnostic should surface at app-host compute_profile"
        );
        assert!(d.primary().path().unwrap_or("").contains("tpu"));
        assert!(d.message().contains("tpu"));
    }

    #[test]
    fn candle_device_unavailable_diagnostic_carries_reason() {
        let d = diagnostics::candle_device_unavailable("metal", "no metal runtime");
        assert_eq!(d.code().as_str(), "INFERENCE_PROFILE/DEVICE_UNAVAILABLE");
        assert_eq!(d.source().as_str(), "inference");
        assert_eq!(
            d.primary().domain().as_str(),
            "app-host.compute_profile",
            "diagnostic should surface at app-host compute_profile"
        );
        assert!(d.message().contains("metal"));
        assert!(d.message().contains("no metal runtime"));
    }
}
