//! Compute profile DTOs.
//!
//! Host-neutral projection of
//! [`reimagine_inference::WorkspaceComputeProfile`] for HTTP and IPC
//! adapters. The projection drops every backend-native handle
//! (Candle `Device`, tensors, loaded model structs, etc.) and replaces
//! inference-internal enums with stable string forms so wire clients
//! do not need to know the inference-internal vocabulary.

use reimagine_inference::{
    BackendInstanceProfile, BackendInstanceStatus, BackendProfile, DTypeProfile, DeviceKind,
    DeviceProfile, MemoryProfile, OperationOptionsProfile, OperationOptionsProfileKind,
    SamplerOptionProfile, SamplerSchedulerPairProfile, SchedulerOptionProfile,
    WorkspaceComputeProfile,
};
use serde::{Deserialize, Serialize};

use super::runs::DiagnosticDto;

/// `GET /compute-profile` response. V1 returns the
/// [`ComputeProfileDto`] projection of the workspace's most recent
/// [`WorkspaceComputeProfile`] snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeProfileDto {
    /// Per-backend profiles, one per registered backend provider.
    pub backend_profiles: Vec<BackendProfileDto>,
    /// Top-level diagnostics that do not belong to a single backend.
    pub diagnostics: Vec<DiagnosticDto>,
}

impl From<WorkspaceComputeProfile> for ComputeProfileDto {
    fn from(value: WorkspaceComputeProfile) -> Self {
        Self {
            backend_profiles: value
                .backend_profiles
                .into_iter()
                .map(BackendProfileDto::from)
                .collect(),
            diagnostics: value.diagnostics.into_iter().map(Into::into).collect(),
        }
    }
}

/// Per-backend profile contributed by one
/// [`reimagine_inference::BackendProfileProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendProfileDto {
    /// Stable backend implementation label (e.g. `"candle"`).
    pub backend: String,
    /// Optional plugin that contributed this backend.
    pub plugin: Option<String>,
    /// Optional extension identity within the contributing plugin.
    pub extension: Option<String>,
    /// Backend instance candidates the provider can construct on this
    /// host. Order is provider-defined.
    pub instances: Vec<BackendInstanceProfileDto>,
    /// Diagnostics emitted by this provider during profile
    /// construction.
    pub diagnostics: Vec<DiagnosticDto>,
}

impl From<BackendProfile> for BackendProfileDto {
    fn from(value: BackendProfile) -> Self {
        Self {
            backend: value.backend.as_str().to_string(),
            plugin: value.plugin.as_ref().map(|p| p.as_str().to_string()),
            extension: value.extension.as_ref().map(|e| e.as_str().to_string()),
            instances: value.instances.into_iter().map(Into::into).collect(),
            diagnostics: value.diagnostics.into_iter().map(Into::into).collect(),
        }
    }
}

/// Profile for a single backend instance candidate
/// (e.g. `"candle:metal"`, `"candle:cpu"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendInstanceProfileDto {
    /// Stable backend instance identity (e.g. `"candle:cpu"`).
    pub instance: String,
    /// Open backend implementation label. Mirrors the enclosing
    /// [`BackendProfileDto::backend`] but is included here so each
    /// instance profile is self-contained.
    pub backend: String,
    /// Device descriptor for this instance.
    pub device: DeviceProfileDto,
    /// Capability identifiers (e.g. `"model.load_bundle"`,
    /// `"diffusion.sample"`) this backend instance advertises as
    /// supported on its device. Rendered as stable string forms; see
    /// [`reimagine_inference::InferenceCapability::as_str`].
    pub capabilities: Vec<String>,
    /// Operation-specific backend-supported option lists.
    pub operation_options: Vec<OperationOptionsProfileDto>,
    /// Whether this instance is currently usable on the host. Stable
    /// wire strings: `"Available"` / `"Unavailable"`.
    pub status: String,
    /// Diagnostics emitted while probing this instance.
    pub diagnostics: Vec<DiagnosticDto>,
}

impl From<BackendInstanceProfile> for BackendInstanceProfileDto {
    fn from(value: BackendInstanceProfile) -> Self {
        Self {
            instance: value.instance.as_str().to_string(),
            backend: value.backend.as_str().to_string(),
            device: DeviceProfileDto::from(value.device),
            capabilities: value
                .capabilities
                .into_iter()
                .map(|c| c.as_str().to_string())
                .collect(),
            operation_options: value
                .operation_options
                .into_iter()
                .map(OperationOptionsProfileDto::from)
                .collect(),
            status: status_label(value.status).to_string(),
            diagnostics: value.diagnostics.into_iter().map(Into::into).collect(),
        }
    }
}

/// Operation-specific options projected to wire-stable strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationOptionsProfileDto {
    pub capability: String,
    pub options: OperationOptionsProfileKindDto,
}

impl From<OperationOptionsProfile> for OperationOptionsProfileDto {
    fn from(value: OperationOptionsProfile) -> Self {
        Self {
            capability: value.capability.as_str().to_string(),
            options: OperationOptionsProfileKindDto::from(value.options),
        }
    }
}

/// Capability-specific operation options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OperationOptionsProfileKindDto {
    DiffusionSample {
        samplers: Vec<SamplerOptionProfileDto>,
        schedulers: Vec<SchedulerOptionProfileDto>,
        supported_pairs: Vec<SamplerSchedulerPairProfileDto>,
    },
}

impl From<OperationOptionsProfileKind> for OperationOptionsProfileKindDto {
    fn from(value: OperationOptionsProfileKind) -> Self {
        match value {
            OperationOptionsProfileKind::DiffusionSample {
                samplers,
                schedulers,
                supported_pairs,
            } => Self::DiffusionSample {
                samplers: samplers.into_iter().map(Into::into).collect(),
                schedulers: schedulers.into_iter().map(Into::into).collect(),
                supported_pairs: supported_pairs.into_iter().map(Into::into).collect(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplerOptionProfileDto {
    pub name: String,
}

impl From<SamplerOptionProfile> for SamplerOptionProfileDto {
    fn from(value: SamplerOptionProfile) -> Self {
        Self {
            name: value.name.as_str().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerOptionProfileDto {
    pub name: String,
}

impl From<SchedulerOptionProfile> for SchedulerOptionProfileDto {
    fn from(value: SchedulerOptionProfile) -> Self {
        Self {
            name: value.name.as_str().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplerSchedulerPairProfileDto {
    pub sampler: String,
    pub scheduler: String,
}

impl From<SamplerSchedulerPairProfile> for SamplerSchedulerPairProfileDto {
    fn from(value: SamplerSchedulerPairProfile) -> Self {
        Self {
            sampler: value.sampler.as_str().to_string(),
            scheduler: value.scheduler.as_str().to_string(),
        }
    }
}

/// Host-neutral descriptor of a backend's device, projected for wire
/// transport. All structured fields except `label` and `kind` are
/// optional so existing callers that only set a label still round
/// trip cleanly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProfileDto {
    /// Original opaque label (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub label: String,
    /// Coarse device kind. Stable wire strings: `"Cpu"`, `"Gpu"`,
    /// `"Remote"`, `"Unknown"`.
    pub kind: String,
    /// Optional human-readable device name (e.g. `"Apple M2 Pro"`).
    pub name: Option<String>,
    /// Optional device ordinal for multi-device backends
    /// (e.g. `0` for `"cuda:0"`).
    pub ordinal: Option<u32>,
    /// Optional memory summary for the device.
    pub memory: Option<MemoryProfileDto>,
    /// Dtype identifiers the backend reports as supported on this
    /// device. Empty when the backend did not probe dtypes.
    pub supported_dtypes: Vec<DTypeProfileDto>,
}

impl From<DeviceProfile> for DeviceProfileDto {
    fn from(value: DeviceProfile) -> Self {
        Self {
            label: value.label,
            kind: kind_label(value.kind).to_string(),
            name: value.name,
            ordinal: value.ordinal,
            memory: value.memory.map(MemoryProfileDto::from),
            supported_dtypes: value
                .supported_dtypes
                .into_iter()
                .map(DTypeProfileDto::from)
                .collect(),
        }
    }
}

/// Optional memory summary attached to a [`DeviceProfileDto`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProfileDto {
    /// Total bytes available on the device, when known.
    pub total_bytes: Option<u64>,
    /// Bytes currently free on the device, when known.
    pub free_bytes: Option<u64>,
}

impl From<MemoryProfile> for MemoryProfileDto {
    fn from(value: MemoryProfile) -> Self {
        Self {
            total_bytes: value.total_bytes,
            free_bytes: value.free_bytes,
        }
    }
}

/// Host-neutral dtype string attached to a [`DeviceProfileDto`] as a
/// supported dtype.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DTypeProfileDto {
    /// Opaque dtype identifier (e.g. `"f32"`).
    pub dtype: String,
}

impl From<DTypeProfile> for DTypeProfileDto {
    fn from(value: DTypeProfile) -> Self {
        Self { dtype: value.dtype }
    }
}

fn kind_label(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Cpu => "Cpu",
        DeviceKind::Gpu => "Gpu",
        DeviceKind::Remote => "Remote",
        DeviceKind::Unknown => "Unknown",
    }
}

fn status_label(status: BackendInstanceStatus) -> &'static str {
    match status {
        BackendInstanceStatus::Available => "Available",
        BackendInstanceStatus::Unavailable => "Unavailable",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_inference::{
        Backend, BackendInstance, BackendInstanceProfile, BackendInstanceStatus, BackendProfile,
        DTypeProfile, DeviceKind, DeviceProfile, MemoryProfile, OperationOptionsProfile,
        SamplerName, SamplerOptionProfile, SamplerSchedulerPairProfile, SchedulerName,
        SchedulerOptionProfile, WorkspaceComputeProfile,
    };
    use reimagine_plugin::{Extension, Plugin};

    fn sample_profile() -> WorkspaceComputeProfile {
        let device = DeviceProfile::new("cpu")
            .with_kind(DeviceKind::Cpu)
            .with_name("cpu-host")
            .with_ordinal(0)
            .with_memory(
                MemoryProfile::new()
                    .with_total(8 * 1024 * 1024 * 1024)
                    .with_free(4 * 1024 * 1024 * 1024),
            )
            .with_supported_dtypes(vec![DTypeProfile::from("f32"), DTypeProfile::from("f16")]);

        let instance = BackendInstanceProfile::new(
            BackendInstance::new("candle:cpu"),
            Backend::new("candle"),
            device,
            BackendInstanceStatus::Available,
        );

        let backend = BackendProfile::new(Backend::new("candle"))
            .with_plugin(
                Plugin::try_from("builtin.candle").expect("valid plugin label"),
                Extension::try_from("backend.candle").expect("valid extension label"),
            )
            .with_instance(instance);

        WorkspaceComputeProfile::new().with_backend_profile(backend)
    }

    #[test]
    fn dto_projection_carries_host_neutral_fields() {
        let dto: ComputeProfileDto = sample_profile().into();
        assert_eq!(dto.backend_profiles.len(), 1);
        let backend = &dto.backend_profiles[0];
        assert_eq!(backend.backend, "candle");
        assert_eq!(backend.plugin.as_deref(), Some("builtin.candle"));
        assert_eq!(backend.extension.as_deref(), Some("backend.candle"));
        assert_eq!(backend.instances.len(), 1);

        let instance = &backend.instances[0];
        assert_eq!(instance.instance, "candle:cpu");
        assert_eq!(instance.backend, "candle");
        assert_eq!(instance.status, "Available");
        assert_eq!(instance.capabilities, Vec::<String>::new());
        assert!(instance.diagnostics.is_empty());

        assert_eq!(instance.device.label, "cpu");
        assert_eq!(instance.device.kind, "Cpu");
        assert_eq!(instance.device.name.as_deref(), Some("cpu-host"));
        assert_eq!(instance.device.ordinal, Some(0));

        let memory = instance
            .device
            .memory
            .as_ref()
            .expect("memory projected through");
        assert_eq!(memory.total_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(memory.free_bytes, Some(4 * 1024 * 1024 * 1024));

        let dtypes: Vec<&str> = instance
            .device
            .supported_dtypes
            .iter()
            .map(|d| d.dtype.as_str())
            .collect();
        assert_eq!(dtypes, vec!["f32", "f16"]);
    }

    #[test]
    fn dto_projection_carries_diffusion_operation_options_as_strings() {
        let instance = BackendInstanceProfile::new(
            BackendInstance::new("candle:cpu"),
            Backend::new("candle"),
            DeviceProfile::new("cpu"),
            BackendInstanceStatus::Available,
        )
        .with_operation_options(OperationOptionsProfile::diffusion_sample(
            vec![SamplerOptionProfile::new(SamplerName::Euler)],
            vec![SchedulerOptionProfile::new(SchedulerName::Normal)],
            vec![SamplerSchedulerPairProfile::new(
                SamplerName::Euler,
                SchedulerName::Normal,
            )],
        ));
        let profile = WorkspaceComputeProfile::new().with_backend_profile(
            BackendProfile::new(Backend::new("candle")).with_instance(instance),
        );

        let dto: ComputeProfileDto = profile.into();
        let options = &dto.backend_profiles[0].instances[0].operation_options[0];
        assert_eq!(options.capability, "diffusion.sample");
        let OperationOptionsProfileKindDto::DiffusionSample {
            samplers,
            schedulers,
            supported_pairs,
        } = &options.options;
        assert_eq!(samplers[0].name, "euler");
        assert_eq!(schedulers[0].name, "normal");
        assert_eq!(supported_pairs[0].sampler, "euler");
        assert_eq!(supported_pairs[0].scheduler, "normal");
    }

    #[test]
    fn dto_projection_does_not_expose_backend_internals() {
        let dto: ComputeProfileDto = sample_profile().into();
        let json = serde_json::to_value(&dto).expect("serialize dto");
        // Verify wire JSON never carries backend-internal types.
        let serialized = serde_json::to_string(&dto).expect("serialize dto to string");
        assert!(
            !serialized.contains("candle_core"),
            "wire form must not leak backend crate names: {serialized}"
        );
        assert!(
            !serialized.contains("Device"),
            "wire form must not leak device handle: {serialized}"
        );
        // `kind` is a stable string, not the inference enum.
        let kind = json["backend_profiles"][0]["instances"][0]["device"]["kind"]
            .as_str()
            .expect("kind is a string");
        assert_eq!(kind, "Cpu");
        let status = json["backend_profiles"][0]["instances"][0]["status"]
            .as_str()
            .expect("status is a string");
        assert_eq!(status, "Available");
    }

    #[test]
    fn dto_roundtrips_through_serde() {
        let original: ComputeProfileDto = sample_profile().into();
        let serialized = serde_json::to_string(&original).expect("serialize");
        let restored: ComputeProfileDto =
            serde_json::from_str(&serialized).expect("roundtrip deserialize");
        assert_eq!(restored, original);
    }

    #[test]
    fn dto_maps_device_kind_strings() {
        let cpu = DeviceProfileDto::from(DeviceProfile::new("cpu").with_kind(DeviceKind::Cpu));
        assert_eq!(cpu.kind, "Cpu");
        let gpu = DeviceProfileDto::from(DeviceProfile::new("metal").with_kind(DeviceKind::Gpu));
        assert_eq!(gpu.kind, "Gpu");
        let remote =
            DeviceProfileDto::from(DeviceProfile::new("remote:foo").with_kind(DeviceKind::Remote));
        assert_eq!(remote.kind, "Remote");
        let unknown =
            DeviceProfileDto::from(DeviceProfile::new("tpu").with_kind(DeviceKind::Unknown));
        assert_eq!(unknown.kind, "Unknown");
    }

    #[test]
    fn dto_maps_status_strings() {
        let available = BackendInstanceProfileDto::from(BackendInstanceProfile::new(
            BackendInstance::new("candle:cpu"),
            Backend::new("candle"),
            DeviceProfile::new("cpu"),
            BackendInstanceStatus::Available,
        ));
        assert_eq!(available.status, "Available");
        let unavailable = BackendInstanceProfileDto::from(BackendInstanceProfile::new(
            BackendInstance::new("candle:tpu"),
            Backend::new("candle"),
            DeviceProfile::new("tpu"),
            BackendInstanceStatus::Unavailable,
        ));
        assert_eq!(unavailable.status, "Unavailable");
    }

    #[test]
    fn nested_dtos_roundtrip_independently() {
        let mem = MemoryProfileDto {
            total_bytes: Some(16),
            free_bytes: None,
        };
        let json = serde_json::to_string(&mem).unwrap();
        let restored: MemoryProfileDto = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, mem);

        let dtype = DTypeProfileDto {
            dtype: "bf16".to_string(),
        };
        let json = serde_json::to_string(&dtype).unwrap();
        let restored: DTypeProfileDto = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, dtype);
    }
}
