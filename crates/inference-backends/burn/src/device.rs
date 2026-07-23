//! `BurnDevice` enum — feature-gated multi-variant.
//!
//! ## Architecture
//!
//! With burn/13, the Burn backend advertises a
//! `burn:<variant>` backend instance per device variant. The set
//! of variants depends on the active Cargo feature:
//!
//! | Feature  | Variant         | Label        |
//! |----------|-----------------|--------------|
//! | `wgpu`   | `Wgpu(_)`       | `wgpu:default`, `wgpu:metal`, `wgpu:vulkan` |
//! | `flex`   | `Flex`          | `flex:cpu`   |
//!
//! The `wgpu` and `flex` features are mutually exclusive at the
//! crate level; only one can be the live production compute backend.

use crate::error::BurnBackendError;

/// Feature-gated device variant.
///
/// Carries only the active production backend variant for the current
/// feature matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnDevice {
    /// GPU backend via burn-wgpu
    /// (`burn_wgpu::WgpuDevice` — Metal/Vulkan/CubeCL CPU).
    #[cfg(feature = "wgpu")]
    Wgpu(burn_wgpu::WgpuDevice),
    /// CPU backend via burn-flex (SIMD + rayon).
    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    Flex,
}

impl BurnDevice {
    /// Construct a [`BurnDevice`] from a textual label.
    ///
    /// `new` is **infallible**: unknown labels fall back to
    /// the active feature's default device so configuration stays valid
    /// even when callers pass an unrecognized device. The
    /// validating resolver is [`try_build_device`](Self::try_build_device).
    ///
    /// Under each feature, `new` maps labels to the canonical
    /// variant for the active feature:
    ///
    /// - Under `wgpu` (default): `"default"` -> `Wgpu(DefaultDevice)`,
    ///   `"metal"` -> `Wgpu(IntegratedGpu(0))`,
    ///   `"vulkan"` -> `Wgpu(DiscreteGpu(0))`.
    /// - Under `flex`: `"cpu"` -> `Flex`, `"flex:cpu"` -> `Flex`.
    pub fn new(label: impl Into<String>) -> Self {
        match label.into().as_str() {
            #[cfg(feature = "wgpu")]
            "default" | "wgpu:default" => Self::Wgpu(burn_wgpu::WgpuDevice::DefaultDevice),
            #[cfg(feature = "wgpu")]
            "metal" => Self::Wgpu(burn_wgpu::WgpuDevice::IntegratedGpu(0)),
            #[cfg(feature = "wgpu")]
            "vulkan" => Self::Wgpu(burn_wgpu::WgpuDevice::DiscreteGpu(0)),
            #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
            "cpu" | "flex:cpu" => Self::Flex,
            _ => Self::active_cpu(),
        }
    }

    /// Canonical short label for this device variant.
    ///
    /// This is the suffix the profile uses after the `burn:`
    /// backend prefix to construct the
    /// [`BackendInstance`](reimagine_inference::BackendInstance)
    /// label (e.g., label `"wgpu:metal"` -> instance
    /// `"burn:wgpu:metal"`).
    pub fn label(&self) -> &str {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => wgpu_device_label(device),
            #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
            Self::Flex => "flex:cpu",
        }
    }

    /// Resolve a textual device label into a concrete
    /// [`BurnDevice`] variant for the active feature.
    ///
    /// Returns an error for unknown labels (skeleton parity
    /// with the V1 `try_build_device` API).
    #[cfg(feature = "wgpu")]
    pub fn try_build_device(label: &str) -> Result<Self, BurnBackendError> {
        match label {
            "default" | "wgpu:default" => Ok(Self::Wgpu(burn_wgpu::WgpuDevice::DefaultDevice)),
            "metal" => Ok(Self::Wgpu(burn_wgpu::WgpuDevice::IntegratedGpu(0))),
            "vulkan" => Ok(Self::Wgpu(burn_wgpu::WgpuDevice::DiscreteGpu(0))),
            other => Err(BurnBackendError::DeviceUnavailable {
                requested: other.to_owned(),
                reason: format!(
                    "unknown Burn device label `{other}`; supported under wgpu feature: default, metal, vulkan"
                ),
            }),
        }
    }

    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    pub fn try_build_device(label: &str) -> Result<Self, BurnBackendError> {
        match label {
            "cpu" | "flex:cpu" => Ok(Self::Flex),
            other => Err(BurnBackendError::DeviceUnavailable {
                requested: other.to_owned(),
                reason: format!(
                    "unknown Burn device label `{other}`; supported under flex feature: cpu"
                ),
            }),
        }
    }

    #[cfg(feature = "wgpu")]
    fn active_cpu() -> Self {
        Self::Wgpu(burn_wgpu::WgpuDevice::DefaultDevice)
    }

    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    fn active_cpu() -> Self {
        Self::Flex
    }

    pub fn default_device() -> Self {
        Self::active_cpu()
    }
}

/// Canonical short label for a [`burn_wgpu::WgpuDevice`].
///
/// The mapping is the V1 stub defined in the burn/13 issue
/// spec (D3): adapter-kind -> `wgpu:{kind}`. Real adapter
/// probing at runtime lives in a later issue; here we
/// enumerate the static variants.
#[cfg(feature = "wgpu")]
fn wgpu_device_label(device: &burn_wgpu::WgpuDevice) -> &'static str {
    use burn_wgpu::WgpuDevice;
    match device {
        WgpuDevice::IntegratedGpu(_) => "wgpu:metal",
        WgpuDevice::DiscreteGpu(_) => "wgpu:vulkan",
        WgpuDevice::Cpu => "wgpu:cpu",
        WgpuDevice::VirtualGpu(_) => "wgpu:virtual",
        WgpuDevice::DefaultDevice => "wgpu:default",
        // `BestAvailable` is deprecated upstream; we route it
        // through `DefaultDevice` for label purposes.
        #[allow(deprecated)]
        WgpuDevice::BestAvailable => "wgpu:default",
        WgpuDevice::Existing(_) => "wgpu:existing",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_default_label_maps_to_feature_default_variant() {
        let device = BurnDevice::default_device();
        // The label returned here must match the backend
        // instance suffix used by `BurnBackend::backend_instance`.
        #[cfg(feature = "wgpu")]
        assert_eq!(device.label(), "wgpu:default");
        #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
        assert_eq!(device.label(), "flex:cpu");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_label_mappings_match_issue_spec() {
        assert_eq!(BurnDevice::new("metal").label(), "wgpu:metal");
        assert_eq!(BurnDevice::new("vulkan").label(), "wgpu:vulkan");
        assert_eq!(BurnDevice::new("default").label(), "wgpu:default");
    }

    #[test]
    fn try_build_device_accepts_known_label_and_rejects_unknown() {
        #[cfg(feature = "wgpu")]
        assert!(BurnDevice::try_build_device("default").is_ok());
        #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
        assert!(BurnDevice::try_build_device("cpu").is_ok());

        let err = BurnDevice::try_build_device("gpu").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("gpu"), "msg: {msg}");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn try_build_device_accepts_metal_and_vulkan_labels() {
        assert!(BurnDevice::try_build_device("metal").is_ok());
        assert!(BurnDevice::try_build_device("vulkan").is_ok());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn try_build_device_rejects_cpu_label_under_wgpu() {
        let err = BurnDevice::try_build_device("cpu").unwrap_err();
        assert!(err.to_string().contains("cpu"));
    }

    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    #[test]
    fn try_build_device_rejects_metal_and_vulkan_labels_under_flex() {
        let err = BurnDevice::try_build_device("metal").unwrap_err();
        assert!(err.to_string().contains("metal"));
        let err = BurnDevice::try_build_device("vulkan").unwrap_err();
        assert!(err.to_string().contains("vulkan"));
    }
}
