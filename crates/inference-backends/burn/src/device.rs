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
//! | always   | `NdarrayCpu`    | `cpu`        |
//! | `wgpu`   | `Wgpu(_)`       | `wgpu:metal`, `wgpu:vulkan`, `wgpu:cpu` |
//! | `flex`   | `Flex`          | `flex:cpu`   |
//!
//! The legacy `NdarrayCpu` variant is always present because
//! `burn-ndarray` remains the compile-time base layer used by
//! the V1 operations (`latent.create_empty`, `text.encode`
//! preflight). Real neural-network forward passes arriving in
//! burn/08f+, burn/10, and burn/11 will route through the
//! GPU/Flex backends.
//!
//! The `wgpu` and `flex` features are mutually exclusive at the
//! crate level because the Burn `Backend` trait has concrete
//! types per feature (`burn-wgpu::WgpuBackend` vs.
//! `burn-flex::FlexBackend`); only one can be the live compute
//! backend. The V1 ndarray path remains as a compile-time
//! compatibility shim.

use crate::error::BurnBackendError;

/// Feature-gated device variant.
///
/// Always carries the legacy `NdarrayCpu` variant because
/// `burn-ndarray` is the V1 compile-time base; the
/// `Wgpu`/`Flex` variants are added by their respective
/// features.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnDevice {
    /// Legacy burn-ndarray CPU backend.
    ///
    /// Always available — V1 operations execute tensors
    /// through `burn-ndarray` even under the `wgpu`/`flex`
    /// features. The wgpu/flex instances are advertised in
    /// the profile and are usable for real forward-pass
    /// execution once burn/08f+, burn/10, and burn/11 land.
    NdarrayCpu,
    /// GPU backend via burn-wgpu
    /// (`burn_wgpu::WgpuDevice` — Metal/Vulkan/CubeCL CPU).
    #[cfg(feature = "wgpu")]
    Wgpu(burn_wgpu::WgpuDevice),
    /// CPU backend via burn-flex (SIMD + rayon).
    #[cfg(feature = "flex")]
    Flex,
}

impl BurnDevice {
    /// Construct a [`BurnDevice`] from a textual label.
    ///
    /// `new` is **infallible**: unknown labels fall back to
    /// [`BurnDevice::NdarrayCpu`] so configuration stays valid
    /// even when callers pass an unrecognized device. The
    /// validating resolver is [`try_build_device`](Self::try_build_device).
    ///
    /// Under each feature, `new` maps labels to the canonical
    /// variant for the active feature:
    ///
    /// - Under `wgpu` (default): `"cpu"` -> `NdarrayCpu`,
    ///   `"metal"` -> `Wgpu(IntegratedGpu(0))`,
    ///   `"vulkan"` -> `Wgpu(DiscreteGpu(0))`.
    /// - Under `flex`: `"cpu"` -> `Flex`, `"flex:cpu"` -> `Flex`.
    /// - With neither: `"cpu"` -> `NdarrayCpu`.
    pub fn new(label: impl Into<String>) -> Self {
        match label.into().as_str() {
            #[cfg(feature = "wgpu")]
            "metal" => Self::Wgpu(burn_wgpu::WgpuDevice::IntegratedGpu(0)),
            #[cfg(feature = "wgpu")]
            "vulkan" => Self::Wgpu(burn_wgpu::WgpuDevice::DiscreteGpu(0)),
            #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
            "cpu" | "flex:cpu" => Self::Flex,
            // Default / fallback — captures the wgpu "cpu" path
            // (legacy ndarray) and any unknown label.
            _ => Self::NdarrayCpu,
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
            Self::NdarrayCpu => "cpu",
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => wgpu_device_label(device),
            #[cfg(feature = "flex")]
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
            "cpu" => Ok(Self::NdarrayCpu),
            "metal" => Ok(Self::Wgpu(burn_wgpu::WgpuDevice::IntegratedGpu(0))),
            "vulkan" => Ok(Self::Wgpu(burn_wgpu::WgpuDevice::DiscreteGpu(0))),
            other => Err(BurnBackendError::DeviceUnavailable {
                requested: other.to_owned(),
                reason: format!(
                    "unknown Burn device label `{other}`; supported under wgpu feature: cpu, metal, vulkan"
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

    #[cfg(not(any(feature = "wgpu", feature = "flex")))]
    pub fn try_build_device(label: &str) -> Result<Self, BurnBackendError> {
        match label {
            "cpu" => Ok(Self::NdarrayCpu),
            other => Err(BurnBackendError::DeviceUnavailable {
                requested: other.to_owned(),
                reason: format!(
                    "unknown Burn device label `{other}`; supported: cpu. \
                     Enable `wgpu` or `flex` feature for GPU/Flex backends."
                ),
            }),
        }
    }

    /// Concrete `burn-ndarray` CPU device used by the V1
    /// operations.
    ///
    /// `latent.create_empty` and the `text.encode` preflight
    /// allocate burn-ndarray tensors regardless of the active
    /// feature; the real GPU/Flex forward passes arrive in
    /// burn/08f+, burn/10, and burn/11. Exposing this helper
    /// lets the V1 operations stay on the legacy backend
    /// without changing the `BurnBackend::device` public type.
    pub fn ndarray_device(&self) -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
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
    fn new_cpu_label_maps_to_feature_default_variant() {
        let device = BurnDevice::new("cpu");
        // The label returned here must match the backend
        // instance suffix used by `BurnBackend::backend_instance`.
        #[cfg(feature = "wgpu")]
        assert_eq!(device.label(), "cpu");
        #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
        assert_eq!(device.label(), "flex:cpu");
        #[cfg(not(any(feature = "wgpu", feature = "flex")))]
        assert_eq!(device.label(), "cpu");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_label_mappings_match_issue_spec() {
        assert_eq!(BurnDevice::new("metal").label(), "wgpu:metal");
        assert_eq!(BurnDevice::new("vulkan").label(), "wgpu:vulkan");
    }

    #[test]
    fn try_build_device_accepts_known_label_and_rejects_unknown() {
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

    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    #[test]
    fn try_build_device_rejects_metal_and_vulkan_labels_under_flex() {
        let err = BurnDevice::try_build_device("metal").unwrap_err();
        assert!(err.to_string().contains("metal"));
        let err = BurnDevice::try_build_device("vulkan").unwrap_err();
        assert!(err.to_string().contains("vulkan"));
    }
}
