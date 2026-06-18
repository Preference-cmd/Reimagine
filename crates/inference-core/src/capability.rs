//! Capability identity and capability report.
//!
//! [`InferenceCapability`] is the closed V1 capability identity used
//! for diagnostics, capability reports, tracing, and bridge policy
//! context. It is **not** the runtime/backend dispatch key: the typed
//! capability method itself is the dispatch. The router never matches
//! on [`InferenceCapability`] to choose a typed method.
//!
//! [`InferenceBackendCapabilities`] and [`InferenceCapabilitySupport`]
//! describe which capabilities a backend advertises and which model
//! series / variant / role constraints each capability carries.

use reimagine_core::BackendKind;
use reimagine_core::model::{ModelRole, ModelSeries, ModelVariant};

// ── InferenceCapability ────────────────────────────────────────────

/// Closed V1 capability identity.
///
/// V1 capabilities map 1:1 to the typed method surface of
/// [`crate::backend::InferenceBackend`] and
/// [`crate::runtime::InferenceRuntime`]. The router dispatches by
/// calling the typed method — never by matching on
/// [`InferenceCapability`].
///
/// This enum is reserved for:
/// - capability reports ([`InferenceBackendCapabilities`]);
/// - diagnostic labels ([`crate::diagnostic`]);
/// - bridge policy context ([`crate::bridge`]);
/// - tracing spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceCapability {
    LoadBundle,
    TextEncode,
    CreateEmptyLatent,
    DiffusionSample,
    LatentDecode,
    ImageSave,
    ImagePreview,
}

impl InferenceCapability {
    /// Stable, dot-separated string label for diagnostics, tracing, and
    /// capability reports. The label is the canonical human-readable
    /// form but must not be used as a dispatch key.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LoadBundle => "model.load_bundle",
            Self::TextEncode => "text.encode",
            Self::CreateEmptyLatent => "latent.create_empty",
            Self::DiffusionSample => "diffusion.sample",
            Self::LatentDecode => "latent.decode",
            Self::ImageSave => "image.save",
            Self::ImagePreview => "image.preview",
        }
    }

    /// All V1 capabilities in a fixed order.
    pub fn all_v1() -> &'static [InferenceCapability] {
        &[
            Self::LoadBundle,
            Self::TextEncode,
            Self::CreateEmptyLatent,
            Self::DiffusionSample,
            Self::LatentDecode,
            Self::ImageSave,
            Self::ImagePreview,
        ]
    }
}

impl std::fmt::Display for InferenceCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── InferenceBackendCapabilities ────────────────────────────────────

/// Describes which capabilities a backend supports and which model
/// families / variants / roles those capabilities apply to.
#[derive(Debug, Clone)]
pub struct InferenceBackendCapabilities {
    backend_kind: BackendKind,
    capabilities: Vec<InferenceCapabilitySupport>,
}

impl InferenceBackendCapabilities {
    pub fn new(backend_kind: BackendKind) -> Self {
        Self {
            backend_kind,
            capabilities: Vec::new(),
        }
    }

    pub fn with_support(mut self, support: InferenceCapabilitySupport) -> Self {
        self.capabilities.push(support);
        self
    }

    pub fn backend_kind(&self) -> &BackendKind {
        &self.backend_kind
    }

    pub fn capability_supports(&self) -> &[InferenceCapabilitySupport] {
        &self.capabilities
    }

    /// Returns `true` when the backend claims to support the given
    /// capability regardless of model constraints.
    pub fn supports_capability(&self, capability: InferenceCapability) -> bool {
        self.capabilities
            .iter()
            .any(|support| support.capability == capability)
    }
}

// ── InferenceCapabilitySupport ──────────────────────────────────────

/// A single capability support entry. All fields except `capability`
/// are optional constraints; `None` means "all variants" or "all
/// series".
#[derive(Debug, Clone)]
pub struct InferenceCapabilitySupport {
    capability: InferenceCapability,
    model_series: Option<ModelSeries>,
    variant: Option<ModelVariant>,
    roles: Vec<ModelRole>,
}

impl InferenceCapabilitySupport {
    pub fn new(capability: InferenceCapability) -> Self {
        Self {
            capability,
            model_series: None,
            variant: None,
            roles: Vec::new(),
        }
    }

    pub fn with_model_series(mut self, series: ModelSeries) -> Self {
        self.model_series = Some(series);
        self
    }

    pub fn with_variant(mut self, variant: ModelVariant) -> Self {
        self.variant = Some(variant);
        self
    }

    pub fn with_role(mut self, role: ModelRole) -> Self {
        self.roles.push(role);
        self
    }

    pub fn capability(&self) -> InferenceCapability {
        self.capability
    }

    pub fn model_series(&self) -> Option<&ModelSeries> {
        self.model_series.as_ref()
    }

    pub fn variant(&self) -> Option<&ModelVariant> {
        self.variant.as_ref()
    }

    pub fn roles(&self) -> &[ModelRole] {
        &self.roles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_labels_are_stable_strings() {
        assert_eq!(
            InferenceCapability::LoadBundle.as_str(),
            "model.load_bundle"
        );
        assert_eq!(InferenceCapability::TextEncode.as_str(), "text.encode");
        assert_eq!(
            InferenceCapability::CreateEmptyLatent.as_str(),
            "latent.create_empty"
        );
        assert_eq!(
            InferenceCapability::DiffusionSample.as_str(),
            "diffusion.sample"
        );
        assert_eq!(InferenceCapability::LatentDecode.as_str(), "latent.decode");
        assert_eq!(InferenceCapability::ImageSave.as_str(), "image.save");
        assert_eq!(InferenceCapability::ImagePreview.as_str(), "image.preview");
    }

    #[test]
    fn all_v1_lists_seven_capabilities_in_order() {
        let caps = InferenceCapability::all_v1();
        assert_eq!(caps.len(), 7);
        assert_eq!(caps[0], InferenceCapability::LoadBundle);
        assert_eq!(caps[1], InferenceCapability::TextEncode);
        assert_eq!(caps[2], InferenceCapability::CreateEmptyLatent);
        assert_eq!(caps[3], InferenceCapability::DiffusionSample);
        assert_eq!(caps[4], InferenceCapability::LatentDecode);
        assert_eq!(caps[5], InferenceCapability::ImageSave);
        assert_eq!(caps[6], InferenceCapability::ImagePreview);
    }

    #[test]
    fn capabilities_advertise_support() {
        let caps = InferenceBackendCapabilities::new(BackendKind::new("candle"))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::DiffusionSample,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::TextEncode,
            ));
        assert!(caps.supports_capability(InferenceCapability::DiffusionSample));
        assert!(caps.supports_capability(InferenceCapability::TextEncode));
        assert!(!caps.supports_capability(InferenceCapability::ImageSave));
    }

    #[test]
    fn capability_support_carries_optional_constraints() {
        let support = InferenceCapabilitySupport::new(InferenceCapability::DiffusionSample)
            .with_model_series(ModelSeries::new("stable_diffusion"))
            .with_variant(ModelVariant::new("sdxl"))
            .with_role(ModelRole::CheckpointBundle);
        assert_eq!(support.model_series().unwrap().as_str(), "stable_diffusion");
        assert_eq!(support.variant().unwrap().as_str(), "sdxl");
        assert_eq!(support.roles().len(), 1);
    }
}
