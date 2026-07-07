use std::collections::BTreeMap;

use reimagine_inference::BackendInstance;

use crate::backend::BurnBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnPerformanceScenario {
    EmptyBackendEnvelope,
    TinyLatentPayloadBytes,
    ModelLoadBundleCacheReuse,
    TextEncodeModuleForward,
    FullTopologyMemoryReview,
}

impl BurnPerformanceScenario {
    pub fn key(self) -> &'static str {
        match self {
            Self::EmptyBackendEnvelope => "empty_backend_envelope",
            Self::TinyLatentPayloadBytes => "tiny_latent_payload_bytes",
            Self::ModelLoadBundleCacheReuse => "model_load_bundle_cache_reuse",
            Self::TextEncodeModuleForward => "text_encode_module_forward",
            Self::FullTopologyMemoryReview => "full_topology_memory_review",
        }
    }

    pub fn ci_safe(self) -> bool {
        match self {
            Self::EmptyBackendEnvelope
            | Self::TinyLatentPayloadBytes
            | Self::ModelLoadBundleCacheReuse => true,
            Self::TextEncodeModuleForward | Self::FullTopologyMemoryReview => false,
        }
    }

    pub fn requires_full_topology(self) -> bool {
        matches!(self, Self::FullTopologyMemoryReview)
    }
}

pub fn burn_performance_scenarios() -> &'static [BurnPerformanceScenario] {
    &[
        BurnPerformanceScenario::EmptyBackendEnvelope,
        BurnPerformanceScenario::TinyLatentPayloadBytes,
        BurnPerformanceScenario::ModelLoadBundleCacheReuse,
        BurnPerformanceScenario::TextEncodeModuleForward,
        BurnPerformanceScenario::FullTopologyMemoryReview,
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnPerformanceObservation {
    pub name: String,
    pub value: usize,
}

impl BurnPerformanceObservation {
    fn new(name: impl Into<String>, value: usize) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnPerformanceEnvelope {
    pub backend_instance: BackendInstance,
    pub device_label: String,
    pub observations: BTreeMap<String, usize>,
}

impl BurnPerformanceEnvelope {
    pub fn observation(&self, name: &str) -> Option<usize> {
        self.observations.get(name).copied()
    }
}

impl BurnBackend {
    pub fn performance_envelope(&self) -> BurnPerformanceEnvelope {
        let observations = [
            BurnPerformanceObservation::new("store_payloads", self.store().payload_count()),
            BurnPerformanceObservation::new(
                "store_bytes_approximate",
                self.store().payload_byte_size(),
            ),
            BurnPerformanceObservation::new(
                "model_cache_bundles",
                self.model_cache().bundle_count(),
            ),
            BurnPerformanceObservation::new(
                "text_encoder_cache_entries",
                self.text_encoder_cache().entry_count(),
            ),
        ]
        .into_iter()
        .map(|observation| (observation.name, observation.value))
        .collect();

        BurnPerformanceEnvelope {
            backend_instance: self.backend_instance(),
            device_label: self.device_label().to_owned(),
            observations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BurnPerformanceScenario, burn_performance_scenarios};

    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
    use reimagine_inference::{CreateEmptyLatentRequest, InferenceBackend};

    use crate::{BurnBackend, BurnBackendConfig};

    #[cfg(feature = "wgpu")]
    fn expected_instance() -> &'static str {
        "burn:wgpu:default"
    }

    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    fn expected_instance() -> &'static str {
        "burn:flex:cpu"
    }

    #[cfg(feature = "wgpu")]
    fn expected_device_label() -> &'static str {
        "wgpu:default"
    }

    #[cfg(all(not(feature = "wgpu"), feature = "flex"))]
    fn expected_device_label() -> &'static str {
        "flex:cpu"
    }

    fn backend() -> BurnBackend {
        BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("burn backend")
    }

    #[test]
    fn scenario_catalog_separates_ci_safe_probes_from_opt_in_work() {
        let scenarios = burn_performance_scenarios();

        assert!(scenarios.contains(&BurnPerformanceScenario::EmptyBackendEnvelope));
        assert!(scenarios.contains(&BurnPerformanceScenario::TinyLatentPayloadBytes));
        assert!(scenarios.contains(&BurnPerformanceScenario::ModelLoadBundleCacheReuse));
        assert!(scenarios.contains(&BurnPerformanceScenario::TextEncodeModuleForward));
        assert!(scenarios.contains(&BurnPerformanceScenario::FullTopologyMemoryReview));
        assert!(
            BurnPerformanceScenario::ModelLoadBundleCacheReuse.ci_safe(),
            "cache reuse should stay deterministic"
        );
        assert!(
            !BurnPerformanceScenario::TextEncodeModuleForward.ci_safe(),
            "module forward cost is an opt-in probe until a synced benchmark harness exists"
        );
        assert!(
            BurnPerformanceScenario::FullTopologyMemoryReview.requires_full_topology(),
            "full topology memory review must not imply executable full graph today"
        );
    }

    #[test]
    fn empty_backend_reports_deterministic_performance_envelope() {
        let backend = backend();

        let envelope = backend.performance_envelope();

        assert_eq!(envelope.backend_instance.as_str(), expected_instance());
        assert_eq!(envelope.device_label, expected_device_label());
        assert_eq!(envelope.observation("store_payloads"), Some(0));
        assert_eq!(envelope.observation("store_bytes_approximate"), Some(0));
        assert_eq!(envelope.observation("model_cache_bundles"), Some(0));
        assert_eq!(envelope.observation("text_encoder_cache_entries"), Some(0));
    }

    #[tokio::test]
    async fn envelope_tracks_store_payload_growth_without_timing_thresholds() {
        let backend = backend();
        let before = backend.performance_envelope();

        backend
            .create_empty_latent(CreateEmptyLatentRequest::new(
                512,
                512,
                1,
                RunId::new("run-metrics"),
                WorkflowId::new("workflow-metrics"),
                WorkflowVersion::new(1),
                NodeId::new("latent-metrics"),
            ))
            .await
            .expect("create empty latent");

        let after = backend.performance_envelope();

        assert_eq!(before.observation("store_payloads"), Some(0));
        assert_eq!(before.observation("store_bytes_approximate"), Some(0));
        assert_eq!(after.observation("store_payloads"), Some(1));
        assert_eq!(after.observation("store_bytes_approximate"), Some(65536));
        assert_eq!(after.observation("model_cache_bundles"), Some(0));
    }
}
