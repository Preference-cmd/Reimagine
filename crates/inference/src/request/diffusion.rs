//! `diffusion.sample` request DTO.

use crate::BackendSelectionOverlay;
use crate::ExecutionConditioning;
use crate::RuntimeLatent;
use crate::RuntimeModelHandle;
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Sampling algorithm selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplerName {
    Euler,
    EulerAncestral,
    Heun,
    Lms,
    Dpmpp2m,
    Dpmpp2mSde,
    Dpmpp3mSde,
    Other(String),
}

impl SamplerName {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Euler => "euler",
            Self::EulerAncestral => "euler_ancestral",
            Self::Heun => "heun",
            Self::Lms => "lms",
            Self::Dpmpp2m => "dpmpp_2m",
            Self::Dpmpp2mSde => "dpmpp_2m_sde",
            Self::Dpmpp3mSde => "dpmpp_3m_sde",
            Self::Other(s) => s.as_str(),
        }
    }

    pub fn from_standard_name(name: impl AsRef<str>) -> Self {
        match name.as_ref() {
            "euler" => Self::Euler,
            "euler_ancestral" => Self::EulerAncestral,
            "heun" => Self::Heun,
            "lms" => Self::Lms,
            "dpmpp_2m" => Self::Dpmpp2m,
            "dpmpp_2m_sde" => Self::Dpmpp2mSde,
            "dpmpp_3m_sde" => Self::Dpmpp3mSde,
            other => Self::Other(other.to_string()),
        }
    }
}

impl std::fmt::Display for SamplerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for SamplerName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SamplerName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from_standard_name(value))
    }
}

/// Scheduler selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerName {
    Normal,
    Karras,
    Exponential,
    SgmUniform,
    Simple,
    DdimUniform,
    Other(String),
}

impl SchedulerName {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Normal => "normal",
            Self::Karras => "karras",
            Self::Exponential => "exponential",
            Self::SgmUniform => "sgm_uniform",
            Self::Simple => "simple",
            Self::DdimUniform => "ddim_uniform",
            Self::Other(s) => s.as_str(),
        }
    }

    pub fn from_standard_name(name: impl AsRef<str>) -> Self {
        match name.as_ref() {
            "normal" => Self::Normal,
            "karras" => Self::Karras,
            "exponential" => Self::Exponential,
            "sgm_uniform" => Self::SgmUniform,
            "simple" => Self::Simple,
            "ddim_uniform" => Self::DdimUniform,
            other => Self::Other(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_name_maps_standard_vocabulary_and_preserves_unknown_names() {
        assert_eq!(SamplerName::from_standard_name("euler"), SamplerName::Euler);
        assert_eq!(
            SamplerName::from_standard_name("euler_ancestral"),
            SamplerName::EulerAncestral
        );
        assert_eq!(SamplerName::from_standard_name("heun"), SamplerName::Heun);
        assert_eq!(SamplerName::from_standard_name("lms"), SamplerName::Lms);
        assert_eq!(
            SamplerName::from_standard_name("dpmpp_2m"),
            SamplerName::Dpmpp2m
        );
        assert_eq!(
            SamplerName::from_standard_name("dpmpp_2m_sde"),
            SamplerName::Dpmpp2mSde
        );
        assert_eq!(
            SamplerName::from_standard_name("dpmpp_3m_sde"),
            SamplerName::Dpmpp3mSde
        );
        assert_eq!(
            SamplerName::from_standard_name("backend_only"),
            SamplerName::Other("backend_only".to_string())
        );
        assert_eq!(SamplerName::Dpmpp2mSde.as_str(), "dpmpp_2m_sde");
    }

    #[test]
    fn scheduler_name_maps_standard_vocabulary_and_preserves_unknown_names() {
        assert_eq!(
            SchedulerName::from_standard_name("normal"),
            SchedulerName::Normal
        );
        assert_eq!(
            SchedulerName::from_standard_name("karras"),
            SchedulerName::Karras
        );
        assert_eq!(
            SchedulerName::from_standard_name("exponential"),
            SchedulerName::Exponential
        );
        assert_eq!(
            SchedulerName::from_standard_name("sgm_uniform"),
            SchedulerName::SgmUniform
        );
        assert_eq!(
            SchedulerName::from_standard_name("simple"),
            SchedulerName::Simple
        );
        assert_eq!(
            SchedulerName::from_standard_name("ddim_uniform"),
            SchedulerName::DdimUniform
        );
        assert_eq!(
            SchedulerName::from_standard_name("backend_only"),
            SchedulerName::Other("backend_only".to_string())
        );
        assert_eq!(SchedulerName::SgmUniform.as_str(), "sgm_uniform");
    }
}

impl std::fmt::Display for SchedulerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for SchedulerName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SchedulerName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from_standard_name(value))
    }
}

/// `diffusion.sample` request.
#[derive(Debug, Clone)]
pub struct DiffusionSampleRequest {
    model: RuntimeModelHandle,
    positive: ExecutionConditioning,
    negative: ExecutionConditioning,
    latent: RuntimeLatent,
    seed: u64,
    steps: u32,
    cfg: f32,
    sampler: SamplerName,
    scheduler: SchedulerName,
    denoise: f32,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
    backend_selection: BackendSelectionOverlay,
}

impl DiffusionSampleRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: RuntimeModelHandle,
        positive: ExecutionConditioning,
        negative: ExecutionConditioning,
        latent: RuntimeLatent,
        seed: u64,
        steps: u32,
        cfg: f32,
        sampler: SamplerName,
        scheduler: SchedulerName,
        denoise: f32,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            model,
            positive,
            negative,
            latent,
            seed,
            steps,
            cfg,
            sampler,
            scheduler,
            denoise,
            run_id,
            workflow_id,
            workflow_version,
            correlation_id: None,
            node_id,
            backend_selection: BackendSelectionOverlay::new(),
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn model(&self) -> &RuntimeModelHandle {
        &self.model
    }

    pub fn positive(&self) -> &ExecutionConditioning {
        &self.positive
    }

    pub fn negative(&self) -> &ExecutionConditioning {
        &self.negative
    }

    pub fn latent(&self) -> &RuntimeLatent {
        &self.latent
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    pub fn cfg(&self) -> f32 {
        self.cfg
    }

    pub fn sampler(&self) -> &SamplerName {
        &self.sampler
    }

    pub fn scheduler(&self) -> &SchedulerName {
        &self.scheduler
    }

    pub fn denoise(&self) -> f32 {
        self.denoise
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn workflow_version(&self) -> WorkflowVersion {
        self.workflow_version
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Backend affinity derived from the model, conditioning, and
    /// latent handles.
    pub fn backend_affinities(&self) -> Vec<crate::BackendInstance> {
        let mut kinds = Vec::new();
        push_unique(&mut kinds, self.model.backend_instance());
        push_unique(
            &mut kinds,
            self.positive.text_embedding().backend_instance(),
        );
        if let Some(pooled) = self.positive.pooled_embedding() {
            push_unique(&mut kinds, pooled.backend_instance());
        }
        push_unique(
            &mut kinds,
            self.negative.text_embedding().backend_instance(),
        );
        if let Some(pooled) = self.negative.pooled_embedding() {
            push_unique(&mut kinds, pooled.backend_instance());
        }
        push_unique(&mut kinds, self.latent.payload().backend_instance());
        kinds
    }

    /// Per-request selection overlay supplied by the runtime.
    pub fn backend_selection_overlay(&self) -> &BackendSelectionOverlay {
        &self.backend_selection
    }

    /// Replace the request's selection overlay (for tests or
    /// runtime-pre-dispatch mutation).
    pub fn set_backend_selection_overlay(&mut self, overlay: BackendSelectionOverlay) {
        self.backend_selection = overlay;
    }
}

fn push_unique(kinds: &mut Vec<crate::BackendInstance>, kind: &crate::BackendInstance) {
    if !kinds.iter().any(|existing| existing == kind) {
        kinds.push(kind.clone());
    }
}
