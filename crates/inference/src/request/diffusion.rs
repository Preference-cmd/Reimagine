//! `diffusion.sample` request DTO.

use crate::ExecutionConditioning;
use crate::RuntimeLatent;
use crate::RuntimeModelHandle;
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

/// Sampling algorithm selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplerName {
    Euler,
    Other(String),
}

impl SamplerName {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Euler => "euler",
            Self::Other(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for SamplerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Scheduler selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerName {
    Normal,
    Other(String),
}

impl SchedulerName {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Normal => "normal",
            Self::Other(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for SchedulerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
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
    pub fn backend_affinities(&self) -> Vec<crate::BackendKind> {
        let mut kinds = Vec::new();
        push_unique(&mut kinds, self.model.backend());
        push_unique(&mut kinds, self.positive.text_embedding().backend());
        if let Some(pooled) = self.positive.pooled_embedding() {
            push_unique(&mut kinds, pooled.backend());
        }
        push_unique(&mut kinds, self.negative.text_embedding().backend());
        if let Some(pooled) = self.negative.pooled_embedding() {
            push_unique(&mut kinds, pooled.backend());
        }
        push_unique(&mut kinds, self.latent.payload().backend());
        kinds
    }
}

fn push_unique(kinds: &mut Vec<crate::BackendKind>, kind: &crate::BackendKind) {
    if !kinds.iter().any(|existing| existing == kind) {
        kinds.push(kind.clone());
    }
}
