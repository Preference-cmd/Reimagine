//! `builtin.ksampler` executor.
//!
//! Maps to `diffusion.sample`. Reads model, positive, negative, and
//! latent inputs plus seed, steps, cfg, sampler, scheduler, denoise
//! params and returns a `latent` output.
//!
//! Slot mapping is executor-owned: the executor owns the `latent`
//! slot for `KSamplerExecutor`. The backend's typed
//! [`DiffusionSampleResponse`] returns the latent handle without any
//! `SlotId` mapping.
//!
//! Retention: the sampled `latent` is declared `RunScoped`. Runtime
//! owns retention enforcement and value lifetime.

use std::sync::Arc;

use crate::{
    DiffusionSampleRequest, DiffusionSampleResponse, ExecutionOutput, InferenceRuntime,
    SamplerName, SchedulerName,
};

use crate::error::into_executor_error;
use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};
use crate::executors::common::{
    optional_correlation_id, optional_string_param, required_conditioning_input,
    required_f32_param, required_i64_param, required_latent_input, required_model_input,
    required_seed_param,
};
use crate::executors::validation::sampled_latent_output;

fn select_sampler(name: Option<String>) -> SamplerName {
    match name.as_deref() {
        Some(name) => SamplerName::from_standard_name(name),
        None => SamplerName::Euler,
    }
}

fn select_scheduler(name: Option<String>) -> SchedulerName {
    match name.as_deref() {
        Some(name) => SchedulerName::from_standard_name(name),
        None => SchedulerName::Normal,
    }
}

/// `builtin.ksampler` executor.
pub struct KSamplerExecutor {
    inference: Arc<dyn InferenceRuntime>,
}

impl KSamplerExecutor {
    pub fn new(inference: Arc<dyn InferenceRuntime>) -> Self {
        Self { inference }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for KSamplerExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let model_handle = required_model_input(&context, "model", "diffusion.sample")?;
        let positive = required_conditioning_input(&context, "positive", "diffusion.sample")?;
        let negative = required_conditioning_input(&context, "negative", "diffusion.sample")?;
        let latent = required_latent_input(&context, "latent", "diffusion.sample")?;

        let seed = required_seed_param(&context, "seed")?;
        let steps_i64 = required_i64_param(&context, "steps")?;
        let steps: u32 = steps_i64
            .try_into()
            .map_err(|_| NodeExecutorError::Failed {
                message: format!("param `steps` must be a non-negative u32, got {steps_i64}"),
            })?;
        let cfg = required_f32_param(&context, "cfg")?;
        let sampler = select_sampler(optional_string_param(&context, "sampler"));
        let scheduler = select_scheduler(optional_string_param(&context, "scheduler"));
        let denoise = required_f32_param(&context, "denoise")?;

        let mut request = DiffusionSampleRequest::new(
            model_handle,
            positive,
            negative,
            latent,
            seed,
            steps,
            cfg,
            sampler,
            scheduler,
            denoise,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = optional_correlation_id(&context) {
            request = request.with_correlation_id(cid);
        }

        let invocation = context.inference_invocation();
        let response: DiffusionSampleResponse = self
            .inference
            .diffusion_sample_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![sampled_latent_output(&response)])
    }
}
