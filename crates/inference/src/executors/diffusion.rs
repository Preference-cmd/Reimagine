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
//! Retention: the sampled `latent` is declared `RunScoped` during
//! this migration, until single-use fan-out diagnostics are
//! implemented in issue 05.

use std::sync::Arc;

use reimagine_core::model::{ParamValue, SlotId};
use reimagine_inference_core::{
    DiffusionSampleRequest, DiffusionSampleResponse, ExecutionConditioning, ExecutionOutput,
    ExecutionValue, InferenceRuntime, RuntimeLatent, RuntimeModelHandle, SamplerName,
    SchedulerName,
};

use crate::error::into_executor_error;
use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};

fn required_input(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<Arc<ExecutionValue>, NodeExecutorError> {
    context
        .inputs()
        .get(&SlotId::new(slot))
        .cloned()
        .ok_or_else(|| NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        })
}

fn required_param_seed(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<u64, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::Seed(v)) => Ok(*v),
        Some(other) => Err(NodeExecutorError::Failed {
            message: format!(
                "param `{slot}` must be a seed, got {}",
                param_kind_name(other)
            ),
        }),
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

fn required_param_i64(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<i64, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::Integer(v)) => Ok(*v),
        Some(other) => Err(NodeExecutorError::Failed {
            message: format!(
                "param `{slot}` must be an integer, got {}",
                param_kind_name(other)
            ),
        }),
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

fn required_param_f32(
    context: &NodeExecutionContext,
    slot: &str,
) -> Result<f32, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::Float(v)) => Ok(*v as f32),
        Some(other) => Err(NodeExecutorError::Failed {
            message: format!(
                "param `{slot}` must be a float, got {}",
                param_kind_name(other)
            ),
        }),
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

fn optional_param_string(context: &NodeExecutionContext, slot: &str) -> Option<String> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn param_kind_name(value: &ParamValue) -> &'static str {
    match value {
        ParamValue::String(_) => "string",
        ParamValue::Text(_) => "text",
        ParamValue::Integer(_) => "integer",
        ParamValue::Float(_) => "float",
        ParamValue::Bool(_) => "bool",
        ParamValue::Seed(_) => "seed",
        ParamValue::Select(_) => "select",
        ParamValue::Path(_) => "path",
        ParamValue::ModelRef(_) => "model_ref",
        ParamValue::Null => "null",
    }
}

fn select_sampler(name: Option<String>) -> SamplerName {
    match name.as_deref() {
        Some("euler") => SamplerName::Euler,
        Some(other) => SamplerName::Other(other.to_string()),
        None => SamplerName::Euler,
    }
}

fn select_scheduler(name: Option<String>) -> SchedulerName {
    match name.as_deref() {
        Some("normal") => SchedulerName::Normal,
        Some(other) => SchedulerName::Other(other.to_string()),
        None => SchedulerName::Normal,
    }
}

fn extract_model_handle(
    value: Arc<ExecutionValue>,
) -> Result<RuntimeModelHandle, NodeExecutorError> {
    match value.as_ref() {
        ExecutionValue::Model(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: "diffusion.sample `model` input must be a Model handle".to_string(),
        }),
    }
}

fn extract_conditioning(
    value: Arc<ExecutionValue>,
) -> Result<ExecutionConditioning, NodeExecutorError> {
    match value.as_ref() {
        ExecutionValue::Conditioning(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: "diffusion.sample conditioning input must be a Conditioning handle"
                .to_string(),
        }),
    }
}

fn extract_latent(value: Arc<ExecutionValue>) -> Result<RuntimeLatent, NodeExecutorError> {
    match value.as_ref() {
        ExecutionValue::Latent(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: "diffusion.sample `latent` input must be a Latent handle".to_string(),
        }),
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
        let model_handle = extract_model_handle(required_input(&context, "model")?)?;
        let positive = extract_conditioning(required_input(&context, "positive")?)?;
        let negative = extract_conditioning(required_input(&context, "negative")?)?;
        let latent = extract_latent(required_input(&context, "latent")?)?;

        let seed = required_param_seed(&context, "seed")?;
        let steps_i64 = required_param_i64(&context, "steps")?;
        let steps: u32 = steps_i64
            .try_into()
            .map_err(|_| NodeExecutorError::Failed {
                message: format!("param `steps` must be a non-negative u32, got {steps_i64}"),
            })?;
        let cfg = required_param_f32(&context, "cfg")?;
        let sampler = select_sampler(optional_param_string(&context, "sampler"));
        let scheduler = select_scheduler(optional_param_string(&context, "scheduler"));
        let denoise = required_param_f32(&context, "denoise")?;

        let correlation_id = context.correlation_id().cloned();
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
        if let Some(cid) = correlation_id {
            request = request.with_correlation_id(cid);
        }

        let response: DiffusionSampleResponse = self
            .inference
            .diffusion_sample(request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![ExecutionOutput::run_scoped(
            SlotId::new("latent"),
            Arc::new(ExecutionValue::Latent(response.into_latent())),
        )])
    }
}
