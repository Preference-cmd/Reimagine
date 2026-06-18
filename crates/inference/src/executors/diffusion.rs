//! `builtin.ksampler` executor.
//!
//! Maps to `diffusion.sample`. Reads model, positive, negative, and
//! latent inputs plus seed, steps, cfg, sampler, scheduler, denoise
//! params and returns a `latent` output.

use std::sync::Arc;

use reimagine_core::model::{SlotId, SlotKind};
use reimagine_runtime::{NodeExecutionContext, NodeExecutor, NodeExecutorError, RuntimeValue};

use reimagine_inference_core::InferenceBackend;
use reimagine_inference_core::InferenceRequest;
use reimagine_inference_core::OP_DIFFUSION_SAMPLE;

use super::validation::{ExpectedOutputSlot, validate_response};

/// `builtin.ksampler` executor.
pub struct KSamplerExecutor {
    backend: Arc<dyn InferenceBackend>,
}

impl KSamplerExecutor {
    pub fn new(backend: Arc<dyn InferenceBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for KSamplerExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        let model = context.inputs().get(&SlotId::new("model")).cloned().ok_or(
            NodeExecutorError::MissingInput {
                slot_id: "model".to_string(),
            },
        )?;
        let positive = context
            .inputs()
            .get(&SlotId::new("positive"))
            .cloned()
            .ok_or(NodeExecutorError::MissingInput {
                slot_id: "positive".to_string(),
            })?;
        let negative = context
            .inputs()
            .get(&SlotId::new("negative"))
            .cloned()
            .ok_or(NodeExecutorError::MissingInput {
                slot_id: "negative".to_string(),
            })?;
        let latent = context
            .inputs()
            .get(&SlotId::new("latent"))
            .cloned()
            .ok_or(NodeExecutorError::MissingInput {
                slot_id: "latent".to_string(),
            })?;

        let mut request = InferenceRequest::new(
            OP_DIFFUSION_SAMPLE.into(),
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        )
        .with_input("model", model)
        .with_input("positive", positive)
        .with_input("negative", negative)
        .with_input("latent", latent);

        // Forward sampling params.
        for param_name in &["seed", "steps", "cfg", "sampler", "scheduler", "denoise"] {
            if let Some(value) = context.params().get(&SlotId::new(*param_name)) {
                request = request.with_param(*param_name, value.clone());
            }
        }

        let response = self
            .backend
            .execute(request)
            .await
            .map_err(|e| crate::error::into_executor_error(e))?;

        let expected = vec![ExpectedOutputSlot::required("latent", SlotKind::Latent)];
        validate_response(&response, &expected, false)
    }
}
