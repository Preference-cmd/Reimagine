//! `builtin.empty_latent_image` executor.
//!
//! Maps to `latent.create_empty`. Reads `width`, `height`, and
//! `batch_size` params and returns a `latent` output.
//!
//! Slot mapping (`latent`) is executor-owned. The backend's typed
//! [`CreateEmptyLatentResponse`] returns the latent handle without
//! any `SlotId` mapping.

use std::sync::Arc;

use reimagine_core::ExecutionValue;
use reimagine_core::model::{ParamValue, SlotId};
use reimagine_inference_core::{
    CreateEmptyLatentRequest, CreateEmptyLatentResponse, InferenceRuntime,
};
use reimagine_runtime::{NodeExecutionContext, NodeExecutor, NodeExecutorError};

use crate::error::into_executor_error;

fn extract_u32(context: &NodeExecutionContext, slot: &str) -> Result<u32, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(ParamValue::Integer(v)) => u32::try_from(*v).map_err(|_| NodeExecutorError::Failed {
            message: format!("param `{slot}` must fit in u32, got {v}"),
        }),
        Some(_) => Err(NodeExecutorError::Failed {
            message: format!("param `{slot}` must be an integer"),
        }),
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

/// `builtin.empty_latent_image` executor.
pub struct EmptyLatentImageExecutor {
    inference: Arc<dyn InferenceRuntime>,
}

impl EmptyLatentImageExecutor {
    pub fn new(inference: Arc<dyn InferenceRuntime>) -> Self {
        Self { inference }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for EmptyLatentImageExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<ExecutionValue>)>, NodeExecutorError> {
        let width = extract_u32(&context, "width")?;
        let height = extract_u32(&context, "height")?;
        let batch_size = extract_u32(&context, "batch_size")?;

        let correlation_id = context.correlation_id().cloned();
        let mut request = CreateEmptyLatentRequest::new(
            width,
            height,
            batch_size,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = correlation_id {
            request = request.with_correlation_id(cid);
        }

        let response: CreateEmptyLatentResponse = self
            .inference
            .create_empty_latent(request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![(
            SlotId::new("latent"),
            Arc::new(ExecutionValue::Latent(response.into_latent())),
        )])
    }
}
