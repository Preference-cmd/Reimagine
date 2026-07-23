//! `builtin.empty_latent_image` executor.
//!
//! Maps to `latent.create_empty`. Reads `width`, `height`, and
//! `batch_size` params and returns a `latent` output.
//!
//! Slot mapping (`latent`) is executor-owned. The backend's typed
//! [`CreateEmptyLatentResponse`] returns the latent handle without
//! any `SlotId` mapping.
//!
//! Retention: the empty latent is declared `RunScoped`. Runtime owns
//! retention enforcement and value lifetime.

use crate::{
    CreateEmptyLatentRequest, CreateEmptyLatentResponse, ExecutionOutput, InferenceRuntime,
};

use crate::error::into_executor_error;
use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};
use crate::executors::common::{optional_correlation_id, required_u32_param};
use crate::executors::validation::latent_output;
use std::sync::Arc;

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
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let width = required_u32_param(&context, "width")?;
        let height = required_u32_param(&context, "height")?;
        let batch_size = required_u32_param(&context, "batch_size")?;

        let mut request = CreateEmptyLatentRequest::new(
            width,
            height,
            batch_size,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = optional_correlation_id(&context) {
            request = request.with_correlation_id(cid);
        }

        let invocation = context.inference_invocation();
        let response: CreateEmptyLatentResponse = self
            .inference
            .create_empty_latent_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![latent_output(&response)])
    }
}
