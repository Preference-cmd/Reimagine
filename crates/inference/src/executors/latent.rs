//! `builtin.empty_latent_image` executor.
//!
//! Maps to `latent.create_empty`. Reads `width`, `height`, and
//! `batch_size` params and returns a `latent` output.

use std::sync::Arc;

use reimagine_core::model::{ParamValue, SlotId, SlotKind};
use reimagine_runtime::{NodeExecutionContext, NodeExecutor, NodeExecutorError, RuntimeValue};

use crate::backend::InferenceBackend;
use crate::operation::OP_LATENT_CREATE_EMPTY;
use crate::request::InferenceRequest;

use super::validation::{ExpectedOutputSlot, validate_response};

/// `builtin.empty_latent_image` executor.
pub struct EmptyLatentImageExecutor {
    backend: Arc<dyn InferenceBackend>,
}

impl EmptyLatentImageExecutor {
    pub fn new(backend: Arc<dyn InferenceBackend>) -> Self {
        Self { backend }
    }
}

fn extract_i64(context: &NodeExecutionContext, slot: &str) -> Result<i64, NodeExecutorError> {
    match context.params().get(&SlotId::new(slot)) {
        Some(param) => match param {
            ParamValue::Integer(v) => Ok(*v),
            _ => Err(NodeExecutorError::Failed {
                message: format!("param `{slot}` must be an integer"),
            }),
        },
        None => Err(NodeExecutorError::MissingInput {
            slot_id: slot.to_string(),
        }),
    }
}

#[async_trait::async_trait]
impl NodeExecutor for EmptyLatentImageExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        let width = extract_i64(&context, "width")?;
        let height = extract_i64(&context, "height")?;
        let batch_size = extract_i64(&context, "batch_size")?;

        let request = InferenceRequest::new(
            OP_LATENT_CREATE_EMPTY.into(),
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        )
        .with_param("width", ParamValue::Integer(width))
        .with_param("height", ParamValue::Integer(height))
        .with_param("batch_size", ParamValue::Integer(batch_size));

        let response = self
            .backend
            .execute(request)
            .await
            .map_err(|e| e.into_executor_error())?;

        let expected = vec![ExpectedOutputSlot::required("latent", SlotKind::Latent)];
        validate_response(&response, &expected, false)
    }
}
