//! `builtin.clip_text_encode` executor.
//!
//! Maps to `text.encode`. Reads the `clip` and `text` inputs,
//! builds a `text.encode` request, and returns a `conditioning` output.

use std::sync::Arc;

use reimagine_core::model::{SlotId, SlotKind};
use reimagine_runtime::{NodeExecutionContext, NodeExecutor, NodeExecutorError, RuntimeValue};

use reimagine_inference_core::InferenceBackend;
use reimagine_inference_core::InferenceRequest;
use reimagine_inference_core::OP_TEXT_ENCODE;

use super::validation::{ExpectedOutputSlot, validate_response};

/// `builtin.clip_text_encode` executor.
pub struct ClipTextEncodeExecutor {
    backend: Arc<dyn InferenceBackend>,
}

impl ClipTextEncodeExecutor {
    pub fn new(backend: Arc<dyn InferenceBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for ClipTextEncodeExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        let clip = context.inputs().get(&SlotId::new("clip")).cloned().ok_or(
            NodeExecutorError::MissingInput {
                slot_id: "clip".to_string(),
            },
        )?;
        let text = context.inputs().get(&SlotId::new("text")).cloned().ok_or(
            NodeExecutorError::MissingInput {
                slot_id: "text".to_string(),
            },
        )?;

        let request = InferenceRequest::new(
            OP_TEXT_ENCODE.into(),
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        )
        .with_input("clip", clip)
        .with_input("text", text);

        let response = self
            .backend
            .execute(request)
            .await
            .map_err(|e| crate::error::into_executor_error(e))?;

        let expected = vec![ExpectedOutputSlot::required(
            "conditioning",
            SlotKind::Conditioning,
        )];
        validate_response(&response, &expected, false)
    }
}
