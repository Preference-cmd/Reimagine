//! `builtin.clip_text_encode` executor.
//!
//! Maps to `text.encode`. Reads the `clip` and `text` inputs,
//! builds a `TextEncodeRequest`, and returns a `conditioning`
//! output.
//!
//! Slot mapping (`conditioning`) is executor-owned. The backend's
//! typed [`TextEncodeResponse`] returns the conditioning handle
//! without any `SlotId` mapping.

use std::sync::Arc;

use reimagine_core::ExecutionValue;
use reimagine_core::model::SlotId;
use reimagine_inference_core::{InferenceRuntime, TextEncodeRequest, TextEncodeResponse};
use reimagine_runtime::{NodeExecutionContext, NodeExecutor, NodeExecutorError, RuntimeValue};

use crate::error::into_executor_error;

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

fn extract_clip(
    value: Arc<ExecutionValue>,
) -> Result<reimagine_core::RuntimeClipHandle, NodeExecutorError> {
    match value.as_ref() {
        ExecutionValue::Clip(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: "text.encode `clip` input must be a Clip handle".to_string(),
        }),
    }
}

/// `builtin.clip_text_encode` executor.
pub struct ClipTextEncodeExecutor {
    inference: Arc<dyn InferenceRuntime>,
}

impl ClipTextEncodeExecutor {
    pub fn new(inference: Arc<dyn InferenceRuntime>) -> Self {
        Self { inference }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for ClipTextEncodeExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        let clip = extract_clip(required_input(&context, "clip")?)?;
        let text = required_input(&context, "text")?;

        let correlation_id = context.correlation_id().cloned();
        let mut request = TextEncodeRequest::new(
            clip,
            text,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = correlation_id {
            request = request.with_correlation_id(cid);
        }

        let response: TextEncodeResponse = self
            .inference
            .text_encode(request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![(
            SlotId::new("conditioning"),
            Arc::new(RuntimeValue::Conditioning(response.into_conditioning())),
        )])
    }
}
