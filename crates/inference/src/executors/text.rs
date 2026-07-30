//! `builtin.clip_text_encode` executor.
//!
//! Maps to `text.encode`. Reads the `clip` and `text` inputs,
//! builds a `TextEncodeRequest`, and returns a `conditioning`
//! output.
//!
//! Slot mapping (`conditioning`) is executor-owned. The backend's
//! typed [`TextEncodeResponse`] returns the conditioning handle
//! without any `SlotId` mapping.
//!
//! Retention: the conditioning output is declared `RunScoped`. It is
//! typically consumed by the ksampler in the same run. Runtime owns
//! retention enforcement and value lifetime.

use crate::{ExecutionOutput, RouterRef, TextEncodeRequest, TextEncodeResponse};

use crate::error::into_executor_error;
use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};
use crate::executors::common::{optional_correlation_id, required_clip_input, required_input};
use crate::executors::validation::conditioning_output;

/// `builtin.clip_text_encode` executor.
pub struct ClipTextEncodeExecutor {
    inference: RouterRef,
}

impl ClipTextEncodeExecutor {
    pub fn new(inference: RouterRef) -> Self {
        Self { inference }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for ClipTextEncodeExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let clip = required_clip_input(&context, "clip", "text.encode")?;
        let text = required_input(&context, "text")?;

        let mut request = TextEncodeRequest::new(
            clip,
            text,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = optional_correlation_id(&context) {
            request = request.with_correlation_id(cid);
        }

        let invocation = context.inference_invocation();
        let response: TextEncodeResponse = self
            .inference
            .load()
            .text_encode_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![conditioning_output(&response)])
    }
}
