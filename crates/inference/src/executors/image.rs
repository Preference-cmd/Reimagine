//! `builtin.vae_decode`, `builtin.save_image`, and
//! `builtin.preview_image` executors.
//!
//! - `vae_decode` maps to `latent.decode`. Reads `vae` and `latent`
//!   inputs, returns an `image` output.
//! - `save_image` maps to `image.save`. Reads `image` input plus
//!   optional `filename_prefix` param. The executor records the
//!   artifact via [`ArtifactPublisher`](crate::artifact_publisher::ArtifactPublisher).
//! - `preview_image` maps to `image.preview`. Reads `image` input.
//!   The executor records the artifact.
//!
//! Slot mapping (`image` for `vae_decode`, no outputs for
//! `save_image` / `preview_image`) is executor-owned. The backend's
//! typed responses return the value or artifact handle without any
//! `SlotId` mapping.
//!
//! Retention: the decoded `image` is declared `RunScoped`. Save and
//! preview nodes emit no execution value outputs; they publish
//! artifact observations instead.

use std::sync::Arc;

use crate::{
    ExecutionOutput, ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest,
    ImageSaveResponse, InferenceRuntime, LatentDecodeRequest, LatentDecodeResponse,
};
use reimagine_core::model::{ArtifactRef, SlotId};

use crate::artifact_publisher::ArtifactEventKind;
use crate::error::into_executor_error;
use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};
use crate::executors::common::{
    optional_correlation_id, optional_string_param, required_image_input, required_latent_input,
    required_vae_input,
};
use crate::executors::validation::image_output;

/// `builtin.vae_decode` executor.
pub struct VaeDecodeExecutor {
    inference: Arc<dyn InferenceRuntime>,
}

impl VaeDecodeExecutor {
    pub fn new(inference: Arc<dyn InferenceRuntime>) -> Self {
        Self { inference }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for VaeDecodeExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let vae = required_vae_input(&context, "vae", "latent.decode")?;
        let latent = required_latent_input(&context, "latent", "latent.decode")?;

        let mut request = LatentDecodeRequest::new(
            vae,
            latent,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = optional_correlation_id(&context) {
            request = request.with_correlation_id(cid);
        }

        // Runtime-side content validation. The candle backend
        // also rejects `EmptyGeometry`, but enforcing the
        // vocabulary at the executor boundary means a future
        // non-candle backend inherits the same semantics for free
        // and surfaces the same `LatentContentError` diagnostic
        // before any tensor work is dispatched.
        if let Err(err) = request.validate() {
            return Err(NodeExecutorError::Failed {
                message: err.to_string(),
            });
        }

        let invocation = context.inference_invocation();
        let response: LatentDecodeResponse = self
            .inference
            .latent_decode_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![image_output(&response)])
    }
}

/// `builtin.save_image` executor.
pub struct SaveImageExecutor {
    inference: Arc<dyn InferenceRuntime>,
}

impl SaveImageExecutor {
    pub fn new(inference: Arc<dyn InferenceRuntime>) -> Self {
        Self { inference }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for SaveImageExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let image = required_image_input(&context, "image", "image.save/image.preview")?;

        let mut request = ImageSaveRequest::new(
            image,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(prefix) = optional_string_param(&context, "filename_prefix") {
            request = request.with_filename_prefix(prefix);
        }
        if let Some(cid) = optional_correlation_id(&context) {
            request = request.with_correlation_id(cid);
        }

        let invocation = context.inference_invocation();
        let response: ImageSaveResponse = self
            .inference
            .image_save_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;
        let reference: ArtifactRef = response.into_artifact();

        let _ = context
            .artifacts()
            .record(SlotId::new("artifact"), reference, ArtifactEventKind::Saved)
            .await;

        // Save/preview nodes have no required runtime outputs.
        Ok(Vec::new())
    }
}

/// `builtin.preview_image` executor.
pub struct PreviewImageExecutor {
    inference: Arc<dyn InferenceRuntime>,
}

impl PreviewImageExecutor {
    pub fn new(inference: Arc<dyn InferenceRuntime>) -> Self {
        Self { inference }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for PreviewImageExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let image = required_image_input(&context, "image", "image.save/image.preview")?;

        let mut request = ImagePreviewRequest::new(
            image,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = optional_correlation_id(&context) {
            request = request.with_correlation_id(cid);
        }

        let invocation = context.inference_invocation();
        let response: ImagePreviewResponse = self
            .inference
            .image_preview_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;
        let reference: ArtifactRef = response.into_artifact();

        let _ = context
            .artifacts()
            .record(
                SlotId::new("artifact"),
                reference,
                ArtifactEventKind::Preview,
            )
            .await;

        Ok(Vec::new())
    }
}
