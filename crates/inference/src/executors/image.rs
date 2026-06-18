//! `builtin.vae_decode`, `builtin.save_image`, and
//! `builtin.preview_image` executors.
//!
//! - `vae_decode` maps to `latent.decode`. Reads `vae` and `latent`
//!   inputs, returns an `image` output.
//! - `save_image` maps to `image.save`. Reads `image` input plus
//!   optional `filename_prefix` param. The executor records the
//!   artifact via [`NodeArtifactCapability`].
//! - `preview_image` maps to `image.preview`. Reads `image` input.
//!   The executor records the artifact.
//!
//! Slot mapping (`image` for `vae_decode`, no outputs for
//! `save_image` / `preview_image`) is executor-owned. The backend's
//! typed responses return the value or artifact handle without any
//! `SlotId` mapping.

use std::sync::Arc;

use reimagine_core::ExecutionValue;
use reimagine_core::model::{ArtifactRef, ParamValue, SlotId};
use reimagine_inference_core::{
    ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse,
    InferenceRuntime, LatentDecodeRequest, LatentDecodeResponse,
};
use reimagine_runtime::{ArtifactEventKind, NodeExecutionContext, NodeExecutor, NodeExecutorError};

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

fn extract_vae(
    value: Arc<ExecutionValue>,
) -> Result<reimagine_core::RuntimeVaeHandle, NodeExecutorError> {
    match value.as_ref() {
        ExecutionValue::Vae(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: "latent.decode `vae` input must be a Vae handle".to_string(),
        }),
    }
}

fn extract_latent(
    value: Arc<ExecutionValue>,
) -> Result<reimagine_core::RuntimeLatent, NodeExecutorError> {
    match value.as_ref() {
        ExecutionValue::Latent(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: "latent.decode `latent` input must be a Latent handle".to_string(),
        }),
    }
}

fn extract_image(
    value: Arc<ExecutionValue>,
) -> Result<reimagine_core::RuntimeImage, NodeExecutorError> {
    match value.as_ref() {
        ExecutionValue::Image(handle) => Ok(handle.clone()),
        _ => Err(NodeExecutorError::Failed {
            message: "image.save/image.preview `image` input must be an Image handle".to_string(),
        }),
    }
}

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
    ) -> Result<Vec<(SlotId, Arc<ExecutionValue>)>, NodeExecutorError> {
        let vae = extract_vae(required_input(&context, "vae")?)?;
        let latent = extract_latent(required_input(&context, "latent")?)?;

        let correlation_id = context.correlation_id().cloned();
        let mut request = LatentDecodeRequest::new(
            vae,
            latent,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = correlation_id {
            request = request.with_correlation_id(cid);
        }

        let response: LatentDecodeResponse = self
            .inference
            .latent_decode(request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![(
            SlotId::new("image"),
            Arc::new(ExecutionValue::Image(response.into_image())),
        )])
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
    ) -> Result<Vec<(SlotId, Arc<ExecutionValue>)>, NodeExecutorError> {
        let image = extract_image(required_input(&context, "image")?)?;

        let mut request = ImageSaveRequest::new(
            image,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(prefix) = context.params().get(&SlotId::new("filename_prefix")) {
            if let ParamValue::String(s) = prefix {
                request = request.with_filename_prefix(s.clone());
            }
        }
        if let Some(cid) = context.correlation_id().cloned() {
            request = request.with_correlation_id(cid);
        }

        let response: ImageSaveResponse = self
            .inference
            .image_save(request)
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
    ) -> Result<Vec<(SlotId, Arc<ExecutionValue>)>, NodeExecutorError> {
        let image = extract_image(required_input(&context, "image")?)?;

        let mut request = ImagePreviewRequest::new(
            image,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = context.correlation_id().cloned() {
            request = request.with_correlation_id(cid);
        }

        let response: ImagePreviewResponse = self
            .inference
            .image_preview(request)
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
