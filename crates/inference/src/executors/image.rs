//! `builtin.vae_decode`, `builtin.save_image`, and
//! `builtin.preview_image` executors.
//!
//! - `vae_decode` maps to `latent.decode`. Reads `vae` and `latent`
//!   inputs, returns an `image` output.
//! - `save_image` maps to `image.save`. Reads `image` input plus
//!   optional `filename_prefix` param. No backend outputs are
//!   required; the executor records the artifact.
//! - `preview_image` maps to `image.preview`. Reads `image` input.
//!   No backend outputs are required; the executor records the
//!   artifact.

use std::sync::Arc;

use reimagine_core::model::SlotId;
use reimagine_runtime::{
    ArtifactEventKind, NodeExecutionContext, NodeExecutor, NodeExecutorError, RuntimeValue,
};

use crate::backend::InferenceBackend;
use crate::operation::{OP_IMAGE_PREVIEW, OP_IMAGE_SAVE, OP_LATENT_DECODE};
use crate::request::InferenceRequest;

use super::validation::{ExpectedOutputSlot, validate_response};

/// `builtin.vae_decode` executor.
pub struct VaeDecodeExecutor {
    backend: Arc<dyn InferenceBackend>,
}

impl VaeDecodeExecutor {
    pub fn new(backend: Arc<dyn InferenceBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for VaeDecodeExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        let vae = context.inputs().get(&SlotId::new("vae")).cloned().ok_or(
            NodeExecutorError::MissingInput {
                slot_id: "vae".to_string(),
            },
        )?;
        let latent = context
            .inputs()
            .get(&SlotId::new("latent"))
            .cloned()
            .ok_or(NodeExecutorError::MissingInput {
                slot_id: "latent".to_string(),
            })?;

        let request = InferenceRequest::new(
            OP_LATENT_DECODE.into(),
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        )
        .with_input("vae", vae)
        .with_input("latent", latent);

        let response = self
            .backend
            .execute(request)
            .await
            .map_err(|e| e.into_executor_error())?;

        let expected = vec![ExpectedOutputSlot::required("image")];
        validate_response(&response, &expected, false)
    }
}

/// `builtin.save_image` executor.
///
/// After the backend returns, the executor records an artifact via
/// the runtime's [`NodeArtifactCapability`].
pub struct SaveImageExecutor {
    backend: Arc<dyn InferenceBackend>,
}

impl SaveImageExecutor {
    pub fn new(backend: Arc<dyn InferenceBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for SaveImageExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        let image = context.inputs().get(&SlotId::new("image")).cloned().ok_or(
            NodeExecutorError::MissingInput {
                slot_id: "image".to_string(),
            },
        )?;

        let mut request = InferenceRequest::new(
            OP_IMAGE_SAVE.into(),
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        )
        .with_input("image", image);

        if let Some(prefix) = context.params().get(&SlotId::new("filename_prefix")) {
            request = request.with_param("filename_prefix", prefix.clone());
        }

        let response = self
            .backend
            .execute(request)
            .await
            .map_err(|e| e.into_executor_error())?;

        // The save backend MUST return at least one output so we
        // have something to record. An empty response is a backend
        // bug, not a silent no-op.
        let first = response
            .outputs()
            .first()
            .ok_or(NodeExecutorError::Failed {
                message: "image.save backend returned no outputs".to_string(),
            })?;

        let reference = reimagine_core::model::ArtifactRef::new(format!(
            "save-image-{}",
            context.node_id().as_str()
        ));
        let _ = context
            .artifacts()
            .record(first.slot_id().clone(), reference, ArtifactEventKind::Saved)
            .await;

        // Save/preview nodes have no required runtime outputs.
        Ok(Vec::new())
    }
}

/// `builtin.preview_image` executor.
///
/// Like `SaveImageExecutor` but records as a preview artifact.
pub struct PreviewImageExecutor {
    backend: Arc<dyn InferenceBackend>,
}

impl PreviewImageExecutor {
    pub fn new(backend: Arc<dyn InferenceBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for PreviewImageExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        let image = context.inputs().get(&SlotId::new("image")).cloned().ok_or(
            NodeExecutorError::MissingInput {
                slot_id: "image".to_string(),
            },
        )?;

        let request = InferenceRequest::new(
            OP_IMAGE_PREVIEW.into(),
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        )
        .with_input("image", image);

        let response = self
            .backend
            .execute(request)
            .await
            .map_err(|e| e.into_executor_error())?;

        let first = response
            .outputs()
            .first()
            .ok_or(NodeExecutorError::Failed {
                message: "image.preview backend returned no outputs".to_string(),
            })?;

        let reference = reimagine_core::model::ArtifactRef::new(format!(
            "preview-image-{}",
            context.node_id().as_str()
        ));
        let _ = context
            .artifacts()
            .record(
                first.slot_id().clone(),
                reference,
                ArtifactEventKind::Preview,
            )
            .await;

        Ok(Vec::new())
    }
}
