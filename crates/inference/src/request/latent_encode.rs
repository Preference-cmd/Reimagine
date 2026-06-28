//! `latent.encode` request DTO.
//!
//! Consumes a [`RuntimeVaeHandle`] and a [`RuntimeImage`] handle and
//! returns a [`RuntimeLatent`] whose [`LatentContent`] is
//! `EncodedImage`. V1 requires VAE and image handles to belong to
//! the same [`BackendInstance`]; cross-backend encode is rejected
//! at the router/bridge boundary, never silently transferred.

use crate::BackendSelectionOverlay;
use crate::RuntimeImage;
use crate::RuntimeVaeHandle;
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

/// `latent.encode` request.
#[derive(Debug, Clone)]
pub struct LatentEncodeRequest {
    vae: RuntimeVaeHandle,
    image: RuntimeImage,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
    backend_selection: BackendSelectionOverlay,
}

impl LatentEncodeRequest {
    pub fn new(
        vae: RuntimeVaeHandle,
        image: RuntimeImage,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            vae,
            image,
            run_id,
            workflow_id,
            workflow_version,
            correlation_id: None,
            node_id,
            backend_selection: BackendSelectionOverlay::new(),
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn vae(&self) -> &RuntimeVaeHandle {
        &self.vae
    }

    pub fn image(&self) -> &RuntimeImage {
        &self.image
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn workflow_id(&self) -> &WorkflowId {
        &self.workflow_id
    }

    pub fn workflow_version(&self) -> WorkflowVersion {
        self.workflow_version
    }

    pub fn correlation_id(&self) -> Option<&CorrelationId> {
        self.correlation_id.as_ref()
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Backend affinity derived from the VAE and image handles.
    /// `latent.encode` requires both handles to live on the same
    /// backend instance; the router compares them.
    pub fn backend_affinities(&self) -> Vec<crate::BackendInstance> {
        let mut kinds = Vec::new();
        push_unique(&mut kinds, self.vae.backend_instance());
        push_unique(&mut kinds, self.image.payload().backend_instance());
        kinds
    }

    /// Per-request selection overlay supplied by the runtime.
    pub fn backend_selection_overlay(&self) -> &BackendSelectionOverlay {
        &self.backend_selection
    }

    /// Replace the request's selection overlay (for tests or
    /// runtime-pre-dispatch mutation).
    pub fn set_backend_selection_overlay(&mut self, overlay: BackendSelectionOverlay) {
        self.backend_selection = overlay;
    }
}

fn push_unique(kinds: &mut Vec<crate::BackendInstance>, kind: &crate::BackendInstance) {
    if !kinds.iter().any(|existing| existing == kind) {
        kinds.push(kind.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, BackendPayloadKey, BackendTensorHandle};
    use reimagine_core::model::{ModelId, TensorDType, TensorShape};

    fn vae_handle() -> RuntimeVaeHandle {
        RuntimeVaeHandle::new(ModelId::new("sdxl"), Backend::new("candle"), "vae-key")
    }

    fn image_handle() -> RuntimeImage {
        let tensor = BackendTensorHandle::new(
            Backend::new("candle"),
            BackendPayloadKey::new("image-key"),
            TensorDType::F32,
            TensorShape::new(vec![1, 3, 64, 64]),
            "cpu",
        );
        RuntimeImage::new(tensor, 64, 64, 1, "rgb")
    }

    #[test]
    fn request_carries_vae_and_image_handles() {
        let req = LatentEncodeRequest::new(
            vae_handle(),
            image_handle(),
            RunId::new("run-1"),
            WorkflowId::new("wf-1"),
            WorkflowVersion::new(1),
            NodeId::new("node-1"),
        );
        assert_eq!(req.vae().backend().as_str(), "candle");
        assert_eq!(req.image().width(), 64);
        assert_eq!(req.image().height(), 64);
        assert_eq!(req.backend_affinities().len(), 1);
        assert_eq!(
            req.backend_affinities()[0],
            crate::BackendInstance::new("candle")
        );
    }

    #[test]
    fn cross_backend_handles_yield_distinct_affinities() {
        let vae = vae_handle();
        let tensor = BackendTensorHandle::new(
            Backend::new("remote"),
            BackendPayloadKey::new("image-key"),
            TensorDType::F32,
            TensorShape::new(vec![1, 3, 64, 64]),
            "cpu",
        );
        let image = RuntimeImage::new(tensor, 64, 64, 1, "rgb");
        let req = LatentEncodeRequest::new(
            vae,
            image,
            RunId::new("run-1"),
            WorkflowId::new("wf-1"),
            WorkflowVersion::new(1),
            NodeId::new("node-1"),
        );
        assert_eq!(req.backend_affinities().len(), 2);
    }
}
