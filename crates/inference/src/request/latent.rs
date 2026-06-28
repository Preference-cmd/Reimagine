//! `latent.create_empty` and `latent.decode` request DTOs.

use crate::BackendSelectionOverlay;
use crate::RuntimeLatent;
use crate::RuntimeVaeHandle;
use crate::latent_space::{
    LatentSpaceError, LatentSpaceMetadata, validate_pixel_dimensions_against,
};
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

/// `latent.create_empty` request.
#[derive(Debug, Clone)]
pub struct CreateEmptyLatentRequest {
    width: u32,
    height: u32,
    batch_size: u32,
    latent_space: LatentSpaceMetadata,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
    backend_selection: BackendSelectionOverlay,
}

impl CreateEmptyLatentRequest {
    pub fn new(
        width: u32,
        height: u32,
        batch_size: u32,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            width,
            height,
            batch_size,
            latent_space: LatentSpaceMetadata::sdxl_base(),
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

    /// Override the request's latent-space metadata. V1 callers can
    /// leave this at the default SDXL base; future work will surface
    /// latent-space selection at the node param level.
    pub fn with_latent_space(mut self, latent_space: LatentSpaceMetadata) -> Self {
        self.latent_space = latent_space;
        self
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn batch_size(&self) -> u32 {
        self.batch_size
    }

    /// Latent-space metadata this request will create. Defaults to
    /// SDXL base when the caller did not override it.
    pub fn latent_space(&self) -> &LatentSpaceMetadata {
        &self.latent_space
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

    /// Consume the request and return a [`RuntimeLatent`] handle
    /// whose payload is built from the request's run/node identity
    /// and latent-space metadata. The returned handle carries the
    /// request's latent-space metadata so downstream operations
    /// (sample, decode) can validate compatibility without
    /// re-deriving it.
    pub fn try_into_latent(self) -> Result<RuntimeLatent, LatentSpaceError> {
        validate_pixel_dimensions_against(self.width, self.height, &self.latent_space)?;

        let scale = self.latent_space.spatial_scale_factor();
        let channels = self.latent_space.channels();
        let payload = crate::BackendTensorHandle::new(
            crate::Backend::from("request"),
            crate::BackendPayloadKey::new(format!(
                "latent:{}:{}",
                self.run_id.as_str(),
                self.node_id.as_str()
            )),
            self.latent_space.dtype(),
            reimagine_core::model::TensorShape::new(vec![
                self.batch_size as usize,
                channels as usize,
                (self.height / scale) as usize,
                (self.width / scale) as usize,
            ]),
            "cpu",
        );
        Ok(RuntimeLatent::new(
            payload,
            self.width,
            self.height,
            self.batch_size,
            channels,
            self.latent_space,
        ))
    }

    pub fn into_latent(self) -> RuntimeLatent {
        self.try_into_latent()
            .expect("CreateEmptyLatentRequest dimensions must be compatible with latent space")
    }

    pub fn backend_affinities(&self) -> Vec<crate::BackendInstance> {
        Vec::new()
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

/// `latent.decode` request.
#[derive(Debug, Clone)]
pub struct LatentDecodeRequest {
    vae: RuntimeVaeHandle,
    latent: RuntimeLatent,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
    backend_selection: BackendSelectionOverlay,
}

impl LatentDecodeRequest {
    pub fn new(
        vae: RuntimeVaeHandle,
        latent: RuntimeLatent,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            vae,
            latent,
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

    pub fn latent(&self) -> &RuntimeLatent {
        &self.latent
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

    pub fn backend_affinities(&self) -> Vec<crate::BackendInstance> {
        let mut kinds = Vec::new();
        push_unique(&mut kinds, self.vae.backend_instance());
        push_unique(&mut kinds, self.latent.payload().backend_instance());
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
    use crate::TensorLayout;
    use crate::latent_space::LatentSpaceMetadata;
    use reimagine_core::model::{NodeId, RunId, TensorDType, WorkflowId, WorkflowVersion};

    fn request() -> CreateEmptyLatentRequest {
        CreateEmptyLatentRequest::new(
            64,
            64,
            1,
            RunId::new("run-1"),
            WorkflowId::new("wf-1"),
            WorkflowVersion::new(1),
            NodeId::new("node-1"),
        )
    }

    #[test]
    fn new_request_defaults_to_sdxl_base_latent_space() {
        let req = request();
        assert_eq!(req.latent_space(), &LatentSpaceMetadata::sdxl_base());
    }

    #[test]
    fn with_latent_space_overrides_default() {
        let custom = LatentSpaceMetadata::new(
            crate::LatentSpaceId::new("custom/v1"),
            8,
            4,
            TensorDType::F32,
            TensorLayout::Nchw,
        );
        let req = request().with_latent_space(custom.clone());
        assert_eq!(req.latent_space(), &custom);
    }

    #[test]
    fn into_latent_carries_request_latent_space() {
        let req = request();
        let expected = req.latent_space().clone();
        let latent = req.into_latent();
        assert_eq!(latent.latent_space(), &expected);
        assert_eq!(latent.channels(), expected.channels());
        assert_eq!(latent.width(), 64);
        assert_eq!(latent.height(), 64);
    }

    #[test]
    fn into_latent_uses_latent_space_scale_factor() {
        let custom = LatentSpaceMetadata::new(
            crate::LatentSpaceId::new("scale/4"),
            4,
            4,
            TensorDType::F32,
            TensorLayout::Nchw,
        );
        let req = CreateEmptyLatentRequest::new(
            64,
            64,
            2,
            RunId::new("run-1"),
            WorkflowId::new("wf-1"),
            WorkflowVersion::new(1),
            NodeId::new("node-1"),
        )
        .with_latent_space(custom);

        let latent = req.into_latent();
        assert_eq!(latent.width(), 64);
        assert_eq!(latent.height(), 64);
        let shape = latent.payload().shape();
        // scale=4: latent dims = 16x16 for a 64x64 image, 2 batch
        assert_eq!(shape.dims(), &[2, 4, 16, 16]);
    }

    #[test]
    fn try_into_latent_rejects_non_divisible_dimensions() {
        let req = CreateEmptyLatentRequest::new(
            63,
            64,
            1,
            RunId::new("run-1"),
            WorkflowId::new("wf-1"),
            WorkflowVersion::new(1),
            NodeId::new("node-1"),
        );

        let err = req.try_into_latent().unwrap_err();
        assert!(matches!(
            err,
            crate::LatentSpaceError::ScaleMismatch {
                axis: "width",
                value: 63,
                scale: 8,
            }
        ));
    }

    #[test]
    fn try_into_latent_rejects_zero_scale() {
        let invalid = LatentSpaceMetadata::new(
            crate::LatentSpaceId::new("invalid/zero-scale"),
            4,
            0,
            TensorDType::F32,
            TensorLayout::Nchw,
        );
        let req = request().with_latent_space(invalid);

        let err = req.try_into_latent().unwrap_err();
        assert!(matches!(
            err,
            crate::LatentSpaceError::InvalidDimensions {
                axis: "spatial_scale_factor",
                value: 0,
                ..
            }
        ));
    }
}
