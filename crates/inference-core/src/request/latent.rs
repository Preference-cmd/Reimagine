//! `latent.create_empty` and `latent.decode` request DTOs.

use crate::RuntimeLatent;
use crate::RuntimeVaeHandle;
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};

/// `latent.create_empty` request.
#[derive(Debug, Clone)]
pub struct CreateEmptyLatentRequest {
    width: u32,
    height: u32,
    batch_size: u32,
    run_id: RunId,
    workflow_id: WorkflowId,
    workflow_version: WorkflowVersion,
    correlation_id: Option<CorrelationId>,
    node_id: NodeId,
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
            run_id,
            workflow_id,
            workflow_version,
            correlation_id: None,
            node_id,
        }
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
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
    /// whose payload is built from the request's run/node identity.
    pub fn into_latent(self) -> RuntimeLatent {
        RuntimeLatent::new(
            crate::BackendTensorHandle::new(
                crate::BackendKind::from("request"),
                crate::BackendPayloadKey::new(format!(
                    "latent:{}:{}",
                    self.run_id.as_str(),
                    self.node_id.as_str()
                )),
                reimagine_core::model::TensorDType::F32,
                reimagine_core::model::TensorShape::new(vec![
                    self.batch_size as usize,
                    4,
                    (self.height / 8) as usize,
                    (self.width / 8) as usize,
                ]),
                "cpu",
            ),
            self.width,
            self.height,
            self.batch_size,
            4,
        )
    }

    pub fn backend_affinities(&self) -> Vec<crate::BackendKind> {
        Vec::new()
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

    pub fn backend_affinities(&self) -> Vec<crate::BackendKind> {
        let mut kinds = Vec::new();
        push_unique(&mut kinds, self.vae.backend());
        push_unique(&mut kinds, self.latent.payload().backend());
        kinds
    }
}

fn push_unique(kinds: &mut Vec<crate::BackendKind>, kind: &crate::BackendKind) {
    if !kinds.iter().any(|existing| existing == kind) {
        kinds.push(kind.clone());
    }
}
