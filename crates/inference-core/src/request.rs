//! V1 inference operation ids and single-envelope request DTO.
//!
//! [`InferenceOperationId`] is the stable, backend-neutral
//! `"domain.action"` identifier that backends dispatch on. The
//! V1 enum and [`ALL_V1_OPERATIONS`] slice live here as
//! `&'static str` constants so they can be matched on cheaply
//! without crossing an `Arc` or dynamic string table.
//!
//! [`InferenceRequest`] is the single-envelope request DTO that
//! backends receive. It owns all the data a backend needs to
//! execute one operation and is deliberately self-contained so the
//! backend call can cross an `.await` boundary without borrowing
//! from [`reimagine_runtime::NodeExecutionContext`].

use std::collections::HashMap;
use std::sync::Arc;

use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::{NodeId, ParamValue, RunId, SlotId, WorkflowId, WorkflowVersion};
use reimagine_core::{BackendKind, ExecutionValue};

use crate::resolver::ResolvedInferenceModel;

// ── InferenceOperationId ───────────────────────────────────────────

/// Stable, backend-neutral identifier for an inference operation.
///
/// V1 uses a dot-separated `"domain.action"` naming convention.
/// New operations should be added as named constants rather than
/// raw string literals so the crate's public API is self-documenting.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct InferenceOperationId(String);

impl InferenceOperationId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for InferenceOperationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for InferenceOperationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for InferenceOperationId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

// ── V1 operation ids ───────────────────────────────────────────────

/// Load a model checkpoint bundle (model + CLIP + VAE handles).
pub const OP_MODEL_LOAD_BUNDLE: &str = "model.load_bundle";
/// Encode text through a text encoder (e.g. CLIP).
pub const OP_TEXT_ENCODE: &str = "text.encode";
/// Create an empty latent tensor.
pub const OP_LATENT_CREATE_EMPTY: &str = "latent.create_empty";
/// Run a diffusion sampling step (e.g. K-sampler).
pub const OP_DIFFUSION_SAMPLE: &str = "diffusion.sample";
/// Decode a latent tensor into pixel space.
pub const OP_LATENT_DECODE: &str = "latent.decode";
/// Save an image to disk.
pub const OP_IMAGE_SAVE: &str = "image.save";
/// Produce a preview image (not persisted).
pub const OP_IMAGE_PREVIEW: &str = "image.preview";

/// All V1 operation ids, in a fixed order.
pub const ALL_V1_OPERATIONS: &[&str] = &[
    OP_MODEL_LOAD_BUNDLE,
    OP_TEXT_ENCODE,
    OP_LATENT_CREATE_EMPTY,
    OP_DIFFUSION_SAMPLE,
    OP_LATENT_DECODE,
    OP_IMAGE_SAVE,
    OP_IMAGE_PREVIEW,
];

// ── InferenceRequest ───────────────────────────────────────────────

/// A backend-neutral inference request.
///
/// Inputs are keyed by `SlotId`, not by positional index, so
/// multi-output nodes and multi-model operations can be expressed
/// without relying on declaration order.
///
/// The `models` vector carries resolved model metadata even for
/// single-model operations (e.g. a checkpoint loader carries one
/// entry). This lets the backend know which model is being operated
/// on without introducing a special "single model" variant.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// The stable operation identifier.
    operation_id: InferenceOperationId,
    /// Input values keyed by `SlotId`.
    inputs: HashMap<SlotId, Arc<ExecutionValue>>,
    /// Typed node parameters keyed by `SlotId`.
    params: HashMap<SlotId, ParamValue>,
    /// Resolved model context. One entry for single-model operations;
    /// multiple for future multi-model operations (base + LoRA, etc).
    models: Vec<ResolvedInferenceModel>,
    /// Run context.
    run_id: RunId,
    /// Workflow context.
    workflow_id: WorkflowId,
    /// Workflow version context.
    workflow_version: WorkflowVersion,
    /// Correlation id from the host.
    correlation_id: Option<CorrelationId>,
    /// Originating node id.
    node_id: NodeId,
}

impl InferenceRequest {
    pub fn new(
        operation_id: InferenceOperationId,
        run_id: RunId,
        workflow_id: WorkflowId,
        workflow_version: WorkflowVersion,
        node_id: NodeId,
    ) -> Self {
        Self {
            operation_id,
            inputs: HashMap::new(),
            params: HashMap::new(),
            models: Vec::new(),
            run_id,
            workflow_id,
            workflow_version,
            correlation_id: None,
            node_id,
        }
    }

    pub fn with_input(mut self, slot_id: impl Into<SlotId>, value: Arc<ExecutionValue>) -> Self {
        self.inputs.insert(slot_id.into(), value);
        self
    }

    pub fn with_inputs(mut self, inputs: HashMap<SlotId, Arc<ExecutionValue>>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn with_param(mut self, slot_id: impl Into<SlotId>, value: ParamValue) -> Self {
        self.params.insert(slot_id.into(), value);
        self
    }

    pub fn with_params(mut self, params: HashMap<SlotId, ParamValue>) -> Self {
        self.params = params;
        self
    }

    pub fn with_model(mut self, model: ResolvedInferenceModel) -> Self {
        self.models.push(model);
        self
    }

    pub fn with_models(mut self, models: Vec<ResolvedInferenceModel>) -> Self {
        self.models = models;
        self
    }

    pub fn with_correlation_id(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn operation_id(&self) -> &InferenceOperationId {
        &self.operation_id
    }

    pub fn inputs(&self) -> &HashMap<SlotId, Arc<ExecutionValue>> {
        &self.inputs
    }

    pub fn params(&self) -> &HashMap<SlotId, ParamValue> {
        &self.params
    }

    pub fn models(&self) -> &[ResolvedInferenceModel] {
        &self.models
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

    /// Backend affinity observed from execution-value handles carried by the
    /// request.
    ///
    /// Param/artifact/null values do not imply backend affinity. Model, tensor,
    /// image, latent, and conditioning values do.
    pub fn backend_affinities(&self) -> Vec<BackendKind> {
        let mut kinds = Vec::new();
        for value in self.inputs.values() {
            collect_value_affinities(value, &mut kinds);
        }
        kinds
    }
}

fn push_unique(kinds: &mut Vec<BackendKind>, kind: &BackendKind) {
    if !kinds.iter().any(|existing| existing == kind) {
        kinds.push(kind.clone());
    }
}

fn collect_value_affinities(value: &ExecutionValue, kinds: &mut Vec<BackendKind>) {
    match value {
        ExecutionValue::Model(handle) => push_unique(kinds, handle.backend()),
        ExecutionValue::Clip(handle) => push_unique(kinds, handle.backend()),
        ExecutionValue::Vae(handle) => push_unique(kinds, handle.backend()),
        ExecutionValue::Latent(handle) => push_unique(kinds, handle.payload().backend()),
        ExecutionValue::Image(handle) => push_unique(kinds, handle.payload().backend()),
        ExecutionValue::Conditioning(conditioning) => {
            push_unique(kinds, conditioning.text_embedding().backend());
            if let Some(pooled) = conditioning.pooled_embedding() {
                push_unique(kinds, pooled.backend());
            }
        }
        ExecutionValue::Param(_) | ExecutionValue::Artifact(_) | ExecutionValue::Null => {}
    }
}
