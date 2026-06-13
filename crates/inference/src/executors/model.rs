//! `builtin.checkpoint_loader` executor.
//!
//! Maps to `model.load_bundle`. Uses the injected
//! [`ModelResolver`](crate::resolver::ModelResolver) to resolve the
//! workflow `ModelRef` into a `ResolvedInferenceModel`, then calls
//! the backend. The backend returns three handles: `model`, `clip`,
//! `vae`.

use std::sync::Arc;

use reimagine_core::model::{ModelRef, ParamValue, SlotId, SlotKind};
use reimagine_runtime::{NodeExecutionContext, NodeExecutor, NodeExecutorError, RuntimeValue};

use crate::backend::InferenceBackend;
use crate::operation::OP_MODEL_LOAD_BUNDLE;
use crate::request::InferenceRequest;
use crate::resolver::ModelResolver;

use super::validation::{ExpectedOutputSlot, validate_response};

/// `builtin.checkpoint_loader` executor.
///
/// Maps the node's `checkpoint` param (a `ModelRef`) through the
/// resolver, then calls `model.load_bundle` on the backend. Returns
/// `model`, `clip`, and `vae` outputs.
pub struct CheckpointLoaderExecutor {
    backend: Arc<dyn InferenceBackend>,
    resolver: Arc<dyn ModelResolver>,
}

impl CheckpointLoaderExecutor {
    pub fn new(backend: Arc<dyn InferenceBackend>, resolver: Arc<dyn ModelResolver>) -> Self {
        Self { backend, resolver }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for CheckpointLoaderExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<(SlotId, Arc<RuntimeValue>)>, NodeExecutorError> {
        // Extract the ModelRef from the params.
        let model_ref: ModelRef = match context.params().get(&SlotId::new("checkpoint")) {
            Some(ParamValue::ModelRef(mr)) => mr.clone(),
            _ => {
                return Err(NodeExecutorError::MissingInput {
                    slot_id: "checkpoint".to_string(),
                });
            }
        };

        // Resolve the model through the injected resolver.
        let resolved = self
            .resolver
            .resolve(&model_ref)
            .await
            .map_err(|e| e.into_executor_error())?;

        // Build the request.
        let request = InferenceRequest::new(
            OP_MODEL_LOAD_BUNDLE.into(),
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        )
        .with_model(resolved);

        // Execute.
        let response = self
            .backend
            .execute(request)
            .await
            .map_err(|e| e.into_executor_error())?;

        // Validate.
        let expected = vec![
            ExpectedOutputSlot::required("model", SlotKind::Model),
            ExpectedOutputSlot::required("clip", SlotKind::Clip),
            ExpectedOutputSlot::required("vae", SlotKind::Vae),
        ];
        validate_response(&response, &expected, false)
    }
}
