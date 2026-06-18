//! `builtin.checkpoint_loader` executor.
//!
//! Maps to `model.load_bundle`. Uses the injected
//! [`ModelResolver`](crate::resolver::ModelResolver) to resolve the
//! workflow `ModelRef` into a `ResolvedInferenceModel`, then calls
//! the backend. The backend returns three handles: `model`, `clip`,
//! `vae`.
//!
//! Slot mapping (`model`, `clip`, `vae`) is owned by this executor
//! because it knows the workflow node shape. The backend's typed
//! [`LoadBundleResponse`] returns the three handles without any
//! `SlotId` mapping.

use std::sync::Arc;

use reimagine_core::model::{ModelRef, ParamValue, SlotId};
use reimagine_inference_core::{
    InferenceBackend, LoadBundleRequest, LoadBundleResponse, ModelResolver,
};
use reimagine_runtime::{NodeExecutionContext, NodeExecutor, NodeExecutorError, RuntimeValue};

use crate::error::into_executor_error;

/// `builtin.checkpoint_loader` executor.
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
        let model_ref: ModelRef = match context.params().get(&SlotId::new("checkpoint")) {
            Some(ParamValue::ModelRef(mr)) => mr.clone(),
            _ => {
                return Err(NodeExecutorError::MissingInput {
                    slot_id: "checkpoint".to_string(),
                });
            }
        };

        let resolved = self
            .resolver
            .resolve(&model_ref)
            .await
            .map_err(into_executor_error)?;

        let correlation_id = context.correlation_id().cloned();
        let mut request = LoadBundleRequest::new(
            resolved,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = correlation_id {
            request = request.with_correlation_id(cid);
        }

        let response: LoadBundleResponse = self
            .backend
            .load_bundle(request)
            .await
            .map_err(into_executor_error)?;

        Ok(checked_output(
            response,
            context.correlation_id(),
            context.node_id().as_str(),
        ))
    }
}

fn checked_output(
    response: LoadBundleResponse,
    correlation_id: Option<&reimagine_core::diagnostic::CorrelationId>,
    node_id: &str,
) -> Vec<(SlotId, Arc<RuntimeValue>)> {
    if let Some(cid) = correlation_id {
        let _ = cid;
    }
    let _ = node_id;
    vec![
        (
            SlotId::new("model"),
            Arc::new(RuntimeValue::Model(response.model().clone())),
        ),
        (
            SlotId::new("clip"),
            Arc::new(RuntimeValue::Clip(response.clip().clone())),
        ),
        (
            SlotId::new("vae"),
            Arc::new(RuntimeValue::Vae(response.vae().clone())),
        ),
    ]
}
