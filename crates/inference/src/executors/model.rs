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
//!
//! Retention: model/clip/vae handles are declared `WorkspaceScoped`
//! because they back the rest of the workflow and should outlive any
//! single run. Runtime owns retention enforcement and value lifetime.

use crate::{
    ExecutionOutput, InferenceRuntime, LoadBundleRequest, LoadBundleResponse, ModelResolver,
};
use reimagine_core::model::{ModelRef, ParamValue, SlotId};

use crate::error::into_executor_error;
use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};
use crate::executors::validation::load_bundle_outputs;
use std::sync::Arc;

/// `builtin.checkpoint_loader` executor.
pub struct CheckpointLoaderExecutor {
    inference: Arc<dyn InferenceRuntime>,
    resolver: Arc<dyn ModelResolver>,
}

impl CheckpointLoaderExecutor {
    pub fn new(inference: Arc<dyn InferenceRuntime>, resolver: Arc<dyn ModelResolver>) -> Self {
        Self {
            inference,
            resolver,
        }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for CheckpointLoaderExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
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

        let invocation = context.inference_invocation();
        let response: LoadBundleResponse = self
            .inference
            .load_bundle_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;

        Ok(load_bundle_outputs(&response))
    }
}
