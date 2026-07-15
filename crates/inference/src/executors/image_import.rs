//! `builtin.load_image` executor.
//!
//! Maps to `image.import`. Reads a `Path` param and returns an
//! `image` output. The executor is the boundary where the workspace
//! safety policy lives: it must use an injected [`ImageSourceResolver`]
//! to turn the raw `Path` param into a workspace-safe
//! [`ResolvedImageSource`] before the request reaches the backend.
//!
//! V1 keeps the executor agnostic of the actual image decoder.
//! `app-host` injects the resolver; backends see only the
//! already-authorized [`ResolvedImageSource`] and never inspect
//! arbitrary paths.
//!
//! Slot mapping (`image`) is executor-owned. The backend's typed
//! [`ImageImportResponse`] returns the image handle without any
//! `SlotId` mapping.
//!
//! Retention: the imported image is declared `RunScoped`. It is
//! typically consumed by `builtin.vae_encode` or `builtin.save_image`
//! in the same run. Runtime owns retention enforcement and value
//! lifetime.

use std::sync::Arc;

use crate::{
    ExecutionOutput, ImageImportRequest, ImageImportResponse, InferenceRuntime, ResolvedImageSource,
};
use reimagine_core::diagnostic::CorrelationId;
use reimagine_core::model::SlotId;

use crate::error::into_executor_error;
use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};
use crate::executors::common::optional_correlation_id;
use crate::executors::validation::imported_image_output;

/// Workspace-safe image source resolver.
///
/// `app-host` implements this trait and injects the implementation
/// into the executor registry. The trait owns the policy that
/// keeps V1 inputs inside `<base_path>/input/` and rejects absolute
/// paths and parent escapes; the inference layer never inspects
/// paths.
pub trait ImageSourceResolver: Send + Sync + 'static {
    /// Resolve a workflow `Path` param into a workspace-safe
    /// [`ResolvedImageSource`].
    fn resolve(&self, path: &std::path::Path) -> Result<ResolvedImageSource, NodeExecutorError>;
}

/// `builtin.load_image` executor.
pub struct LoadImageExecutor {
    inference: Arc<dyn InferenceRuntime>,
    resolver: Arc<dyn ImageSourceResolver>,
}

impl LoadImageExecutor {
    pub fn new(
        inference: Arc<dyn InferenceRuntime>,
        resolver: Arc<dyn ImageSourceResolver>,
    ) -> Self {
        Self {
            inference,
            resolver,
        }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for LoadImageExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let path_param = context.params().get(&SlotId::new("image")).ok_or_else(|| {
            NodeExecutorError::MissingInput {
                slot_id: "image".to_string(),
            }
        })?;
        let raw_path = match path_param {
            reimagine_core::model::ParamValue::Path(p) => std::path::PathBuf::from(p),
            other => {
                return Err(NodeExecutorError::Failed {
                    message: format!(
                        "builtin.load_image `image` param must be a path, got {}",
                        param_kind_name(other)
                    ),
                });
            }
        };
        let source = self.resolver.resolve(&raw_path)?;

        let mut request = ImageImportRequest::new(
            source,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = optional_correlation_id(&context) {
            request = request.with_correlation_id(cid);
        }

        let invocation = context.inference_invocation();
        let response: ImageImportResponse = self
            .inference
            .image_import_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![imported_image_output(&response)])
    }
}

fn param_kind_name(value: &reimagine_core::model::ParamValue) -> &'static str {
    use reimagine_core::model::ParamValue;
    match value {
        ParamValue::String(_) => "string",
        ParamValue::Text(_) => "text",
        ParamValue::Integer(_) => "integer",
        ParamValue::Float(_) => "float",
        ParamValue::Bool(_) => "bool",
        ParamValue::Seed(_) => "seed",
        ParamValue::Select(_) => "select",
        ParamValue::Path(_) => "path",
        ParamValue::ModelRef(_) => "model_ref",
        ParamValue::Null => "null",
    }
}

// Make `CorrelationId` reachable from this module so the
// `with_correlation_id` call resolves unambiguously.
#[allow(dead_code)]
fn _unused_for_resolve(_cid: &CorrelationId) {}
