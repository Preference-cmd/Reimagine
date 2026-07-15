//! `builtin.vae_encode` executor.
//!
//! Maps to `latent.encode`. Reads `vae` and `image` inputs and
//! returns a `latent` output whose runtime content class is
//! `EncodedImage`. Backends that do not yet implement real VAE
//! encoding return a precise `BackendNotImplemented` rather than a
//! silent placeholder tensor.
//!
//! The executor enforces the V1 invariant that VAE and image
//! handles must share a [`BackendInstance`] *before* dispatching
//! the typed request; cross-backend encode is rejected at the
//! executor boundary so the runtime surfaces a clean bridge error.
//!
//! Slot mapping (`latent`) is executor-owned. The backend's typed
//! [`LatentEncodeResponse`] returns the latent handle without any
//! `SlotId` mapping.
//!
//! Retention: the encoded `latent` is declared `RunScoped`. It is
//! typically consumed by `builtin.ksampler` or `builtin.vae_decode`
//! in the same run. Runtime owns retention enforcement and value
//! lifetime.

use std::sync::Arc;

use crate::{ExecutionOutput, InferenceRuntime, LatentEncodeRequest, LatentEncodeResponse};

use crate::error::into_executor_error;
use crate::executor::{NodeExecutionContext, NodeExecutor, NodeExecutorError};
use crate::executors::common::{optional_correlation_id, required_image_input, required_vae_input};
use crate::executors::validation::encoded_latent_output;

/// `builtin.vae_encode` executor.
pub struct VaeEncodeExecutor {
    inference: Arc<dyn InferenceRuntime>,
}

impl VaeEncodeExecutor {
    pub fn new(inference: Arc<dyn InferenceRuntime>) -> Self {
        Self { inference }
    }
}

#[async_trait::async_trait]
impl NodeExecutor for VaeEncodeExecutor {
    async fn execute(
        &self,
        context: NodeExecutionContext,
    ) -> Result<Vec<ExecutionOutput>, NodeExecutorError> {
        let vae = required_vae_input(&context, "vae", "latent.encode")?;
        let image = required_image_input(&context, "image", "latent.encode")?;

        // V1 requires VAE and image handles to live on the same
        // backend instance. The router will reject mismatched
        // affinities with `BackendBridgeRequired` / `Unsupported`,
        // but the executor also surfaces a precise, deterministic
        // error message so the upstream node producing the
        // mismatch gets a clear attribution.
        if vae.backend_instance() != image.payload().backend_instance() {
            return Err(NodeExecutorError::Failed {
                message: format!(
                    "builtin.vae_encode requires VAE handle backend instance `{}` and image handle backend instance `{}` to match; cross-instance encode requires an explicit bridge (unsupported in V1)",
                    vae.backend_instance().as_str(),
                    image.payload().backend_instance().as_str()
                ),
            });
        }

        let mut request = LatentEncodeRequest::new(
            vae,
            image,
            context.run_id().clone(),
            context.workflow_id().clone(),
            context.workflow_version(),
            context.node_id().clone(),
        );
        if let Some(cid) = optional_correlation_id(&context) {
            request = request.with_correlation_id(cid);
        }

        let invocation = context.inference_invocation();
        let response: LatentEncodeResponse = self
            .inference
            .latent_encode_with_invocation(&invocation, request)
            .await
            .map_err(into_executor_error)?;

        Ok(vec![encoded_latent_output(&response)])
    }
}
