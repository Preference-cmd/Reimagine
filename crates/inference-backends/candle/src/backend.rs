use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::Device;
use reimagine_inference::InferenceBackend;
use reimagine_inference::capability::{InferenceBackendCapabilities, InferenceOperationSupport};
use reimagine_inference::operation::*;
use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::InferenceResponse;

use crate::config::CandleBackendConfig;
use crate::error::{BackendNotImplementedError, CandleBackendError};
use crate::operation::*;
use crate::resource::CandleRunResourceBackend;
use crate::store::{CandleModelCache, CandleStore};

#[derive(Debug)]
pub struct CandleBackend {
    config: CandleBackendConfig,
    device: Arc<Device>,
    store: Arc<CandleStore>,
    model_cache: Arc<CandleModelCache>,
    next_image_seq: AtomicU64,
}

impl Clone for CandleBackend {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            device: Arc::clone(&self.device),
            store: Arc::clone(&self.store),
            model_cache: Arc::clone(&self.model_cache),
            next_image_seq: AtomicU64::new(self.next_image_seq.load(Ordering::Relaxed)),
        }
    }
}

impl CandleBackend {
    pub fn new(config: CandleBackendConfig) -> Result<Self, CandleBackendError> {
        let device = config.device().try_build_device()?;
        Ok(Self {
            config,
            device: Arc::new(device),
            store: Arc::new(CandleStore::new()),
            model_cache: Arc::new(CandleModelCache::new()),
            next_image_seq: AtomicU64::new(0),
        })
    }

    pub fn config(&self) -> &CandleBackendConfig {
        &self.config
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn device_label(&self) -> &str {
        self.config.device().label()
    }

    pub fn store(&self) -> &Arc<CandleStore> {
        &self.store
    }

    pub fn model_cache(&self) -> &Arc<CandleModelCache> {
        &self.model_cache
    }

    pub fn output_dir(&self) -> &Path {
        self.config().output_dir()
    }

    pub fn resource_backend(&self) -> CandleRunResourceBackend {
        CandleRunResourceBackend::new(self.store.clone(), self.model_cache.clone())
    }

    pub fn next_image_seq(&self) -> u64 {
        self.next_image_seq.fetch_add(1, Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl InferenceBackend for CandleBackend {
    fn backend_kind(&self) -> &str {
        "candle"
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        InferenceBackendCapabilities::new(self.backend_kind())
            .with_support(InferenceOperationSupport::new(OP_MODEL_LOAD_BUNDLE.into()))
            .with_support(InferenceOperationSupport::new(
                OP_LATENT_CREATE_EMPTY.into(),
            ))
            .with_support(InferenceOperationSupport::new(OP_TEXT_ENCODE.into()))
            .with_support(InferenceOperationSupport::new(OP_DIFFUSION_SAMPLE.into()))
            .with_support(InferenceOperationSupport::new(OP_LATENT_DECODE.into()))
            .with_support(InferenceOperationSupport::new(OP_IMAGE_SAVE.into()))
            .with_support(InferenceOperationSupport::new(OP_IMAGE_PREVIEW.into()))
    }

    async fn execute(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, reimagine_inference::error::InferenceError> {
        let result = match request.operation_id().as_str() {
            OP_MODEL_LOAD_BUNDLE => execute_model_load_bundle(&request, self),
            OP_LATENT_CREATE_EMPTY => execute_latent_create_empty(self, &request),
            OP_TEXT_ENCODE => execute_text_encode(&request, self),
            OP_DIFFUSION_SAMPLE => execute_diffusion_sample(&request, self),
            OP_LATENT_DECODE => execute_latent_decode(&request, self),
            OP_IMAGE_SAVE => execute_image_save(&request, self),
            OP_IMAGE_PREVIEW => execute_image_preview(&request, self),
            _ => Err(CandleBackendError::BackendNotImplemented(
                BackendNotImplementedError::new(
                    self.backend_kind(),
                    request.operation_id().clone(),
                    "operation not implemented",
                ),
            )),
        };
        result.map_err(|e| match e {
            CandleBackendError::BackendNotImplemented(err) => {
                reimagine_inference::error::InferenceError::BackendNotImplemented {
                    operation_id: err.operation_id().to_string(),
                    backend_kind: err.backend_kind().to_string(),
                    message: Some(err.message().to_string()),
                }
            }
            CandleBackendError::InvalidRequest(message) => {
                reimagine_inference::error::InferenceError::BackendExecutionFailed { message }
            }
            CandleBackendError::DeviceUnavailable { reason, .. } => {
                reimagine_inference::error::InferenceError::BackendExecutionFailed {
                    message: reason,
                }
            }
            CandleBackendError::UnsupportedModelFamily {
                model_id,
                series,
                variant,
            } => reimagine_inference::error::InferenceError::BackendExecutionFailed {
                message: format!(
                    "candle backend has no loader for model `{model_id}` (series `{series}`, variant `{variant}`)"
                ),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
    use reimagine_inference::operation::OP_TEXT_ENCODE;

    fn backend() -> CandleBackend {
        CandleBackend::new(CandleBackendConfig::new(
            "/tmp/reimagine-candle-unit",
            "/tmp/reimagine-candle-unit-output",
        ))
        .unwrap()
    }

    fn base_request(operation_id: &str) -> InferenceRequest {
        InferenceRequest::new(
            operation_id.into(),
            RunId::new("run-test"),
            WorkflowId::new("wf-test"),
            WorkflowVersion::new(1),
            NodeId::new("node-test"),
        )
    }

    #[test]
    fn backend_kind_is_candle() {
        let backend = backend();
        assert_eq!(backend.backend_kind(), "candle");
    }

    #[test]
    fn capabilities_lists_all_v1_operations() {
        let backend = backend();
        let caps = backend.capabilities();
        assert_eq!(caps.backend_kind(), "candle");
        for op in reimagine_inference::operation::ALL_V1_OPERATIONS {
            assert!(caps.supports_operation(&(*op).into()));
        }
    }

    #[tokio::test]
    async fn execute_unknown_operation_returns_not_implemented_with_message() {
        let backend = backend();
        let err = backend
            .execute(base_request("custom.unknown"))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("candle"), "{msg}");
        assert!(msg.contains("custom.unknown"), "{msg}");
    }

    #[tokio::test]
    async fn execute_text_encode_requires_clip_input() {
        let backend = backend();
        let err = backend
            .execute(base_request(OP_TEXT_ENCODE))
            .await
            .unwrap_err();
        let exec_err = err.into_executor_error();
        let msg = exec_err.to_string();
        assert!(msg.contains("clip"), "{msg}");
    }
}
