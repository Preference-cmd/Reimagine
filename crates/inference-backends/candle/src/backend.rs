use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::Device;
use reimagine_inference::{
    Backend, BackendInstance, CreateEmptyLatentRequest, CreateEmptyLatentResponse,
    DiffusionSampleRequest, DiffusionSampleResponse, ImagePreviewRequest, ImagePreviewResponse,
    ImageSaveRequest, ImageSaveResponse, InferenceBackend, InferenceBackendCapabilities,
    InferenceCapability, InferenceCapabilitySupport, InferenceError, LatentDecodeRequest,
    LatentDecodeResponse, LoadBundleRequest, LoadBundleResponse, TextEncodeRequest,
    TextEncodeResponse,
};

use crate::config::CandleBackendConfig;
use crate::error::CandleBackendError;
use crate::operation::*;
use crate::resource::CandleResourceMechanism;
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

    pub fn backend_instance(&self) -> BackendInstance {
        BackendInstance::new(format!("candle:{}", self.device_label()))
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

    pub fn resource_mechanism(
        &self,
        plugin: Option<reimagine_plugin::Plugin>,
        extension: Option<reimagine_plugin::Extension>,
        device: Option<reimagine_inference::DeviceProfile>,
    ) -> CandleResourceMechanism {
        let backend_instance = self.backend_instance();
        let backend_label = self.backend_kind().clone();
        CandleResourceMechanism::new(
            backend_instance,
            backend_label,
            plugin,
            extension,
            device,
            self.store.clone(),
            self.model_cache.clone(),
        )
    }

    pub fn next_image_seq(&self) -> u64 {
        self.next_image_seq.fetch_add(1, Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl InferenceBackend for CandleBackend {
    fn backend_kind(&self) -> &Backend {
        static KIND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
        KIND.get_or_init(|| Backend::new("candle"))
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        let caps = InferenceBackendCapabilities::new(self.backend_kind().clone())
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::LoadBundle,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::CreateEmptyLatent,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::TextEncode,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::DiffusionSample,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::LatentDecode,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::ImageSave,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::ImagePreview,
            ));
        caps
    }

    async fn load_bundle(
        &self,
        request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        map_err(execute_model_load_bundle(request, self))
    }

    async fn text_encode(
        &self,
        request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        map_err(execute_text_encode(request, self))
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        map_err(execute_latent_create_empty(self, request))
    }

    async fn diffusion_sample(
        &self,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        map_err(execute_diffusion_sample(request, self))
    }

    async fn latent_decode(
        &self,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        map_err(execute_latent_decode(request, self))
    }

    async fn image_save(
        &self,
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        map_err(execute_image_save(request, self))
    }

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        map_err(execute_image_preview(request, self))
    }
}

fn map_err<T>(result: Result<T, CandleBackendError>) -> Result<T, InferenceError> {
    result.map_err(|e| match e {
        CandleBackendError::BackendNotImplemented(err) => {
            InferenceError::BackendNotImplemented {
                capability: err.capability(),
                backend_kind: err.backend_kind().to_string(),
                message: Some(err.message().to_string()),
            }
        }
        CandleBackendError::InvalidRequest(message) => {
            InferenceError::BackendExecutionFailed { message }
        }
        CandleBackendError::DeviceUnavailable { reason, .. } => {
            InferenceError::BackendExecutionFailed { message: reason }
        }
        CandleBackendError::UnsupportedModelFamily {
            model_id,
            series,
            variant,
        } => InferenceError::BackendExecutionFailed {
            message: format!(
                "candle backend has no loader for model `{model_id}` (series `{series}`, variant `{variant}`)"
            ),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_core::model::{NodeId, RunId, WorkflowId, WorkflowVersion};
    use reimagine_inference::TextEncodeRequest;

    fn backend() -> CandleBackend {
        CandleBackend::new(CandleBackendConfig::new(
            "/tmp/reimagine-candle-unit",
            "/tmp/reimagine-candle-unit-output",
        ))
        .unwrap()
    }

    fn base_load_bundle_request() -> LoadBundleRequest {
        // We can't actually call load_bundle without a resolved model,
        // but we can verify capabilities advertise it.
        unimplemented!()
    }

    #[test]
    fn backend_kind_is_candle() {
        let backend = backend();
        assert_eq!(backend.backend_kind().as_str(), "candle");
    }

    #[test]
    fn capabilities_lists_all_v1_capabilities() {
        let backend = backend();
        let caps = backend.capabilities();
        assert_eq!(caps.backend_kind().as_str(), "candle");
        for cap in InferenceCapability::all_v1() {
            assert!(
                caps.supports_capability(*cap),
                "capability report should include {cap}"
            );
        }
    }

    #[tokio::test]
    async fn text_encode_without_loaded_bundle_returns_error() {
        let backend = backend();
        let clip = reimagine_inference::RuntimeClipHandle::new(
            reimagine_core::model::ModelId::new("missing"),
            Backend::new("candle"),
            reimagine_inference::BackendPayloadKey::new("k"),
        );
        let text = std::sync::Arc::new(reimagine_inference::ExecutionValue::Param(
            reimagine_core::model::ParamValue::String("hi".to_string()),
        ));
        let req = TextEncodeRequest::new(
            clip,
            text,
            RunId::new("r"),
            WorkflowId::new("w"),
            WorkflowVersion::new(1),
            NodeId::new("n"),
        );
        let err = backend.text_encode(req).await.unwrap_err();
        let msg = match err {
            InferenceError::BackendExecutionFailed { message } => message,
            other => panic!("expected BackendExecutionFailed, got {other:?}"),
        };
        assert!(
            msg.contains("no loaded model bundle"),
            "expected missing-bundle error, got {msg}"
        );
    }

    #[allow(dead_code)]
    fn _referenced() {
        let _ = base_load_bundle_request();
    }
}
