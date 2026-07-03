use std::sync::Arc;

use burn_ndarray::NdArrayDevice;
use reimagine_inference::{
    Backend, BackendInstance, CreateEmptyLatentRequest, CreateEmptyLatentResponse,
    DiffusionSampleRequest, DiffusionSampleResponse, ImageImportRequest, ImageImportResponse,
    ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse,
    InferenceBackend, InferenceBackendCapabilities, InferenceCapability,
    InferenceCapabilitySupport, InferenceError, LatentDecodeRequest, LatentDecodeResponse,
    LatentEncodeRequest, LatentEncodeResponse, LoadBundleRequest, LoadBundleResponse,
    TextEncodeRequest, TextEncodeResponse,
};

use crate::config::BurnBackendConfig;
use crate::error::BurnBackendError;
use crate::operation::{
    execute_latent_create_empty, execute_model_load_bundle, execute_text_encode,
    map_to_inference_error,
};
use crate::profile::{BACKEND_LABEL, BurnProfileProvider};
use crate::resource::BurnBackendInstanceRuntimeHooks;
use crate::store::{BurnModelCache, BurnStore};

#[derive(Debug, Clone)]
pub struct BurnBackend {
    config: BurnBackendConfig,
    device: NdArrayDevice,
    store: Arc<BurnStore>,
    model_cache: Arc<BurnModelCache>,
}

impl BurnBackend {
    pub fn new(config: BurnBackendConfig) -> Result<Self, BurnBackendError> {
        let device = config.device().try_build_device()?;
        Ok(Self {
            config,
            device,
            store: Arc::new(BurnStore::new()),
            model_cache: Arc::new(BurnModelCache::new()),
        })
    }

    pub fn config(&self) -> &BurnBackendConfig {
        &self.config
    }

    pub fn device(&self) -> &NdArrayDevice {
        &self.device
    }

    pub fn device_label(&self) -> &str {
        self.config.device_label()
    }

    pub fn backend_instance(&self) -> BackendInstance {
        BackendInstance::new(format!("{BACKEND_LABEL}:{}", self.device_label()))
    }

    pub fn store(&self) -> &Arc<BurnStore> {
        &self.store
    }

    pub fn model_cache(&self) -> &Arc<BurnModelCache> {
        &self.model_cache
    }

    pub fn runtime_hooks(
        &self,
        plugin: Option<reimagine_plugin::Plugin>,
        extension: Option<reimagine_plugin::Extension>,
        device: Option<reimagine_inference::DeviceProfile>,
    ) -> BurnBackendInstanceRuntimeHooks {
        BurnBackendInstanceRuntimeHooks::new(
            self.backend_instance(),
            self.backend_kind().clone(),
            plugin,
            extension,
            device,
            self.store.clone(),
            self.model_cache.clone(),
        )
    }

    fn not_implemented<T>(&self, capability: InferenceCapability) -> Result<T, InferenceError> {
        Err(InferenceError::BackendNotImplemented {
            capability,
            backend_kind: BACKEND_LABEL.to_owned(),
            message: Some(
                "Burn backend skeleton is registered for discovery but does not execute inference yet"
                    .to_owned(),
            ),
        })
    }
}

fn map_err<T>(result: Result<T, BurnBackendError>) -> Result<T, InferenceError> {
    result.map_err(|err| InferenceError::BackendExecutionFailed {
        message: err.to_string(),
    })
}

#[async_trait::async_trait]
impl InferenceBackend for BurnBackend {
    fn backend_kind(&self) -> &Backend {
        static KIND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
        KIND.get_or_init(BurnProfileProvider::backend_kind)
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        InferenceBackendCapabilities::new(self.backend_kind().clone())
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::LoadBundle,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::CreateEmptyLatent,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::TextEncode,
            ))
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
        // burn/08f implements the real text.encode pipeline: validate
        // the request, tokenize the prompt, store the conditioning
        // payload, and return backend-affine handles. The CLIP-L/CLIP-G
        // tensor forward pass is wired for correct shape metadata;
        // the actual tensor execution is a follow-up deepening.
        execute_text_encode(self, request).map_err(|err| InferenceError::BackendExecutionFailed {
            message: err.to_string(),
        })
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        execute_latent_create_empty(self, request).map_err(map_to_inference_error)
    }

    async fn diffusion_sample(
        &self,
        _request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        self.not_implemented(InferenceCapability::DiffusionSample)
    }

    async fn latent_decode(
        &self,
        _request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        self.not_implemented(InferenceCapability::LatentDecode)
    }

    async fn latent_encode(
        &self,
        _request: LatentEncodeRequest,
    ) -> Result<LatentEncodeResponse, InferenceError> {
        self.not_implemented(InferenceCapability::LatentEncode)
    }

    async fn image_import(
        &self,
        _request: ImageImportRequest,
    ) -> Result<ImageImportResponse, InferenceError> {
        self.not_implemented(InferenceCapability::ImageImport)
    }

    async fn image_save(
        &self,
        _request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        self.not_implemented(InferenceCapability::ImageSave)
    }

    async fn image_preview(
        &self,
        _request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        self.not_implemented(InferenceCapability::ImagePreview)
    }
}
