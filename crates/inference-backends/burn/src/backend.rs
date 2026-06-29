use burn_ndarray::NdArrayDevice;
use reimagine_inference::{
    Backend, BackendInstance, CreateEmptyLatentRequest, CreateEmptyLatentResponse,
    DiffusionSampleRequest, DiffusionSampleResponse, ImageImportRequest, ImageImportResponse,
    ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse,
    InferenceBackend, InferenceBackendCapabilities, InferenceCapability, InferenceError,
    LatentDecodeRequest, LatentDecodeResponse, LatentEncodeRequest, LatentEncodeResponse,
    LoadBundleRequest, LoadBundleResponse, TextEncodeRequest, TextEncodeResponse,
};

use crate::config::BurnBackendConfig;
use crate::error::BurnBackendError;
use crate::profile::{BACKEND_LABEL, BurnProfileProvider};

#[derive(Debug, Clone)]
pub struct BurnBackend {
    config: BurnBackendConfig,
    device: NdArrayDevice,
}

impl BurnBackend {
    pub fn new(config: BurnBackendConfig) -> Result<Self, BurnBackendError> {
        let device = config.device().try_build_device()?;
        Ok(Self { config, device })
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

#[async_trait::async_trait]
impl InferenceBackend for BurnBackend {
    fn backend_kind(&self) -> &Backend {
        static KIND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
        KIND.get_or_init(BurnProfileProvider::backend_kind)
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        InferenceBackendCapabilities::new(self.backend_kind().clone())
    }

    async fn load_bundle(
        &self,
        _request: LoadBundleRequest,
    ) -> Result<LoadBundleResponse, InferenceError> {
        self.not_implemented(InferenceCapability::LoadBundle)
    }

    async fn text_encode(
        &self,
        _request: TextEncodeRequest,
    ) -> Result<TextEncodeResponse, InferenceError> {
        self.not_implemented(InferenceCapability::TextEncode)
    }

    async fn create_empty_latent(
        &self,
        _request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        self.not_implemented(InferenceCapability::CreateEmptyLatent)
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
