use std::sync::Arc;

use reimagine_inference::{
    Backend, BackendInstance, CreateEmptyLatentRequest, CreateEmptyLatentResponse,
    DiffusionSampleRequest, DiffusionSampleResponse, ImageImportRequest, ImageImportResponse,
    ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse,
    InferenceBackend, InferenceBackendCapabilities, InferenceCapability,
    InferenceCapabilitySupport, InferenceError, LatentDecodeRequest, LatentDecodeResponse,
    LatentEncodeRequest, LatentEncodeResponse, LoadBundleRequest, LoadBundleResponse,
    TextEncodeRequest, TextEncodeResponse,
};

use crate::active_backend::{ActiveBurnBackend, active_device};
use crate::config::BurnBackendConfig;
use crate::device::BurnDevice;
use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::text_conditioning::cache::SdxlTextEncoderCache;
use crate::operation::{
    execute_diffusion_sample, execute_image_preview, execute_image_save,
    execute_latent_create_empty, execute_latent_decode, execute_model_load_bundle,
    execute_text_encode, map_to_inference_error,
};
use crate::profile::{BACKEND_LABEL, BurnProfileProvider};
use crate::resource::BurnBackendInstanceRuntimeHooks;
use crate::runtime::BurnRuntime;
use crate::store::{BurnModelCache, BurnStore};

#[derive(Debug, Clone)]
pub struct BurnBackend {
    config: BurnBackendConfig,
    device: BurnDevice,
    store: Arc<BurnStore>,
    model_cache: Arc<BurnModelCache>,
    active_runtime: Arc<BurnRuntime<ActiveBurnBackend>>,
    text_encoder_cache: Arc<SdxlTextEncoderCache<ActiveBurnBackend>>,
}

impl BurnBackend {
    pub fn new(config: BurnBackendConfig) -> Result<Self, BurnBackendError> {
        // The config layer stores a `BurnDevice` already
        // constructed from the user's label string, so we just
        // clone the resolved variant. Validating
        // `try_build_device` is exposed as an associated
        // function for callers that want precise errors for
        // unknown labels — config validation runs through the
        // `new` constructor and a generic `cpu` fallback.
        let device = config.device().clone();
        let active_device = active_device(&device);
        Ok(Self {
            config,
            device,
            store: Arc::new(BurnStore::new()),
            model_cache: Arc::new(BurnModelCache::new()),
            active_runtime: Arc::new(BurnRuntime::new(active_device)),
            text_encoder_cache: Arc::new(SdxlTextEncoderCache::new()),
        })
    }

    pub fn config(&self) -> &BurnBackendConfig {
        &self.config
    }

    pub fn device(&self) -> &BurnDevice {
        &self.device
    }

    pub fn device_label(&self) -> &str {
        // The profile advertises one backend instance per
        // active feature-device combination (e.g.,
        // `burn:wgpu:default`, `burn:flex:cpu`). `device_label`
        // returns the short suffix used to construct the
        // `burn:<label>` instance.
        self.device.label()
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

    #[allow(dead_code)]
    pub(crate) fn active_runtime(&self) -> &Arc<BurnRuntime<ActiveBurnBackend>> {
        &self.active_runtime
    }

    pub(crate) fn text_encoder_cache(&self) -> &Arc<SdxlTextEncoderCache<ActiveBurnBackend>> {
        &self.text_encoder_cache
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
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::DiffusionSample,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::LatentDecode,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::ImageImport,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::ImageSave,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::ImagePreview,
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
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        map_err(execute_diffusion_sample(self, request))
    }

    async fn latent_decode(
        &self,
        request: LatentDecodeRequest,
    ) -> Result<LatentDecodeResponse, InferenceError> {
        map_err(execute_latent_decode(self, request))
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
        request: ImageSaveRequest,
    ) -> Result<ImageSaveResponse, InferenceError> {
        execute_image_save(request, self).map_err(|err| InferenceError::BackendExecutionFailed {
            message: err.to_string(),
        })
    }

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        execute_image_preview(request, self).map_err(|err| InferenceError::BackendExecutionFailed {
            message: err.to_string(),
        })
    }
}
