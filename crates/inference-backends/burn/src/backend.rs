use std::sync::Arc;

use reimagine_inference::{
    Backend, BackendInstance, CreateEmptyLatentRequest, CreateEmptyLatentResponse,
    DiffusionSampleRequest, DiffusionSampleResponse, ImageImportRequest, ImageImportResponse,
    ImagePreviewRequest, ImagePreviewResponse, ImageSaveRequest, ImageSaveResponse,
    InferenceBackend, InferenceBackendCapabilities, InferenceCapability,
    InferenceCapabilitySupport, InferenceError, LatentDecodeRequest, LatentDecodeResponse,
    LatentEncodeRequest, LatentEncodeResponse, LoadBundleRequest, LoadBundleResponse,
    ResourceHintSink, ResourceHints, TextEncodeRequest, TextEncodeResponse,
};
use tokio_util::sync::CancellationToken;

use crate::active_backend::{ActiveBurnBackend, active_device};
use crate::config::BurnBackendConfig;
use crate::device::BurnDevice;
use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::BurnSdxlComponentRole;
use crate::models::stable_diffusion::sdxl::text_conditioning::cache::SdxlTextEncoderCache;
use crate::operation::{
    execute_diffusion_sample, execute_image_preview, execute_image_save,
    execute_latent_create_empty, execute_latent_decode, execute_model_load_bundle,
    execute_text_encode,
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

    /// Backend-wide cancellation token, shared with the serve loop.
    pub fn cancellation(&self) -> Arc<CancellationToken> {
        self.active_runtime.cancellation().clone()
    }

    /// Whether the current operation should abort. A request-scoped
    /// token installed via [`crate::with_request_cancellation`] takes
    /// precedence over the backend-wide token.
    pub fn is_cancelled(&self) -> bool {
        crate::cancellation::is_cancelled(self.active_runtime.cancellation())
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

    /// Capabilities that never depend on loaded model state.
    fn static_capabilities(&self) -> InferenceBackendCapabilities {
        InferenceBackendCapabilities::new(self.backend_kind().clone())
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::LoadBundle,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::CreateEmptyLatent,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::ImageSave,
            ))
            .with_support(InferenceCapabilitySupport::new(
                InferenceCapability::ImagePreview,
            ))
    }

    /// Capability report narrowed to what the model cache actually
    /// has loaded. Component-dependent capabilities are only
    /// advertised once the corresponding model component is cached;
    /// the report never advertises more than the static base set.
    fn dynamic_capabilities(&self) -> InferenceBackendCapabilities {
        let mut caps = self.static_capabilities();
        let cache = self.model_cache();
        if cache.has_component_role(BurnSdxlComponentRole::TextEncoder)
            || cache.has_component_role(BurnSdxlComponentRole::TextEncoder2)
        {
            caps = caps.with_support(InferenceCapabilitySupport::new(
                InferenceCapability::TextEncode,
            ));
        }
        if cache.has_component_role(BurnSdxlComponentRole::Diffusion) {
            caps = caps.with_support(InferenceCapabilitySupport::new(
                InferenceCapability::DiffusionSample,
            ));
        }
        if cache.has_component_role(BurnSdxlComponentRole::Vae) {
            caps = caps.with_support(InferenceCapabilitySupport::new(
                InferenceCapability::LatentDecode,
            ));
        }
        caps
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
    result.map_err(burn_error_to_inference_error)
}

/// Map a [`BurnBackendError`] into [`InferenceError`], preserving
/// structured variant information so callers can distinguish error
/// types programmatically.
fn burn_error_to_inference_error(err: BurnBackendError) -> InferenceError {
    match err {
        BurnBackendError::DeviceUnavailable { requested, reason } => {
            InferenceError::DeviceUnavailable {
                device: format!("{requested}: {reason}"),
            }
        }
        BurnBackendError::MissingComponent(component) => InferenceError::ModelNotLoaded {
            model_id: component,
        },
        BurnBackendError::ComponentValidation { path, source } => {
            InferenceError::ComponentValidation {
                component: path.display().to_string(),
                reason: source.to_string(),
            }
        }
        BurnBackendError::Tokenizer(error) => InferenceError::TokenizationFailed {
            message: error.to_string(),
        },
        BurnBackendError::CacheIncompatible(message) => {
            InferenceError::ModelNotLoaded { model_id: message }
        }
        BurnBackendError::Cancelled => InferenceError::Cancelled,
        other => InferenceError::BackendExecutionFailed {
            message: other.to_string(),
        },
    }
}

#[async_trait::async_trait]
impl InferenceBackend for BurnBackend {
    fn backend_kind(&self) -> &Backend {
        static KIND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
        KIND.get_or_init(BurnProfileProvider::backend_kind)
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        self.dynamic_capabilities()
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
        execute_text_encode(self, request).map_err(burn_error_to_inference_error)
    }

    async fn create_empty_latent(
        &self,
        request: CreateEmptyLatentRequest,
    ) -> Result<CreateEmptyLatentResponse, InferenceError> {
        execute_latent_create_empty(self, request).map_err(burn_error_to_inference_error)
    }

    async fn diffusion_sample(
        &self,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        map_err(execute_diffusion_sample(self, request, None))
    }

    async fn diffusion_sample_with_invocation(
        &self,
        invocation: &reimagine_inference::InferenceInvocation,
        request: DiffusionSampleRequest,
    ) -> Result<DiffusionSampleResponse, InferenceError> {
        self.admit_invocation(invocation)?;
        let result = execute_diffusion_sample(self, request, Some(invocation.progress().as_ref()));
        self.finish_invocation(invocation);
        map_err(result)
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
        execute_image_save(request, self).map_err(burn_error_to_inference_error)
    }

    async fn image_preview(
        &self,
        request: ImagePreviewRequest,
    ) -> Result<ImagePreviewResponse, InferenceError> {
        execute_image_preview(request, self).map_err(burn_error_to_inference_error)
    }

    // Override InferenceBackend defaults to delegate to same internal logic
    // as ResourceHintSink, so both trait paths work.

    async fn apply_resource_hints(&self, hints: ResourceHints) -> Result<(), InferenceError> {
        <Self as ResourceHintSink>::apply_resource_hints(self, hints).await
    }

    fn current_vram_usage(&self) -> Option<u64> {
        <Self as ResourceHintSink>::current_vram_usage(self)
    }

    fn loaded_model_count(&self) -> usize {
        <Self as ResourceHintSink>::loaded_model_count(self)
    }
}

#[async_trait::async_trait]
impl ResourceHintSink for BurnBackend {
    async fn apply_resource_hints(&self, hints: ResourceHints) -> Result<(), InferenceError> {
        tracing::debug!(
            run_id = %hints.run_id,
            vram_budget = ?hints.vram_budget,
            prefetch_models = hints.prefetch.next_model_ids.len(),
            lifecycle_entries = hints.component_lifecycle.len(),
            "applying resource hints to BurnBackend"
        );

        // Forward VRAM budget to model cache for LRU eviction
        if let Some(budget) = hints.vram_budget {
            self.model_cache.apply_vram_budget(budget);
        }

        Ok(())
    }

    fn current_vram_usage(&self) -> Option<u64> {
        let model_bytes = self.model_cache.total_byte_size();
        let payload_bytes = self.store.payload_byte_size() as u64;
        Some(model_bytes + payload_bytes)
    }

    fn loaded_model_count(&self) -> usize {
        self.model_cache.bundle_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::stable_diffusion::sdxl::{
        BurnLoadedModelBundle, BurnLoadedSdxlBundle, BurnSdxlComponentRole,
    };
    use reimagine_core::model::ModelId;
    use reimagine_inference::BackendPayloadKey;
    use std::path::PathBuf;

    fn backend() -> BurnBackend {
        BurnBackend::new(BurnBackendConfig::new("/models", "/output")).expect("burn backend")
    }

    fn clip_only_bundle(model_id: &str) -> Arc<BurnLoadedModelBundle> {
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new(model_id),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![
            (
                BurnSdxlComponentRole::TextEncoder,
                PathBuf::from("text_encoder/model.safetensors"),
            ),
            (
                BurnSdxlComponentRole::TextEncoder2,
                PathBuf::from("text_encoder_2/model.safetensors"),
            ),
        ]);
        Arc::new(BurnLoadedModelBundle::StableDiffusionSdxl(Arc::new(bundle)))
    }

    fn full_bundle(model_id: &str) -> Arc<BurnLoadedModelBundle> {
        let bundle = BurnLoadedSdxlBundle::for_test_only(
            ModelId::new(model_id),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![
            (
                BurnSdxlComponentRole::TextEncoder,
                PathBuf::from("text_encoder/model.safetensors"),
            ),
            (
                BurnSdxlComponentRole::TextEncoder2,
                PathBuf::from("text_encoder_2/model.safetensors"),
            ),
            (
                BurnSdxlComponentRole::Diffusion,
                PathBuf::from("unet/model.safetensors"),
            ),
            (
                BurnSdxlComponentRole::Vae,
                PathBuf::from("vae/model.safetensors"),
            ),
        ]);
        Arc::new(BurnLoadedModelBundle::StableDiffusionSdxl(Arc::new(bundle)))
    }

    #[test]
    fn capabilities_empty_cache_advertises_only_component_independent_ops() {
        let caps = backend().capabilities();
        assert!(caps.supports_capability(InferenceCapability::LoadBundle));
        assert!(caps.supports_capability(InferenceCapability::CreateEmptyLatent));
        assert!(caps.supports_capability(InferenceCapability::ImageSave));
        assert!(caps.supports_capability(InferenceCapability::ImagePreview));
        assert!(!caps.supports_capability(InferenceCapability::TextEncode));
        assert!(!caps.supports_capability(InferenceCapability::DiffusionSample));
        assert!(!caps.supports_capability(InferenceCapability::LatentDecode));
    }

    #[test]
    fn capabilities_clip_only_bundle_advertises_text_encode_only() {
        let backend = backend();
        backend
            .model_cache()
            .insert_bundle(ModelId::new("clip-only"), clip_only_bundle("clip-only"));

        let caps = backend.capabilities();
        assert!(caps.supports_capability(InferenceCapability::TextEncode));
        assert!(!caps.supports_capability(InferenceCapability::DiffusionSample));
        assert!(!caps.supports_capability(InferenceCapability::LatentDecode));
    }

    #[test]
    fn capabilities_full_bundle_advertises_component_dependent_ops() {
        let backend = backend();
        backend
            .model_cache()
            .insert_bundle(ModelId::new("sdxl"), full_bundle("sdxl"));

        let caps = backend.capabilities();
        assert!(caps.supports_capability(InferenceCapability::TextEncode));
        assert!(caps.supports_capability(InferenceCapability::DiffusionSample));
        assert!(caps.supports_capability(InferenceCapability::LatentDecode));
    }
}
