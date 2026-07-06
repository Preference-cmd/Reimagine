use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use burn_tensor::backend::Backend;
use reimagine_core::model::ModelId;

use crate::error::BurnBackendError;
use crate::models::stable_diffusion::sdxl::loaded::BurnLoadedSdxlBundle;
use crate::models::stable_diffusion::sdxl::text_conditioning::loading::load_text_encoder_modules;
use crate::models::stable_diffusion::sdxl::text_conditioning::module::SdxlTextEncoders;
use crate::runtime::BurnRuntime;

#[derive(Debug)]
pub(crate) struct SdxlTextEncoderCache<B: Backend> {
    entries: Mutex<HashMap<ModelId, SdxlTextEncoderCacheEntry<B>>>,
    _backend: PhantomData<B>,
}

#[derive(Debug, Clone)]
struct SdxlTextEncoderCacheEntry<B: Backend> {
    source_signature: crate::models::stable_diffusion::sdxl::BurnSdxlSourceSignature,
    encoders: Arc<SdxlTextEncoders<B>>,
}

impl<B: Backend> Default for SdxlTextEncoderCache<B> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> SdxlTextEncoderCache<B> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get_or_load(
        &self,
        runtime: &BurnRuntime<B>,
        bundle: &BurnLoadedSdxlBundle,
    ) -> Result<Arc<SdxlTextEncoders<B>>, BurnBackendError> {
        self.get_or_load_with(bundle, || load_text_encoder_modules(runtime, bundle))
    }

    fn get_or_load_with(
        &self,
        bundle: &BurnLoadedSdxlBundle,
        load: impl FnOnce() -> Result<SdxlTextEncoders<B>, BurnBackendError>,
    ) -> Result<Arc<SdxlTextEncoders<B>>, BurnBackendError> {
        let mut entries = self.entries.lock().expect("text encoder cache poisoned");
        if let Some(entry) = entries.get(bundle.model_id())
            && entry.source_signature == *bundle.source_signature()
        {
            return Ok(entry.encoders.clone());
        }

        let encoders = Arc::new(load()?);
        entries.insert(
            bundle.model_id().clone(),
            SdxlTextEncoderCacheEntry::<B> {
                source_signature: bundle.source_signature().clone(),
                encoders: encoders.clone(),
            },
        );
        Ok(encoders)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use reimagine_core::model::ModelId;
    use reimagine_inference::BackendPayloadKey;

    use crate::config::BurnBackendConfig;
    use crate::models::stable_diffusion::sdxl::component::BurnSdxlComponentRole;
    use crate::models::stable_diffusion::sdxl::loaded::BurnLoadedSdxlBundle;
    use crate::models::stable_diffusion::sdxl::text_conditioning::module::SdxlTextEncoders;
    use crate::runtime::BurnRuntime;
    use crate::text_encoder::clip::{ClipTextEncoderProfile, ClipTextEncoderVariant};

    #[test]
    fn get_or_load_reuses_module_for_same_bundle_signature() {
        let temp = tempfile::tempdir().expect("temp dir");
        let bundle = tiny_bundle(temp.path(), "first");
        let cache = super::SdxlTextEncoderCache::new();
        let module = tiny_active_text_encoders(&active_test_device());

        let first = cache
            .get_or_load_with(&bundle, || Ok(module.clone()))
            .expect("first load");
        let second = cache
            .get_or_load_with(&bundle, || {
                panic!("cache should not reload compatible bundle")
            })
            .expect("cached load");

        assert!(
            Arc::ptr_eq(&first, &second),
            "cache should reuse the same loaded text encoder modules"
        );
    }

    #[test]
    fn get_or_load_replaces_module_when_bundle_signature_changes() {
        let temp = tempfile::tempdir().expect("temp dir");
        let first_bundle = tiny_bundle(temp.path(), "first");
        let second_bundle = tiny_bundle(temp.path(), "second");
        let cache = super::SdxlTextEncoderCache::new();

        let first = cache
            .get_or_load_with(&first_bundle, || {
                Ok(tiny_active_text_encoders(&active_test_device()))
            })
            .expect("first load");
        let second = cache
            .get_or_load_with(&second_bundle, || {
                Ok(tiny_active_text_encoders(&active_test_device()))
            })
            .expect("second load");

        assert!(
            !Arc::ptr_eq(&first, &second),
            "cache must reload when the compatible bundle signature changes"
        );
    }

    #[test]
    fn get_or_load_accepts_active_runtime_backend() {
        let temp = tempfile::tempdir().expect("temp dir");
        let bundle = tiny_bundle(temp.path(), "active");
        let cache = super::SdxlTextEncoderCache::new();
        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<ActiveBurnBackend>::new(active_device(config.device()));

        let encoders: Arc<SdxlTextEncoders<ActiveBurnBackend>> = cache
            .get_or_load_with(&bundle, || Ok(tiny_active_text_encoders(runtime.device())))
            .expect("active backend text encoders");

        assert_eq!(encoders.clip_l.block_count(), 1);
    }

    fn active_test_device() -> burn_tensor::Device<ActiveBurnBackend> {
        let config = BurnBackendConfig::new("/models", "/output");
        active_device(config.device())
    }

    fn tiny_active_text_encoders(
        device: &burn_tensor::Device<ActiveBurnBackend>,
    ) -> SdxlTextEncoders<ActiveBurnBackend> {
        let profile = ClipTextEncoderProfile {
            variant: ClipTextEncoderVariant::ClipL,
            target_prefix: "test.text_encoder".to_string(),
            num_layers: 1,
            width: 2,
            heads: 1,
            inner_width: 8,
            vocab_size: 16,
            sequence_length: 5,
            produces_pooled_output: false,
        };
        let pooled_profile = ClipTextEncoderProfile {
            produces_pooled_output: true,
            variant: ClipTextEncoderVariant::OpenClipG,
            target_prefix: "test.text_encoder_2".to_string(),
            ..profile.clone()
        };
        SdxlTextEncoders::<ActiveBurnBackend>::init_from_profiles(&profile, &pooled_profile, device)
    }

    fn tiny_bundle(root: &Path, label: &str) -> BurnLoadedSdxlBundle {
        let primary_path = root.join(format!("{label}-text_encoder.safetensors"));
        let secondary_path = root.join(format!("{label}-text_encoder_2.safetensors"));
        std::fs::write(&primary_path, label).expect("primary component");
        std::fs::write(&secondary_path, label).expect("secondary component");
        BurnLoadedSdxlBundle::for_test_only(
            ModelId::new("unit-sdxl"),
            BackendPayloadKey::new("clip"),
        )
        .with_test_components(vec![
            (BurnSdxlComponentRole::TextEncoder, primary_path),
            (BurnSdxlComponentRole::TextEncoder2, secondary_path),
        ])
    }
}
