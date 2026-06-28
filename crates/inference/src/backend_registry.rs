//! Inference backend registry keyed by [`BackendInstance`].
//!
//! The registry stores configured backend instances in registration
//! order. The router's selection policy and affinity checks rely on
//! the deterministic order. Instances are addressable by their
//! `BackendInstance` identity and expose a [`BackendInstanceDescriptor`]
//! for diagnostics and selection.

use std::sync::Arc;

use crate::backend::InferenceBackend;
use crate::backend_selection::{Backend, BackendInstance, BackendInstanceDescriptor};
use crate::capability::InferenceBackendCapabilities;

/// One registered backend instance.
struct BackendRegistryEntry {
    instance: BackendInstance,
    backend: Arc<dyn InferenceBackend>,
    descriptor: BackendInstanceDescriptor,
}

impl std::fmt::Debug for BackendRegistryEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendRegistryEntry")
            .field("instance", &self.instance)
            .field("backend", &"<dyn InferenceBackend>")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

/// Registry that holds concrete backend instances in registration
/// order.
///
/// V1 keeps the registry in app-host and exposes it to the
/// executor-facing router through `DefaultInferenceRuntime`. App-host
/// chooses the order, the open backend label, and the descriptor
/// metadata; the registry stores the result verbatim.
#[derive(Default)]
pub struct InferenceBackendRegistry {
    entries: Vec<BackendRegistryEntry>,
}

impl InferenceBackendRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a backend instance.
    ///
    /// `descriptor` carries the open [`Backend`] label, optional
    /// device metadata, and optional plugin provenance. The instance
    /// identity is taken from the descriptor; `backend` is the
    /// concrete `InferenceBackend` implementation.
    pub fn register(
        &mut self,
        descriptor: BackendInstanceDescriptor,
        backend: Arc<dyn InferenceBackend>,
    ) {
        let instance = descriptor.instance.clone();
        self.entries.push(BackendRegistryEntry {
            instance,
            backend,
            descriptor,
        });
    }

    /// Look up a registered instance by its [`BackendInstance`]
    /// identity.
    pub fn get(&self, instance: &BackendInstance) -> Option<RegisteredInstance> {
        self.entries
            .iter()
            .find(|e| &e.instance == instance)
            .map(|e| RegisteredInstance {
                backend: Arc::clone(&e.backend),
                descriptor: e.descriptor.clone(),
            })
    }

    /// First registered instance. The router does not call this; it
    /// exists for diagnostic and test convenience.
    pub fn first(&self) -> Option<RegisteredInstance> {
        self.entries.first().map(|e| RegisteredInstance {
            backend: Arc::clone(&e.backend),
            descriptor: e.descriptor.clone(),
        })
    }

    /// Number of registered instances.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no backend is registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Registered instance identities in registration order.
    pub fn instances(&self) -> Vec<BackendInstance> {
        self.entries.iter().map(|e| e.instance.clone()).collect()
    }

    /// Registered instance descriptors in registration order.
    pub fn descriptors(&self) -> Vec<BackendInstanceDescriptor> {
        self.entries.iter().map(|e| e.descriptor.clone()).collect()
    }

    /// Registered instance descriptors, filtered to those whose
    /// open [`Backend`] label matches `backend`.
    pub fn descriptors_for_backend(&self, backend: &Backend) -> Vec<BackendInstanceDescriptor> {
        self.entries
            .iter()
            .filter(|e| &e.descriptor.backend == backend)
            .map(|e| e.descriptor.clone())
            .collect()
    }

    /// Merge every registered backend's capability report into a
    /// single combined report.
    pub fn merged_capabilities(&self) -> MergedInferenceBackendCapabilities {
        let mut merged = MergedInferenceBackendCapabilities::default();
        for entry in &self.entries {
            merged.add(entry.backend.capabilities());
        }
        merged
    }
}

/// A registered instance plus its descriptor.
#[derive(Clone)]
pub struct RegisteredInstance {
    pub backend: Arc<dyn InferenceBackend>,
    pub descriptor: BackendInstanceDescriptor,
}

impl std::fmt::Debug for RegisteredInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredInstance")
            .field("backend", &"<dyn InferenceBackend>")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl std::fmt::Debug for InferenceBackendRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceBackendRegistry")
            .field("instances", &self.instances())
            .field("count", &self.entries.len())
            .finish()
    }
}

/// Merged capability report across every registered backend.
///
/// V1 stores per-kind capabilities and exposes them as a flat
/// slice. Future per-capability aggregation can be added without
/// breaking the current shape.
#[derive(Debug, Clone, Default)]
pub struct MergedInferenceBackendCapabilities {
    by_kind: Vec<InferenceBackendCapabilities>,
}

impl MergedInferenceBackendCapabilities {
    pub fn add(&mut self, caps: InferenceBackendCapabilities) {
        self.by_kind.push(caps);
    }

    pub fn by_kind(&self) -> &[InferenceBackendCapabilities] {
        &self.by_kind
    }

    pub fn kind_count(&self) -> usize {
        self.by_kind.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{InferenceCapability, InferenceCapabilitySupport};
    use crate::inference_error::InferenceError;
    use crate::request::diffusion::DiffusionSampleRequest;
    use crate::request::image::{ImagePreviewRequest, ImageSaveRequest};
    use crate::request::image_import::ImageImportRequest;
    use crate::request::latent::{CreateEmptyLatentRequest, LatentDecodeRequest};
    use crate::request::latent_encode::LatentEncodeRequest;
    use crate::request::model::LoadBundleRequest;
    use crate::request::text::TextEncodeRequest;
    use crate::response::diffusion::DiffusionSampleResponse;
    use crate::response::image::{ImagePreviewResponse, ImageSaveResponse};
    use crate::response::image_import::ImageImportResponse;
    use crate::response::latent::{CreateEmptyLatentResponse, LatentDecodeResponse};
    use crate::response::latent_encode::LatentEncodeResponse;
    use crate::response::model::LoadBundleResponse;
    use crate::response::text::TextEncodeResponse;

    fn echo_caps() -> InferenceBackendCapabilities {
        InferenceBackendCapabilities::new(Backend::new("echo")).with_support(
            InferenceCapabilitySupport::new(InferenceCapability::CreateEmptyLatent),
        )
    }

    struct EchoBackend {
        kind: Backend,
    }

    impl EchoBackend {
        fn new() -> Self {
            Self {
                kind: Backend::new("echo"),
            }
        }
    }

    #[async_trait::async_trait]
    impl InferenceBackend for EchoBackend {
        fn backend_kind(&self) -> &Backend {
            &self.kind
        }
        fn capabilities(&self) -> InferenceBackendCapabilities {
            echo_caps()
        }
        async fn load_bundle(
            &self,
            _request: LoadBundleRequest,
        ) -> Result<LoadBundleResponse, InferenceError> {
            unimplemented!()
        }
        async fn text_encode(
            &self,
            _request: TextEncodeRequest,
        ) -> Result<TextEncodeResponse, InferenceError> {
            unimplemented!()
        }
        async fn create_empty_latent(
            &self,
            _request: CreateEmptyLatentRequest,
        ) -> Result<CreateEmptyLatentResponse, InferenceError> {
            unimplemented!()
        }
        async fn diffusion_sample(
            &self,
            _request: DiffusionSampleRequest,
        ) -> Result<DiffusionSampleResponse, InferenceError> {
            unimplemented!()
        }
        async fn latent_decode(
            &self,
            _request: LatentDecodeRequest,
        ) -> Result<LatentDecodeResponse, InferenceError> {
            unimplemented!()
        }
        async fn latent_encode(
            &self,
            _request: LatentEncodeRequest,
        ) -> Result<LatentEncodeResponse, InferenceError> {
            unimplemented!()
        }
        async fn image_import(
            &self,
            _request: ImageImportRequest,
        ) -> Result<ImageImportResponse, InferenceError> {
            unimplemented!()
        }
        async fn image_save(
            &self,
            _request: ImageSaveRequest,
        ) -> Result<ImageSaveResponse, InferenceError> {
            unimplemented!()
        }
        async fn image_preview(
            &self,
            _request: ImagePreviewRequest,
        ) -> Result<ImagePreviewResponse, InferenceError> {
            unimplemented!()
        }
    }

    #[test]
    fn register_and_lookup_by_instance() {
        let mut reg = InferenceBackendRegistry::new();
        let instance = BackendInstance::new("echo:main");
        let descriptor = BackendInstanceDescriptor::new(instance.clone(), Backend::new("echo"));
        reg.register(descriptor, Arc::new(EchoBackend::new()));
        assert!(reg.get(&instance).is_some());
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.instances(), vec![instance]);
    }

    #[test]
    fn first_returns_none_when_empty_and_some_when_registered() {
        let mut reg = InferenceBackendRegistry::new();
        assert!(reg.first().is_none());
        assert!(reg.is_empty());

        let instance = BackendInstance::new("echo:main");
        let descriptor = BackendInstanceDescriptor::new(instance, Backend::new("echo"));
        reg.register(descriptor, Arc::new(EchoBackend::new()));
        assert!(reg.first().is_some());
        assert!(!reg.is_empty());
    }

    #[test]
    fn descriptors_preserve_registration_order() {
        let mut reg = InferenceBackendRegistry::new();
        for label in ["a", "b", "c"] {
            let instance = BackendInstance::new(label);
            let descriptor = BackendInstanceDescriptor::new(instance, Backend::new(label));
            reg.register(descriptor, Arc::new(EchoBackend::new()));
        }
        let instances: Vec<String> = reg
            .instances()
            .into_iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(instances, vec!["a", "b", "c"]);
    }

    #[test]
    fn descriptors_for_backend_filters_by_label() {
        let mut reg = InferenceBackendRegistry::new();
        let a = BackendInstanceDescriptor::new(
            BackendInstance::new("candle:metal"),
            Backend::new("candle"),
        );
        let b =
            BackendInstanceDescriptor::new(BackendInstance::new("burn:cuda"), Backend::new("burn"));
        reg.register(a.clone(), Arc::new(EchoBackend::new()));
        reg.register(b.clone(), Arc::new(EchoBackend::new()));

        let candle = reg.descriptors_for_backend(&Backend::new("candle"));
        assert_eq!(candle.len(), 1);
        assert_eq!(candle[0].instance, a.instance);

        let burn = reg.descriptors_for_backend(&Backend::new("burn"));
        assert_eq!(burn.len(), 1);
        assert_eq!(burn[0].instance, b.instance);
    }

    #[test]
    fn merged_capabilities_collects_one_entry_per_kind() {
        let mut reg = InferenceBackendRegistry::new();
        let descriptor =
            BackendInstanceDescriptor::new(BackendInstance::new("echo:main"), Backend::new("echo"));
        reg.register(descriptor, Arc::new(EchoBackend::new()));
        let merged = reg.merged_capabilities();
        assert_eq!(merged.kind_count(), 1);
        assert_eq!(merged.by_kind().len(), 1);
    }
}
