//! Inference backend registry keyed by core-owned `BackendKind`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::InferenceBackend;
use crate::capability::InferenceBackendCapabilities;
use reimagine_core::BackendKind;

/// Registry that holds concrete backends and dispatches by `kind`.
///
/// `kind` is the stable backend identifier (e.g. `"candle"`,
/// `"remote"`). V1 keeps the registry in app-host and exposes it
/// to the executor-facing router through `DefaultInferenceRuntime`.
#[derive(Default)]
pub struct InferenceBackendRegistry {
    backends: HashMap<BackendKind, Arc<dyn InferenceBackend>>,
}

impl InferenceBackendRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a backend. The backend's `backend_kind()` becomes its
    /// registry key. Re-registering under the same key replaces the
    /// previous entry.
    pub fn register(&mut self, backend: Arc<dyn InferenceBackend>) {
        let kind = backend.backend_kind().clone();
        self.backends.insert(kind, backend);
    }

    /// Look up a backend by `kind`.
    pub fn get(&self, kind: &BackendKind) -> Option<Arc<dyn InferenceBackend>> {
        self.backends.get(kind).cloned()
    }

    /// First registered backend. V1 single-backend workspaces rely on
    /// this; per-request backend selection is follow-up work.
    pub fn first(&self) -> Option<Arc<dyn InferenceBackend>> {
        self.backends.values().next().cloned()
    }

    /// Number of registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Returns `true` when no backend is registered.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// All registered backend kinds in insertion order.
    pub fn kinds(&self) -> Vec<BackendKind> {
        self.backends.keys().cloned().collect()
    }

    /// Merge every registered backend's capability report into a
    /// single combined report.
    pub fn merged_capabilities(&self) -> MergedInferenceBackendCapabilities {
        let mut merged = MergedInferenceBackendCapabilities::default();
        for backend in self.backends.values() {
            merged.add(backend.capabilities());
        }
        merged
    }
}

impl std::fmt::Debug for InferenceBackendRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceBackendRegistry")
            .field("kinds", &self.kinds())
            .field("count", &self.backends.len())
            .finish()
    }
}

/// Merged capability report across every registered backend.
///
/// V1 stores per-kind capabilities and exposes them as a flat
/// slice. Future per-operation aggregation can be added without
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
    use crate::capability::InferenceOperationSupport;
    use crate::error::InferenceError;
    use crate::request::InferenceRequest;
    use crate::request::OP_LATENT_CREATE_EMPTY;
    use crate::response::InferenceResponse;
    use reimagine_core::BackendKind;

    fn echo_caps() -> InferenceBackendCapabilities {
        InferenceBackendCapabilities::new(BackendKind::new("echo")).with_support(
            InferenceOperationSupport::new(OP_LATENT_CREATE_EMPTY.into()),
        )
    }

    struct EchoBackend;

    #[async_trait::async_trait]
    impl InferenceBackend for EchoBackend {
        fn backend_kind(&self) -> &BackendKind {
            static KIND: std::sync::OnceLock<BackendKind> = std::sync::OnceLock::new();
            KIND.get_or_init(|| BackendKind::new("echo"))
        }
        fn capabilities(&self) -> InferenceBackendCapabilities {
            echo_caps()
        }
        async fn execute(
            &self,
            _request: InferenceRequest,
        ) -> Result<InferenceResponse, InferenceError> {
            Ok(InferenceResponse::new(vec![]))
        }
    }

    #[test]
    fn register_and_lookup_by_kind() {
        let mut reg = InferenceBackendRegistry::new();
        reg.register(Arc::new(EchoBackend));
        assert!(reg.get(&BackendKind::new("echo")).is_some());
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.kinds(), vec![BackendKind::new("echo")]);
    }

    #[test]
    fn first_returns_none_when_empty_and_some_when_registered() {
        let mut reg = InferenceBackendRegistry::new();
        assert!(reg.first().is_none());
        assert!(reg.is_empty());
        reg.register(Arc::new(EchoBackend));
        assert!(reg.first().is_some());
        assert!(!reg.is_empty());
    }

    #[test]
    fn merged_capabilities_collects_one_entry_per_kind() {
        let mut reg = InferenceBackendRegistry::new();
        reg.register(Arc::new(EchoBackend));
        let merged = reg.merged_capabilities();
        assert_eq!(merged.kind_count(), 1);
        assert_eq!(merged.by_kind().len(), 1);
    }
}
