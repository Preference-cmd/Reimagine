//! Backend-neutral resource management hints.
//!
//! The runtime sends [`ResourceHints`] to backends before each stage
//! to communicate VRAM budgets, prefetch needs, and component lifecycle
//! intent. Backends that do not implement resource hints continue to
//! work via default trait method implementations.

use reimagine_core::model::{ComponentRole, ModelId, RunId};

/// VRAM budget for a run or stage, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VramBudget {
    /// Total VRAM budget in bytes. `None` means "use all available".
    pub total_bytes: Option<u64>,
    /// Reserved VRAM in bytes (for OS, other apps).
    pub reserved_bytes: u64,
}

impl VramBudget {
    pub fn unlimited() -> Self {
        Self {
            total_bytes: None,
            reserved_bytes: 0,
        }
    }

    pub fn with_total_bytes(mut self, bytes: u64) -> Self {
        self.total_bytes = Some(bytes);
        self
    }

    pub fn with_reserved_bytes(mut self, bytes: u64) -> Self {
        self.reserved_bytes = bytes;
        self
    }
}

/// Hint for what the next stage will need, enabling prefetch.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PrefetchHint {
    /// Model IDs the next stage will need loaded.
    pub next_model_ids: Vec<ModelId>,
    /// Component roles the next stage will need.
    pub next_component_roles: Vec<ComponentRole>,
}

/// Component lifecycle action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ComponentLifecycleAction {
    /// Keep the component loaded in memory for the next stage.
    KeepResident,
    /// The component can be evicted after this stage completes.
    Deactivate,
    /// The component must be loaded now (prefetch).
    Activate,
}

/// Resource hints sent from runtime to backend before each stage.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceHints {
    pub run_id: RunId,
    pub vram_budget: Option<VramBudget>,
    pub prefetch: PrefetchHint,
    /// Component lifecycle hints keyed by model ID.
    pub component_lifecycle: Vec<ComponentLifecycleEntry>,
}

impl ResourceHints {
    pub fn new(run_id: RunId) -> Self {
        Self {
            run_id,
            vram_budget: None,
            prefetch: PrefetchHint::default(),
            component_lifecycle: Vec::new(),
        }
    }

    pub fn with_vram_budget(mut self, budget: VramBudget) -> Self {
        self.vram_budget = Some(budget);
        self
    }

    pub fn with_prefetch(mut self, prefetch: PrefetchHint) -> Self {
        self.prefetch = prefetch;
        self
    }

    pub fn with_component_lifecycle(mut self, entries: Vec<ComponentLifecycleEntry>) -> Self {
        self.component_lifecycle = entries;
        self
    }
}

/// Entry for component lifecycle hints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentLifecycleEntry {
    pub model_id: ModelId,
    pub action: ComponentLifecycleAction,
}

impl ComponentLifecycleEntry {
    pub fn new(model_id: ModelId, action: ComponentLifecycleAction) -> Self {
        Self { model_id, action }
    }
}

/// Extension trait for backends that support resource hints.
///
/// Backends that do not implement this trait continue to work via
/// the default method implementations on [`InferenceBackend`](crate::InferenceBackend).
#[async_trait::async_trait]
pub trait ResourceHintSink: Send + Sync {
    /// Apply resource hints for the upcoming stage.
    async fn apply_resource_hints(
        &self,
        hints: ResourceHints,
    ) -> Result<(), crate::inference_error::InferenceError>;

    /// Query current VRAM usage (bytes) for this backend instance.
    fn current_vram_usage(&self) -> Option<u64> {
        None
    }

    /// Query the number of loaded model bundles.
    fn loaded_model_count(&self) -> usize {
        0
    }
}
