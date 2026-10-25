//! Known request operations carried by an `ExecuteRequest`.
//!
//! Operations are identified by name and carry a JSON payload whose
//! shape is defined by the `reimagine-inference` serde types. New
//! capabilities are added as operations (a name plus a payload shape)
//! rather than as new RPCs, so the protobuf contract stays stable.

pub mod request_operation {
    /// Apply resource-management hints (VRAM budget, prefetch intent,
    /// component lifecycle) for the upcoming stage.
    ///
    /// Payload is a JSON-serialized
    /// `reimagine_inference::ResourceHints`. The operation is
    /// advisory: backends that do not support it respond with an
    /// `unknown_operation` backend error and the caller must continue.
    pub const APPLY_RESOURCE_HINTS: &str = "resource.apply_hints";

    /// Set the cross-run model cache VRAM budget.
    ///
    /// Payload is a JSON-serialized `reimagine_inference::VramBudget`.
    /// The worker forwards it to `BurnModelCache::apply_vram_budget`,
    /// evicting least-recently-used bundles to fit the budget. Unlike
    /// `resource.apply_hints` (advisory, stage-scoped), this operation
    /// is the explicit cache-control path used on worker drain/switch
    /// and run handoff.
    pub const SET_MODEL_CACHE_BUDGET: &str = "model.set_cache_budget";
}
