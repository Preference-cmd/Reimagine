//! Producer-declared execution output contract.
//!
//! [`ExecutionOutput`] pairs an [`ExecutionValue`] with the retention
//! policy the producer intends. Runtime stores the retention alongside
//! the value (see `reimagine_runtime::RunValueStore`) and uses it for
//! future last-use analysis in issue 05.
//!
//! The retention policy is part of the executor contract. The
//! producer — typically a built-in `inference` executor — knows whether
//! the value is single-use, run-scoped, or workspace-scoped, and the
//! runtime should not have to guess from the DAG shape alone.

use std::sync::Arc;

use reimagine_core::model::SlotId;

use super::value::ExecutionValue;

/// Retention category declared by the producer of an [`ExecutionValue`].
///
/// Runtime uses the category to decide when the value (and the backend
/// payload it references) can be released. The exact release behavior is
/// owned by issue 05; for now the runtime stores the policy next to the
/// value and continues to hold values for the full run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionValueRetention {
    /// The value is consumed exactly once by the next downstream node.
    ///
    /// Single-use values should be released as soon as the downstream
    /// node that consumes them completes. Single-use fan-out diagnostics
    /// are intentionally out of scope for issue 02; the runtime stores
    /// the policy but does not act on it yet.
    SingleUse,
    /// The value lives for the duration of a single run.
    ///
    /// This is the V1 default for most executor outputs (latents,
    /// images, conditioning, params).
    RunScoped,
    /// The value lives across runs within the same workspace.
    ///
    /// In V1 only checkpoint loader outputs (model, clip, vae handles)
    /// carry this category. The runtime keeps them alive until the run
    /// completes; future workspace-scoped lifecycle behavior is owned by
    /// later issues.
    WorkspaceScoped,
}

impl ExecutionValueRetention {
    /// Stable string label for diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SingleUse => "single_use",
            Self::RunScoped => "run_scoped",
            Self::WorkspaceScoped => "workspace_scoped",
        }
    }
}

impl std::fmt::Display for ExecutionValueRetention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Producer-declared execution output.
///
/// Bundles the produced [`ExecutionValue`] with the [`SlotId`] it should
/// be stored under and the [`ExecutionValueRetention`] policy that tells
/// runtime how long the value (and its backend payload) should live.
///
/// Executors return `Vec<ExecutionOutput>` from `NodeExecutor::execute`.
/// The runner inserts each output into the per-run value store and
/// records the retention policy for later lifecycle use.
#[derive(Debug, Clone)]
pub struct ExecutionOutput {
    slot_id: SlotId,
    value: Arc<ExecutionValue>,
    retention: ExecutionValueRetention,
}

impl ExecutionOutput {
    /// Build a new output with the given slot, value, and retention.
    pub fn new(
        slot_id: SlotId,
        value: Arc<ExecutionValue>,
        retention: ExecutionValueRetention,
    ) -> Self {
        Self {
            slot_id,
            value,
            retention,
        }
    }

    /// Convenience: build a `RunScoped` output (the V1 default for most
    /// executor outputs).
    pub fn run_scoped(slot_id: SlotId, value: Arc<ExecutionValue>) -> Self {
        Self::new(slot_id, value, ExecutionValueRetention::RunScoped)
    }

    /// Convenience: build a `SingleUse` output.
    pub fn single_use(slot_id: SlotId, value: Arc<ExecutionValue>) -> Self {
        Self::new(slot_id, value, ExecutionValueRetention::SingleUse)
    }

    /// Convenience: build a `WorkspaceScoped` output (used by the
    /// checkpoint loader for model/clip/vae handles).
    pub fn workspace_scoped(slot_id: SlotId, value: Arc<ExecutionValue>) -> Self {
        Self::new(slot_id, value, ExecutionValueRetention::WorkspaceScoped)
    }

    /// Output slot id declared by the executor.
    pub fn slot_id(&self) -> &SlotId {
        &self.slot_id
    }

    /// Borrow the underlying execution value.
    pub fn value(&self) -> &Arc<ExecutionValue> {
        &self.value
    }

    /// Consume the output and return the inner `Arc<ExecutionValue>`.
    pub fn into_value(self) -> Arc<ExecutionValue> {
        self.value
    }

    /// Declared retention policy.
    pub fn retention(&self) -> ExecutionValueRetention {
        self.retention
    }
}
