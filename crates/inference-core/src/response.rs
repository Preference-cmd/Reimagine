//! Backend-neutral inference response.
//!
//! [`InferenceResponse`] is the return type of
//! [`InferenceBackend::execute`](crate::backend::InferenceBackend::execute).
//! It carries slot-aware outputs that mirror the runtime's
//! `NodeExecutionOutputs` shape so the executor adapter can return
//! them directly.

use std::sync::Arc;

use reimagine_core::ExecutionValue;
use reimagine_core::model::SlotId;

/// A single named output from an inference operation.
///
/// Output order must not matter; executors and runtime treat outputs
/// as a map keyed by `SlotId`. Backends should return outputs in
/// whatever order is natural for their implementation.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    slot_id: SlotId,
    value: Arc<ExecutionValue>,
}

impl InferenceOutput {
    pub fn new(slot_id: impl Into<SlotId>, value: Arc<ExecutionValue>) -> Self {
        Self {
            slot_id: slot_id.into(),
            value,
        }
    }

    pub fn slot_id(&self) -> &SlotId {
        &self.slot_id
    }

    pub fn value(&self) -> &Arc<ExecutionValue> {
        &self.value
    }

    /// Consume the output and return its parts.
    pub fn into_parts(self) -> (SlotId, Arc<ExecutionValue>) {
        (self.slot_id, self.value)
    }
}

/// Backend-neutral inference response.
///
/// The response is a list of slot-keyed outputs. It does not carry
/// backend-native payload types; all values are
/// [`ExecutionValue`](reimagine_core::ExecutionValue) handles.
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    outputs: Vec<InferenceOutput>,
}

impl InferenceResponse {
    pub fn new(outputs: Vec<InferenceOutput>) -> Self {
        Self { outputs }
    }

    pub fn outputs(&self) -> &[InferenceOutput] {
        &self.outputs
    }

    /// Consume the response and return the raw output list.
    pub fn into_outputs(self) -> Vec<InferenceOutput> {
        self.outputs
    }

    /// Convert the response into the runtime's
    /// `NodeExecutionOutputs` shape.
    pub fn into_node_outputs(self) -> Vec<(SlotId, Arc<ExecutionValue>)> {
        self.outputs
            .into_iter()
            .map(InferenceOutput::into_parts)
            .collect()
    }
}
