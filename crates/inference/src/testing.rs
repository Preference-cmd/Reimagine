//! Fake / stub backend for tests.
//!
//! [`FakeBackend`] is a minimal [`InferenceBackend`] implementation
//! that returns either a canned response or a deterministic
//! `BackendNotImplemented` error. The module is unconditionally
//! public so that downstream crate integration tests (which cannot
//! access `#[cfg(test)]` items from a dependency) can use it.
//!
//! Production code must not depend on `FakeBackend`. A future
//! refinement may gate the module behind a `testing` Cargo feature.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use reimagine_core::model::{ParamValue, SlotId};
use reimagine_runtime::RuntimeValue;

use crate::backend::InferenceBackend;
use crate::capability::{InferenceBackendCapabilities, InferenceOperationSupport};
use crate::error::InferenceError;
use crate::operation::InferenceOperationId;
use crate::request::InferenceRequest;
use crate::response::{InferenceOutput, InferenceResponse};

/// A canned response for a specific operation id.
#[derive(Debug, Clone)]
pub struct FakeOperationResponse {
    pub outputs: Vec<(SlotId, Arc<RuntimeValue>)>,
}

impl FakeOperationResponse {
    pub fn new(outputs: Vec<(SlotId, Arc<RuntimeValue>)>) -> Self {
        Self { outputs }
    }
}

/// A minimal fake backend for tests.
///
/// Operations are registered via [`with_operation`](FakeBackend::with_operation)
/// or [`insert_operation`](FakeBackend::insert_operation). Any
/// operation not registered returns `BackendNotImplemented`.
///
/// # Example
///
/// ```
/// use reimagine_inference::{FakeBackend, InferenceBackend};
/// use reimagine_inference::operation::OP_LATENT_CREATE_EMPTY;
///
/// let backend = FakeBackend::new("fake")
///     .with_operation(OP_LATENT_CREATE_EMPTY, vec![]);
/// assert!(InferenceBackend::capabilities(&backend)
///     .supports_operation(&OP_LATENT_CREATE_EMPTY.into()));
/// ```
pub struct FakeBackend {
    kind: String,
    operations: Mutex<HashMap<InferenceOperationId, FakeOperationResponse>>,
}

impl FakeBackend {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            operations: Mutex::new(HashMap::new()),
        }
    }

    /// Register a canned response for an operation id.
    pub fn with_operation(
        self,
        operation_id: impl Into<InferenceOperationId>,
        outputs: Vec<(SlotId, Arc<RuntimeValue>)>,
    ) -> Self {
        self.insert_operation(operation_id, outputs);
        self
    }

    /// Insert a canned response at runtime (takes `&self`).
    pub fn insert_operation(
        &self,
        operation_id: impl Into<InferenceOperationId>,
        outputs: Vec<(SlotId, Arc<RuntimeValue>)>,
    ) {
        let mut ops = self.operations.lock().expect("fake backend poisoned");
        ops.insert(operation_id.into(), FakeOperationResponse::new(outputs));
    }
}

#[async_trait::async_trait]
impl InferenceBackend for FakeBackend {
    fn backend_kind(&self) -> &str {
        &self.kind
    }

    fn capabilities(&self) -> InferenceBackendCapabilities {
        let ops = self.operations.lock().expect("fake backend poisoned");
        let mut caps = InferenceBackendCapabilities::new(&self.kind);
        for op_id in ops.keys() {
            caps = caps.with_support(InferenceOperationSupport::new(op_id.clone()));
        }
        caps
    }

    async fn execute(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, InferenceError> {
        let ops = self.operations.lock().expect("fake backend poisoned");
        match ops.get(request.operation_id()) {
            Some(canned) => Ok(InferenceResponse::new(
                canned
                    .outputs
                    .iter()
                    .map(|(slot, val)| InferenceOutput::new(slot.clone(), val.clone()))
                    .collect(),
            )),
            None => Err(InferenceError::BackendNotImplemented {
                operation_id: request.operation_id().to_string(),
                backend_kind: self.kind.clone(),
            }),
        }
    }
}

impl std::fmt::Debug for FakeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ops = self.operations.lock().expect("fake backend poisoned");
        f.debug_struct("FakeBackend")
            .field("kind", &self.kind)
            .field("registered_operations", &ops.len())
            .finish()
    }
}

/// Helper to build a simple `Arc<RuntimeValue>` output pair for tests.
pub fn fake_output(slot: &str, value: impl Into<ParamValue>) -> (SlotId, Arc<RuntimeValue>) {
    (
        SlotId::new(slot),
        Arc::new(RuntimeValue::Param(value.into())),
    )
}
