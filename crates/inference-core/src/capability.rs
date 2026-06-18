//! Backend capability report.
//!
//! A backend publishes its capability report so the executor
//! registration layer and app-host can determine which operations
//! are supported before attempting to run a workflow.

use reimagine_core::BackendKind;
use reimagine_core::model::{ModelRole, ModelSeries, ModelVariant};

use crate::request::InferenceOperationId;

/// Describes which operations a backend supports and which model
/// families/variants those operations apply to.
#[derive(Debug, Clone)]
pub struct InferenceBackendCapabilities {
    backend_kind: BackendKind,
    operations: Vec<InferenceOperationSupport>,
}

impl InferenceBackendCapabilities {
    pub fn new(backend_kind: BackendKind) -> Self {
        Self {
            backend_kind,
            operations: Vec::new(),
        }
    }

    pub fn with_support(mut self, support: InferenceOperationSupport) -> Self {
        self.operations.push(support);
        self
    }

    pub fn backend_kind(&self) -> &BackendKind {
        &self.backend_kind
    }

    pub fn operations(&self) -> &[InferenceOperationSupport] {
        &self.operations
    }

    /// Returns `true` when the backend claims to support the given
    /// operation id regardless of model constraints.
    pub fn supports_operation(&self, operation_id: &InferenceOperationId) -> bool {
        self.operations
            .iter()
            .any(|op| &op.operation_id == operation_id)
    }
}

/// A single operation support entry. All fields except `operation_id`
/// are optional constraints; `None` means "all variants" or "all
/// series".
#[derive(Debug, Clone)]
pub struct InferenceOperationSupport {
    operation_id: InferenceOperationId,
    model_series: Option<ModelSeries>,
    variant: Option<ModelVariant>,
    roles: Vec<ModelRole>,
}

impl InferenceOperationSupport {
    pub fn new(operation_id: InferenceOperationId) -> Self {
        Self {
            operation_id,
            model_series: None,
            variant: None,
            roles: Vec::new(),
        }
    }

    pub fn with_model_series(mut self, series: ModelSeries) -> Self {
        self.model_series = Some(series);
        self
    }

    pub fn with_variant(mut self, variant: ModelVariant) -> Self {
        self.variant = Some(variant);
        self
    }

    pub fn with_role(mut self, role: ModelRole) -> Self {
        self.roles.push(role);
        self
    }

    pub fn operation_id(&self) -> &InferenceOperationId {
        &self.operation_id
    }

    pub fn model_series(&self) -> Option<&ModelSeries> {
        self.model_series.as_ref()
    }

    pub fn variant(&self) -> Option<&ModelVariant> {
        self.variant.as_ref()
    }

    pub fn roles(&self) -> &[ModelRole] {
        &self.roles
    }
}
