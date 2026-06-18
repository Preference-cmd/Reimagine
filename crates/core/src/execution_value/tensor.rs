//! Tensor metadata that flows on a public execution value.
//!
//! The actual tensor payload lives in backend-owned stores; only the
//! `dtype` / `shape` / `device_label` triple is part of the public
//! value envelope. `TensorDType` and `TensorShape` are reused from the
//! `core::model` facade; this module introduces a small group type
//! `BackendTensorMetadata` for callers that want to pass or display
//! the triple as a unit.

use crate::model::{TensorDType, TensorShape};

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackendTensorMetadata {
    pub dtype: TensorDType,
    pub shape: TensorShape,
    pub device_label: String,
}

impl BackendTensorMetadata {
    pub fn new(dtype: TensorDType, shape: TensorShape, device_label: impl Into<String>) -> Self {
        Self {
            dtype,
            shape,
            device_label: device_label.into(),
        }
    }

    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }
}
