//! Backend-affine handle types that ride on an [`ExecutionValue`].
//!
//! These handles are cheap, cloneable, and refer to backend-owned
//! payload stores. The handle carries the minimum metadata a caller
//! needs to route or display the value (backend kind, payload key,
//! shape/dtype/device, model id and role) without copying the heavy
//! payload.

use crate::model::{ModelId, ModelRole, TensorDType, TensorShape};

use super::backend::{BackendKind, BackendPayloadKey};
use super::tensor::BackendTensorMetadata;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackendTensorHandle {
    backend: BackendKind,
    payload_key: BackendPayloadKey,
    dtype: TensorDType,
    shape: TensorShape,
    device_label: String,
}

impl BackendTensorHandle {
    pub fn new(
        backend: BackendKind,
        payload_key: BackendPayloadKey,
        dtype: TensorDType,
        shape: TensorShape,
        device_label: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            payload_key,
            dtype,
            shape,
            device_label: device_label.into(),
        }
    }

    pub fn backend(&self) -> &BackendKind {
        &self.backend
    }

    pub fn payload_key(&self) -> &BackendPayloadKey {
        &self.payload_key
    }

    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn shape(&self) -> TensorShape {
        self.shape.clone()
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    /// View the handle's tensor metadata as a single [`BackendTensorMetadata`].
    pub fn metadata(&self) -> BackendTensorMetadata {
        BackendTensorMetadata::new(self.dtype, self.shape.clone(), self.device_label.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RuntimeModelHandle {
    model_id: ModelId,
    role: ModelRole,
    backend: BackendKind,
    payload_key: BackendPayloadKey,
    device_label: Option<String>,
}

impl RuntimeModelHandle {
    pub fn new(
        model_id: ModelId,
        role: ModelRole,
        backend: BackendKind,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        Self {
            model_id,
            role,
            backend,
            payload_key: payload_key.into(),
            device_label: None,
        }
    }

    pub fn with_device(mut self, device_label: impl Into<String>) -> Self {
        self.device_label = Some(device_label.into());
        self
    }

    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    pub fn role(&self) -> ModelRole {
        self.role
    }

    pub fn backend(&self) -> &BackendKind {
        &self.backend
    }

    pub fn payload_key(&self) -> &BackendPayloadKey {
        &self.payload_key
    }

    pub fn device_label(&self) -> Option<&str> {
        self.device_label.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RuntimeClipHandle(RuntimeModelHandle);

impl RuntimeClipHandle {
    pub fn new(
        model_id: ModelId,
        backend: BackendKind,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        Self(RuntimeModelHandle::new(
            model_id,
            ModelRole::TextEncoder,
            backend,
            payload_key,
        ))
    }

    pub fn with_device(mut self, device_label: impl Into<String>) -> Self {
        self.0 = self.0.with_device(device_label);
        self
    }

    pub fn model_id(&self) -> &ModelId {
        self.0.model_id()
    }

    pub fn backend(&self) -> &BackendKind {
        self.0.backend()
    }

    pub fn payload_key(&self) -> &BackendPayloadKey {
        self.0.payload_key()
    }

    pub fn device_label(&self) -> Option<&str> {
        self.0.device_label()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RuntimeVaeHandle(RuntimeModelHandle);

impl RuntimeVaeHandle {
    pub fn new(
        model_id: ModelId,
        backend: BackendKind,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        Self(RuntimeModelHandle::new(
            model_id,
            ModelRole::Vae,
            backend,
            payload_key,
        ))
    }

    pub fn with_device(mut self, device_label: impl Into<String>) -> Self {
        self.0 = self.0.with_device(device_label);
        self
    }

    pub fn model_id(&self) -> &ModelId {
        self.0.model_id()
    }

    pub fn backend(&self) -> &BackendKind {
        self.0.backend()
    }

    pub fn payload_key(&self) -> &BackendPayloadKey {
        self.0.payload_key()
    }

    pub fn device_label(&self) -> Option<&str> {
        self.0.device_label()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RuntimeLatent {
    payload: BackendTensorHandle,
    width: u32,
    height: u32,
    batch: u32,
    channels: u32,
}

impl RuntimeLatent {
    pub fn new(
        payload: BackendTensorHandle,
        width: u32,
        height: u32,
        batch: u32,
        channels: u32,
    ) -> Self {
        Self {
            payload,
            width,
            height,
            batch,
            channels,
        }
    }

    pub fn payload(&self) -> &BackendTensorHandle {
        &self.payload
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RuntimeImage {
    payload: BackendTensorHandle,
    width: u32,
    height: u32,
    batch: u32,
    color_space: String,
}

impl RuntimeImage {
    pub fn new(
        payload: BackendTensorHandle,
        width: u32,
        height: u32,
        batch: u32,
        color_space: impl Into<String>,
    ) -> Self {
        Self {
            payload,
            width,
            height,
            batch,
            color_space: color_space.into(),
        }
    }

    pub fn payload(&self) -> &BackendTensorHandle {
        &self.payload
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn color_space(&self) -> &str {
        &self.color_space
    }
}
