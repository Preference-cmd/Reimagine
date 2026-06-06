//! Runtime values passed between node executors during a single run.

use reimagine_core::model::{
    ArtifactRef, ModelId, ModelRole, ParamValue, TensorDType, TensorShape,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackendKind(String);

impl BackendKind {
    pub fn new(kind: impl Into<String>) -> Self {
        Self(kind.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for BackendKind {
    fn from(kind: String) -> Self {
        Self(kind)
    }
}

impl From<&str> for BackendKind {
    fn from(kind: &str) -> Self {
        Self(kind.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackendPayloadKey(String);

impl BackendPayloadKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BackendPayloadKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for BackendPayloadKey {
    fn from(key: String) -> Self {
        Self(key)
    }
}

impl From<&str> for BackendPayloadKey {
    fn from(key: &str) -> Self {
        Self(key.to_owned())
    }
}

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

    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
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
pub struct ConditioningMetadata {
    width: u32,
    height: u32,
    crop_x: u32,
    crop_y: u32,
    target_width: u32,
    target_height: u32,
}

impl ConditioningMetadata {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            crop_x: 0,
            crop_y: 0,
            target_width: width,
            target_height: height,
        }
    }

    pub fn with_crop(mut self, crop_x: u32, crop_y: u32) -> Self {
        self.crop_x = crop_x;
        self.crop_y = crop_y;
        self
    }

    pub fn with_target_size(mut self, target_width: u32, target_height: u32) -> Self {
        self.target_width = target_width;
        self.target_height = target_height;
        self
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn crop_x(&self) -> u32 {
        self.crop_x
    }

    pub fn crop_y(&self) -> u32 {
        self.crop_y
    }

    pub fn target_width(&self) -> u32 {
        self.target_width
    }

    pub fn target_height(&self) -> u32 {
        self.target_height
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RuntimeConditioning {
    text_embedding: BackendTensorHandle,
    pooled_embedding: Option<BackendTensorHandle>,
    metadata: ConditioningMetadata,
}

impl RuntimeConditioning {
    pub fn new(text_embedding: BackendTensorHandle, metadata: ConditioningMetadata) -> Self {
        Self {
            text_embedding,
            pooled_embedding: None,
            metadata,
        }
    }

    pub fn with_pooled_embedding(mut self, pooled_embedding: BackendTensorHandle) -> Self {
        self.pooled_embedding = Some(pooled_embedding);
        self
    }

    pub fn text_embedding(&self) -> &BackendTensorHandle {
        &self.text_embedding
    }

    pub fn pooled_embedding(&self) -> Option<&BackendTensorHandle> {
        self.pooled_embedding.as_ref()
    }

    pub fn metadata(&self) -> &ConditioningMetadata {
        &self.metadata
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

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RuntimeValue {
    Param(ParamValue),
    Model(RuntimeModelHandle),
    Clip(RuntimeClipHandle),
    Vae(RuntimeVaeHandle),
    Latent(RuntimeLatent),
    Conditioning(RuntimeConditioning),
    Image(RuntimeImage),
    Artifact(ArtifactRef),
    Null,
}

impl RuntimeValue {
    pub fn as_param(&self) -> Option<&ParamValue> {
        match self {
            Self::Param(param) => Some(param),
            _ => None,
        }
    }
}
