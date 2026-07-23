//! Backend-affine handle types that ride on an [`ExecutionValue`].
//!
//! These handles are cheap, cloneable, and refer to backend-owned
//! payload stores. The handle carries the minimum metadata a caller
//! needs to route or display the value (backend kind, payload key,
//! shape/dtype/device, model id and role) without copying the heavy
//! payload.

use reimagine_core::model::{ModelId, ModelRole, TensorDType, TensorShape};

use crate::execution_value::backend::BackendPayloadKey;
use crate::execution_value::tensor::BackendTensorMetadata;
use crate::latent_content::LatentContent;
use crate::latent_space::LatentSpaceMetadata;
use crate::{Backend, BackendInstance};

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackendTensorHandle {
    backend: Backend,
    backend_instance: BackendInstance,
    payload_key: BackendPayloadKey,
    dtype: TensorDType,
    shape: TensorShape,
    device_label: String,
}

impl BackendTensorHandle {
    pub fn new(
        backend: Backend,
        payload_key: BackendPayloadKey,
        dtype: TensorDType,
        shape: TensorShape,
        device_label: impl Into<String>,
    ) -> Self {
        let backend_instance = BackendInstance::new(backend.as_str());
        Self::with_instance(
            backend,
            backend_instance,
            payload_key,
            dtype,
            shape,
            device_label,
        )
    }

    pub fn with_instance(
        backend: Backend,
        backend_instance: BackendInstance,
        payload_key: impl Into<BackendPayloadKey>,
        dtype: TensorDType,
        shape: TensorShape,
        device_label: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            backend_instance,
            payload_key: payload_key.into(),
            dtype,
            shape,
            device_label: device_label.into(),
        }
    }

    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    pub fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
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
    backend: Backend,
    backend_instance: BackendInstance,
    payload_key: BackendPayloadKey,
    device_label: Option<String>,
}

impl RuntimeModelHandle {
    pub fn new(
        model_id: ModelId,
        role: ModelRole,
        backend: Backend,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        let backend_instance = BackendInstance::new(backend.as_str());
        Self::with_instance(model_id, role, backend, backend_instance, payload_key)
    }

    pub fn with_instance(
        model_id: ModelId,
        role: ModelRole,
        backend: Backend,
        backend_instance: BackendInstance,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        Self {
            model_id,
            role,
            backend,
            backend_instance,
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

    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    pub fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
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
        backend: Backend,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        Self(RuntimeModelHandle::new(
            model_id,
            ModelRole::TextEncoder,
            backend,
            payload_key,
        ))
    }

    pub fn with_instance(
        model_id: ModelId,
        backend: Backend,
        backend_instance: BackendInstance,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        Self(RuntimeModelHandle::with_instance(
            model_id,
            ModelRole::TextEncoder,
            backend,
            backend_instance,
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

    pub fn backend(&self) -> &Backend {
        self.0.backend()
    }

    pub fn backend_instance(&self) -> &BackendInstance {
        self.0.backend_instance()
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
        backend: Backend,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        Self(RuntimeModelHandle::new(
            model_id,
            ModelRole::Vae,
            backend,
            payload_key,
        ))
    }

    pub fn with_instance(
        model_id: ModelId,
        backend: Backend,
        backend_instance: BackendInstance,
        payload_key: impl Into<BackendPayloadKey>,
    ) -> Self {
        Self(RuntimeModelHandle::with_instance(
            model_id,
            ModelRole::Vae,
            backend,
            backend_instance,
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

    pub fn backend(&self) -> &Backend {
        self.0.backend()
    }

    pub fn backend_instance(&self) -> &BackendInstance {
        self.0.backend_instance()
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
    latent_space: LatentSpaceMetadata,
    content: LatentContent,
}

impl RuntimeLatent {
    /// Build a [`RuntimeLatent`] handle with explicit latent-space
    /// metadata and content semantics.
    ///
    /// The metadata answers "which latent space is this compatible
    /// with"; [`LatentContent`] answers "is this payload a real
    /// latent, or just empty geometry?". Callers that want
    /// SDXL-base txt2img geometry should pass
    /// `LatentContent::EmptyGeometry`; callers that want a sampled
    /// or encoded latent should pass `Sampled` or `EncodedImage`.
    /// Backends producing real tensors downstream should never
    /// accidentally default to the wrong content class.
    pub fn new(
        payload: BackendTensorHandle,
        width: u32,
        height: u32,
        batch: u32,
        channels: u32,
        latent_space: LatentSpaceMetadata,
        content: LatentContent,
    ) -> Self {
        Self {
            payload,
            width,
            height,
            batch,
            channels,
            latent_space,
            content,
        }
    }

    /// Build a [`RuntimeLatent`] handle using the V1 SDXL base
    /// latent-space metadata and `EmptyGeometry` content.
    ///
    /// Prefer [`RuntimeLatent::new`] for new code; this helper
    /// exists so test fixtures and V1 hard-coded paths do not
    /// have to spell out the SDXL metadata record and content
    /// class.
    pub fn with_sdxl_base(
        payload: BackendTensorHandle,
        width: u32,
        height: u32,
        batch: u32,
        channels: u32,
    ) -> Self {
        Self::new(
            payload,
            width,
            height,
            batch,
            channels,
            LatentSpaceMetadata::sdxl_base(),
            LatentContent::EmptyGeometry,
        )
    }

    /// Replace the latent-space metadata. Used by the candle
    /// backend when materializing a sampled latent so the output
    /// handle carries the bundle's expected latent space even when
    /// the input latent used a different (but compatible) record.
    pub fn with_latent_space(mut self, latent_space: LatentSpaceMetadata) -> Self {
        self.latent_space = latent_space;
        self
    }

    /// Replace the runtime content classification. Backends use
    /// this when materializing a real latent payload (sampled by
    /// `diffusion.sample`, encoded by `latent.encode`, etc.) so
    /// downstream capabilities can reject empty geometry before
    /// decoding or partial-denoising.
    pub fn with_content(mut self, content: LatentContent) -> Self {
        self.content = content;
        self
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

    /// Latent-space metadata this handle carries. Backends compare
    /// this against their expected latent space before tensor
    /// operations.
    pub fn latent_space(&self) -> &LatentSpaceMetadata {
        &self.latent_space
    }

    /// Runtime content classification for this latent payload.
    ///
    /// See [`LatentContent`] for the full vocabulary. Empty
    /// geometry latents must not be decoded or partially denoised.
    pub fn content(&self) -> LatentContent {
        self.content
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
