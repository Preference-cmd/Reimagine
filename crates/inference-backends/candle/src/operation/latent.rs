//! `latent.create_empty` and `latent.decode` operations.
//!
//! ## `latent.create_empty`
//!
//! Allocates a zero-initialized SDXL latent tensor in the backend store
//! and returns a [`CreateEmptyLatentResponse`] carrying a lightweight
//! `RuntimeLatent` handle.
//!
//! ## `latent.decode`
//!
//! Consumes a VAE handle and a sampled latent, dispatches to the loaded
//! model family's decoder through the graph facade, stores the resulting
//! image payload in the backend store, and returns a
//! [`LatentDecodeResponse`] carrying a lightweight `RuntimeImage`
//! handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! SDXL-specific VAE decoding lives behind the facade in
//! `models/stable_diffusion/sdxl/vae.rs`.

use candle_core::{DType, Tensor};
use reimagine_core::model::{TensorDType, TensorShape};
use reimagine_inference::{
    BackendPayloadKey, BackendTensorHandle, CreateEmptyLatentRequest, CreateEmptyLatentResponse,
    InferenceBackend, LatentDecodeRequest, LatentDecodeResponse, RuntimeImage, RuntimeLatent,
    RuntimeVaeHandle,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::graph::{LatentDecodeInput, LatentDecodeResult};

fn validate_latent_dimensions(
    width: u32,
    height: u32,
    batch_size: u32,
) -> Result<(), CandleBackendError> {
    if width == 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent width must be positive (got {width})"
        )));
    }
    if height == 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent height must be positive (got {height})"
        )));
    }
    if batch_size == 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent batch_size must be positive (got {batch_size})"
        )));
    }
    if width % 8 != 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent width must be divisible by 8 (got {width})"
        )));
    }
    if height % 8 != 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent height must be divisible by 8 (got {height})"
        )));
    }
    Ok(())
}

pub fn execute_latent_create_empty(
    backend: &CandleBackend,
    request: CreateEmptyLatentRequest,
) -> Result<CreateEmptyLatentResponse, CandleBackendError> {
    let width = request.width();
    let height = request.height();
    let batch_size = request.batch_size();

    validate_latent_dimensions(width, height, batch_size)?;

    let channels: u32 = 4;
    let latent_width = width / 8;
    let latent_height = height / 8;
    let shape: Vec<usize> = vec![
        batch_size as usize,
        channels as usize,
        latent_height as usize,
        latent_width as usize,
    ];

    let tensor =
        Tensor::zeros(shape.clone(), DType::F32, backend.device().as_ref()).map_err(|err| {
            CandleBackendError::InvalidRequest(format!(
                "failed to allocate latent tensor for shape {shape:?}: {err}"
            ))
        })?;

    let payload_key = BackendPayloadKey::new(format!(
        "latent:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    backend
        .store()
        .insert_latent(request.run_id().clone(), payload_key.clone(), tensor);

    let latent = RuntimeLatent::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            payload_key,
            TensorDType::F32,
            TensorShape::new(shape),
            backend.device_label(),
        ),
        width,
        height,
        batch_size,
        channels,
    );
    Ok(CreateEmptyLatentResponse::new(latent))
}

pub fn execute_latent_decode(
    request: LatentDecodeRequest,
    backend: &CandleBackend,
) -> Result<LatentDecodeResponse, CandleBackendError> {
    let vae_handle = request.vae().clone();
    let latent_handle = request.latent().clone();

    if vae_handle.backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode `vae` handle belongs to backend `{}`, expected `{}`",
            vae_handle.backend().as_str(),
            backend.backend_kind()
        )));
    }
    if latent_handle.payload().backend() != backend.backend_kind() {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode `latent` handle belongs to backend `{}`, expected `{}`",
            latent_handle.payload().backend().as_str(),
            backend.backend_kind()
        )));
    }

    let bundle = require_bundle_for_vae(&vae_handle, backend)?;
    bundle.validate_vae_handle(&vae_handle)?;

    let input_latent = backend
        .store()
        .get_latent(latent_handle.payload().payload_key())?;

    let LatentDecodeResult { image } = bundle.decode_latent(
        LatentDecodeInput {
            latent: input_latent,
        },
        backend.device().as_ref(),
    )?;

    let payload_key = BackendPayloadKey::new(format!(
        "image:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    backend
        .store()
        .insert_image(request.run_id().clone(), payload_key.clone(), image.clone());

    let image_value = RuntimeImage::new(
        BackendTensorHandle::with_instance(
            backend.backend_kind().clone(),
            backend.backend_instance(),
            payload_key,
            TensorDType::F32,
            TensorShape::new(image.dims()),
            backend.device_label(),
        ),
        image.width(),
        image.height(),
        image.batch(),
        image.color_space().to_string(),
    );

    Ok(LatentDecodeResponse::new(image_value))
}

fn require_bundle_for_vae(
    vae: &RuntimeVaeHandle,
    backend: &CandleBackend,
) -> Result<std::sync::Arc<crate::models::LoadedModelBundle>, CandleBackendError> {
    backend.model_cache().get_bundle(vae.model_id()).ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!(
            "no loaded model bundle found for model `{}`; load the model first via `model.load_bundle`",
            vae.model_id().as_str()
        ))
    })
}
