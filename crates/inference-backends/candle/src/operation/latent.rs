//! `latent.create_empty` and `latent.decode` operations.
//!
//! ## `latent.create_empty`
//!
//! Allocates a zero-initialized SDXL latent tensor in the backend store
//! and returns a lightweight `RuntimeValue::Latent` handle.
//!
//! ## `latent.decode`
//!
//! Consumes a VAE handle and a sampled latent, dispatches to the SDXL VAE
//! placeholder decoder ([`SdxlVaeDecoder`]), stores the resulting image
//! payload in the backend store, and returns a lightweight
//! `RuntimeValue::Image` handle.
//!
//! The operation is model-family-neutral at the protocol level.
//! SDXL-specific VAE decoding lives in
//! `models/stable_diffusion/sdxl/vae.rs`.

use std::sync::Arc;

use candle_core::{DType, Tensor};
use reimagine_core::model::{ParamValue, SlotId, TensorDType, TensorShape};
use reimagine_inference::InferenceBackend;
use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::{InferenceOutput, InferenceResponse};
use reimagine_runtime::{
    BackendKind, BackendPayloadKey, RuntimeImage, RuntimeLatent, RuntimeVaeHandle, RuntimeValue,
};

use crate::backend::CandleBackend;
use crate::error::CandleBackendError;
use crate::models::stable_diffusion::sdxl::vae::SdxlVaeDecoder;
use crate::models::{LoadedModelBundle, LoadedSdxlBundle};

fn extract_i64(
    params: &std::collections::HashMap<reimagine_core::model::SlotId, ParamValue>,
    slot: &str,
) -> Result<i64, CandleBackendError> {
    match params.get(&reimagine_core::model::SlotId::new(slot)) {
        Some(ParamValue::Integer(v)) => Ok(*v),
        Some(_) => Err(CandleBackendError::InvalidRequest(format!(
            "param `{slot}` must be an integer"
        ))),
        None => Err(CandleBackendError::InvalidRequest(format!(
            "missing param `{slot}`"
        ))),
    }
}

fn validate_latent_dimensions(
    width: i64,
    height: i64,
    batch_size: i64,
) -> Result<(), CandleBackendError> {
    if width <= 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent width must be positive (got {width})"
        )));
    }
    if height <= 0 {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent height must be positive (got {height})"
        )));
    }
    if batch_size <= 0 {
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
    // SDXL latent dimensions are cast to `u32` below; reject values
    // that would overflow the cast before we attempt it.
    let max_dim = u32::MAX as i64;
    if width > max_dim || height > max_dim || batch_size > max_dim {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent dimensions must fit in u32 (width={width}, height={height}, batch_size={batch_size}, max={max_dim})"
        )));
    }
    Ok(())
}

pub fn execute_latent_create_empty(
    backend: &CandleBackend,
    request: &InferenceRequest,
) -> Result<InferenceResponse, CandleBackendError> {
    let width = extract_i64(request.params(), "width")?;
    let height = extract_i64(request.params(), "height")?;
    let batch_size = extract_i64(request.params(), "batch_size")?;

    validate_latent_dimensions(width, height, batch_size)?;

    let channels: u32 = 4;
    let latent_width = (width as u32) / 8;
    let latent_height = (height as u32) / 8;
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

    let latent = RuntimeValue::Latent(RuntimeLatent::new(
        reimagine_runtime::BackendTensorHandle::new(
            BackendKind::from(backend.backend_kind()),
            payload_key,
            TensorDType::F32,
            TensorShape::new(shape),
            backend.device_label(),
        ),
        width as u32,
        height as u32,
        batch_size as u32,
        channels,
    ));
    Ok(InferenceResponse::new(vec![InferenceOutput::new(
        "latent",
        Arc::new(latent),
    )]))
}

pub fn execute_latent_decode(
    request: &InferenceRequest,
    backend: &CandleBackend,
) -> Result<InferenceResponse, CandleBackendError> {
    let vae_value = require_input(request, "vae")?;
    let latent_value = require_input(request, "latent")?;
    let vae_handle = require_vae_handle(vae_value, backend)?;
    let latent_handle = require_latent_handle(latent_value, backend)?;

    let sdxl_bundle = require_sdxl_bundle_for_vae(&vae_handle, backend)?;

    if vae_handle.payload_key() != &sdxl_bundle.vae_payload_key {
        return Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode `vae` payload `{}` does not match loaded SDXL VAE payload `{}` for model `{}`",
            vae_handle.payload_key().as_str(),
            sdxl_bundle.vae_payload_key.as_str(),
            sdxl_bundle.model_id.as_str()
        )));
    }

    let input_latent = backend
        .store()
        .get_latent(latent_handle.payload().payload_key())?;

    let decoder = SdxlVaeDecoder::new();
    let image = decoder.decode(&input_latent, backend.device().as_ref())?;

    let payload_key = BackendPayloadKey::new(format!(
        "image:{}:{}",
        request.run_id().as_str(),
        request.node_id().as_str()
    ));

    backend
        .store()
        .insert_image(request.run_id().clone(), payload_key.clone(), image.clone());

    let image_value = RuntimeValue::Image(RuntimeImage::new(
        reimagine_runtime::BackendTensorHandle::new(
            BackendKind::from(backend.backend_kind()),
            payload_key,
            TensorDType::F32,
            TensorShape::new(image.dims()),
            backend.device_label(),
        ),
        image.width(),
        image.height(),
        image.batch(),
        image.color_space().to_string(),
    ));

    Ok(InferenceResponse::new(vec![InferenceOutput::new(
        "image",
        Arc::new(image_value),
    )]))
}

fn require_input<'a>(
    request: &'a InferenceRequest,
    slot: &str,
) -> Result<&'a Arc<RuntimeValue>, CandleBackendError> {
    request.inputs().get(&SlotId::new(slot)).ok_or_else(|| {
        CandleBackendError::InvalidRequest(format!("latent.decode requires a `{slot}` input"))
    })
}

fn require_vae_handle(
    value: &RuntimeValue,
    backend: &CandleBackend,
) -> Result<RuntimeVaeHandle, CandleBackendError> {
    match value {
        RuntimeValue::Vae(handle) => {
            if handle.backend().as_str() != backend.backend_kind() {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "latent.decode `vae` handle belongs to backend `{}`, expected `{}`",
                    handle.backend().as_str(),
                    backend.backend_kind()
                )));
            }
            Ok(handle.clone())
        }
        _ => Err(CandleBackendError::InvalidRequest(
            "latent.decode `vae` input must be a Vae handle".to_string(),
        )),
    }
}

fn require_latent_handle(
    value: &RuntimeValue,
    backend: &CandleBackend,
) -> Result<reimagine_runtime::RuntimeLatent, CandleBackendError> {
    match value {
        RuntimeValue::Latent(handle) => {
            if handle.payload().backend().as_str() != backend.backend_kind() {
                return Err(CandleBackendError::InvalidRequest(format!(
                    "latent.decode `latent` handle belongs to backend `{}`, expected `{}`",
                    handle.payload().backend().as_str(),
                    backend.backend_kind()
                )));
            }
            Ok(handle.clone())
        }
        _ => Err(CandleBackendError::InvalidRequest(
            "latent.decode `latent` input must be a Latent handle".to_string(),
        )),
    }
}

fn require_sdxl_bundle_for_vae(
    vae: &reimagine_runtime::RuntimeVaeHandle,
    backend: &CandleBackend,
) -> Result<Arc<LoadedSdxlBundle>, CandleBackendError> {
    let bundle = backend
        .model_cache()
        .get_bundle(vae.model_id())
        .ok_or_else(|| {
            CandleBackendError::InvalidRequest(format!(
                "no loaded model bundle found for model `{}`; load the model first via `model.load_bundle`",
                vae.model_id().as_str()
            ))
        })?;

    extract_sdxl_bundle(&bundle, vae.model_id().as_str())
}

fn extract_sdxl_bundle(
    bundle: &Arc<LoadedModelBundle>,
    model_id: &str,
) -> Result<Arc<LoadedSdxlBundle>, CandleBackendError> {
    match bundle.as_ref() {
        LoadedModelBundle::StableDiffusionSdxl(sdxl) => Ok(sdxl.clone()),
        #[allow(unreachable_patterns)]
        _ => Err(CandleBackendError::InvalidRequest(format!(
            "latent.decode only supports stable_diffusion/sdxl bundles; model `{model_id}` is not SDXL"
        ))),
    }
}
