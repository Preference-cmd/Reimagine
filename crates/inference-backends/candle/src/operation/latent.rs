use std::sync::Arc;

use candle_core::{DType, Tensor};
use reimagine_core::model::{ParamValue, TensorDType, TensorShape};
use reimagine_inference::InferenceBackend;
use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::{InferenceOutput, InferenceResponse};
use reimagine_runtime::{BackendKind, BackendPayloadKey, RuntimeLatent, RuntimeValue};

use crate::backend::CandleBackend;
use crate::error::{BackendNotImplementedError, CandleBackendError};

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
    backend_kind: &str,
) -> Result<InferenceResponse, CandleBackendError> {
    Err(CandleBackendError::BackendNotImplemented(
        BackendNotImplementedError::new(
            backend_kind,
            request.operation_id().clone(),
            "latent decode not implemented",
        ),
    ))
}
