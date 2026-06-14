use std::sync::Arc;

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

pub fn execute_latent_create_empty(
    backend: &CandleBackend,
    request: &InferenceRequest,
) -> Result<InferenceResponse, CandleBackendError> {
    let width = extract_i64(request.params(), "width")?;
    let height = extract_i64(request.params(), "height")?;
    let batch_size = extract_i64(request.params(), "batch_size")?;

    // SDXL VAE latent channels; V1 placeholder.
    let channels: u32 = 4;
    let latent_width = (width as u32) / 8;
    let latent_height = (height as u32) / 8;

    let latent = RuntimeValue::Latent(RuntimeLatent::new(
        reimagine_runtime::BackendTensorHandle::new(
            BackendKind::from(backend.backend_kind()),
            BackendPayloadKey::new(format!(
                "latent:{}:{}",
                request.run_id().as_str(),
                request.node_id().as_str()
            )),
            TensorDType::F32,
            TensorShape::new(vec![
                batch_size as usize,
                channels as usize,
                latent_height as usize,
                latent_width as usize,
            ]),
            backend.device().label(),
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
