use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::InferenceResponse;

use crate::error::{BackendNotImplementedError, CandleBackendError};

pub fn execute_image_save(
    request: &InferenceRequest,
    backend_kind: &str,
) -> Result<InferenceResponse, CandleBackendError> {
    Err(CandleBackendError::BackendNotImplemented(
        BackendNotImplementedError::new(
            backend_kind,
            request.operation_id().clone(),
            "image save not implemented",
        ),
    ))
}

pub fn execute_image_preview(
    request: &InferenceRequest,
    backend_kind: &str,
) -> Result<InferenceResponse, CandleBackendError> {
    Err(CandleBackendError::BackendNotImplemented(
        BackendNotImplementedError::new(
            backend_kind,
            request.operation_id().clone(),
            "image preview not implemented",
        ),
    ))
}
