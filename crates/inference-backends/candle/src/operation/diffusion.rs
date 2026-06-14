use reimagine_inference::request::InferenceRequest;
use reimagine_inference::response::InferenceResponse;

use crate::error::{BackendNotImplementedError, CandleBackendError};

pub fn execute_diffusion_sample(
    request: &InferenceRequest,
    backend_kind: &str,
) -> Result<InferenceResponse, CandleBackendError> {
    Err(CandleBackendError::BackendNotImplemented(
        BackendNotImplementedError::new(
            backend_kind,
            request.operation_id().clone(),
            "diffusion sample not implemented",
        ),
    ))
}
