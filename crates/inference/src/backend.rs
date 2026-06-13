//! The backend-neutral inference execution trait.
//!
//! [`InferenceBackend`] is the central async trait that concrete
//! backends implement. An executor adapter calls
//! `backend.execute(request)` and the backend returns an
//! [`InferenceResponse`](crate::response::InferenceResponse) or
//! an [`InferenceError`](crate::InferenceError).

use crate::capability::InferenceBackendCapabilities;
use crate::error::InferenceError;
use crate::request::InferenceRequest;
use crate::response::InferenceResponse;

/// Backend-neutral inference execution trait.
///
/// V1 uses `async_trait` for a readable async trait-object surface,
/// matching the pattern already used by `runtime::NodeExecutor` and
/// `agent::AgentProvider`.
#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync + 'static {
    /// The stable kind identifier for this backend (e.g. `"candle"`,
    /// `"fake"`, `"remote"`).
    fn backend_kind(&self) -> &str;

    /// The capabilities this backend advertises.
    fn capabilities(&self) -> InferenceBackendCapabilities;

    /// Execute a single inference operation.
    async fn execute(&self, request: InferenceRequest)
    -> Result<InferenceResponse, InferenceError>;
}
