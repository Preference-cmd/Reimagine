//! HTTP error envelope and `IntoResponse` mapping for the Axum host.
//!
//! Every error returned by a route handler funnels through this enum so
//! callers see a stable JSON shape (`{ "error": { "code": ..., "message": ... } }`)
//! and so app-host errors are translated to deterministic HTTP status
//! codes. Tauri / Axum-specific types must not leak through.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use reimagine_app_host::{AppHostError, ArtifactAccessError};
use reimagine_core::model::{RunId, WorkflowId};
use serde::Serialize;
use serde_json::json;

pub type AxumHostResult<T> = Result<T, AxumHostError>;

/// Stable, host-facing error variants for the Axum HTTP layer.
///
/// The set is deliberately narrower than [`AppHostError`]: the Axum
/// layer is only responsible for the cross-cutting HTTP concerns
/// (request parsing, workspace state lookup) and for translating
/// app-host errors into HTTP status codes.
#[derive(Debug)]
pub enum AxumHostError {
    /// The requested workflow id is not registered in the workspace.
    UnknownWorkflow { workflow_id: WorkflowId },
    /// The requested run id is not known to the runtime store.
    UnknownRun { run_id: RunId },
    /// The requested model id is not registered in the manifest.
    UnknownModel { model_id: String },
    /// The requested artifact id is not known to any run.
    UnknownArtifact,
    /// The artifact reference failed path safety validation.
    UnsafeArtifactReference,
    /// The artifact record exists but the backing file no longer exists.
    ArtifactFileGone,
    /// The artifact media type is not supported.
    UnsupportedArtifactMedia,
    /// Request body could not be deserialized.
    BadRequest { message: String },
    /// App-host returned an error that does not map to a more specific
    /// variant; the underlying message is preserved for the client.
    AppHost(AppHostError),
}

impl std::fmt::Display for AxumHostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownWorkflow { workflow_id } => {
                write!(f, "unknown workflow `{workflow_id}`")
            }
            Self::UnknownRun { run_id } => write!(f, "unknown run `{run_id}`"),
            Self::UnknownModel { model_id } => write!(f, "unknown model `{model_id}`"),
            Self::UnknownArtifact => write!(f, "unknown artifact"),
            Self::UnsafeArtifactReference => write!(f, "unsafe artifact reference"),
            Self::ArtifactFileGone => write!(f, "artifact file gone"),
            Self::UnsupportedArtifactMedia => write!(f, "unsupported media type"),
            Self::BadRequest { message } => write!(f, "bad request: {message}"),
            Self::AppHost(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for AxumHostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AppHost(error) => Some(error),
            _ => None,
        }
    }
}

impl From<AppHostError> for AxumHostError {
    fn from(value: AppHostError) -> Self {
        match value {
            AppHostError::UnknownWorkflow { workflow_id } => Self::UnknownWorkflow { workflow_id },
            AppHostError::UnknownRun { run_id } => Self::UnknownRun { run_id },
            other => Self::AppHost(other),
        }
    }
}

impl From<&str> for AxumHostError {
    fn from(model_id: &str) -> Self {
        Self::UnknownModel {
            model_id: model_id.to_owned(),
        }
    }
}

impl From<String> for AxumHostError {
    fn from(model_id: String) -> Self {
        Self::UnknownModel { model_id }
    }
}

impl From<ArtifactAccessError> for AxumHostError {
    fn from(value: ArtifactAccessError) -> Self {
        match value {
            ArtifactAccessError::UnknownArtifact => Self::UnknownArtifact,
            ArtifactAccessError::UnsafeReference => Self::UnsafeArtifactReference,
            ArtifactAccessError::FileGone => Self::ArtifactFileGone,
            ArtifactAccessError::UnsupportedMedia => Self::UnsupportedArtifactMedia,
        }
    }
}

/// Stable JSON error payload. The shape is part of the V1 HTTP
/// contract: clients should rely on `error.code` for branching and
/// `error.message` for human display.
#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    error: ErrorPayload<'a>,
}

#[derive(Debug, Serialize)]
struct ErrorPayload<'a> {
    code: &'a str,
    message: String,
}

impl IntoResponse for AxumHostError {
    fn into_response(self) -> Response {
        let (status, code, message) = match &self {
            Self::UnknownWorkflow { workflow_id } => (
                StatusCode::NOT_FOUND,
                "unknown_workflow",
                format!("unknown workflow `{workflow_id}`"),
            ),
            Self::UnknownRun { run_id } => (
                StatusCode::NOT_FOUND,
                "unknown_run",
                format!("unknown run `{run_id}`"),
            ),
            Self::UnknownModel { model_id } => (
                StatusCode::NOT_FOUND,
                "unknown_model",
                format!("unknown model `{model_id}`"),
            ),
            Self::UnknownArtifact => (
                StatusCode::NOT_FOUND,
                "unknown_artifact",
                "unknown artifact".to_string(),
            ),
            Self::UnsafeArtifactReference => (
                StatusCode::NOT_FOUND,
                "unsafe_artifact_reference",
                "unsafe artifact reference".to_string(),
            ),
            Self::ArtifactFileGone => (
                StatusCode::GONE,
                "artifact_file_gone",
                "artifact file gone".to_string(),
            ),
            Self::UnsupportedArtifactMedia => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media",
                "unsupported media type".to_string(),
            ),
            Self::BadRequest { message } => {
                (StatusCode::BAD_REQUEST, "bad_request", message.clone())
            }
            Self::AppHost(AppHostError::WorkflowIdPathUnsafe { workflow_id }) => (
                StatusCode::BAD_REQUEST,
                "unsafe_workflow_id",
                format!("workflow id `{workflow_id}` is not safe as a file name"),
            ),
            Self::AppHost(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                error.to_string(),
            ),
        };
        let body = Json(json!(ErrorBody {
            error: ErrorPayload { code, message },
        }));
        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn unknown_workflow_maps_to_404() {
        let err = AxumHostError::UnknownWorkflow {
            workflow_id: WorkflowId::new("missing"),
        };
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unknown_workflow");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("missing")
        );
    }

    #[tokio::test]
    async fn bad_request_maps_to_400() {
        let err = AxumHostError::BadRequest {
            message: "nope".to_string(),
        };
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn app_host_unknown_workflow_converts() {
        let err: AxumHostError = AppHostError::UnknownWorkflow {
            workflow_id: WorkflowId::new("x"),
        }
        .into();
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unsafe_workflow_id_maps_to_400() {
        let err: AxumHostError = AppHostError::WorkflowIdPathUnsafe {
            workflow_id: WorkflowId::new("../escape"),
        }
        .into();
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsafe_workflow_id");
    }
}
