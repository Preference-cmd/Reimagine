//! `GET /artifacts/:artifact_id` route.
//!
//! Serves PNG artifacts produced by workflow runs.

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use reimagine_app_host::ArtifactAccessError;
use reimagine_core::model::ArtifactId;
use tokio::fs::File;
use tokio_util::io::ReaderStream;

use crate::state::AxumHostState;

/// `GET /artifacts/:artifact_id` — serve a PNG artifact produced by a
/// workflow run.
///
/// Returns:
/// - 200 OK with `content-type: image/png` for valid PNG artifacts
/// - 404 Not Found for unknown artifact ids or unsafe references
/// - 410 Gone for artifact records whose file no longer exists
/// - 415 Unsupported Media Type for non-PNG artifacts
pub async fn get(State(state): State<AxumHostState>, Path(artifact_id): Path<String>) -> Response {
    let id = ArtifactId::new(artifact_id);

    let access = match state.workspace().resolve_artifact(&id) {
        Ok(access) => access,
        Err(ArtifactAccessError::UnknownArtifact) => {
            return (StatusCode::NOT_FOUND, "unknown artifact").into_response();
        }
        Err(ArtifactAccessError::UnsafeReference) => {
            return (StatusCode::NOT_FOUND, "unsafe artifact reference").into_response();
        }
        Err(ArtifactAccessError::FileGone) => {
            return (StatusCode::GONE, "artifact file gone").into_response();
        }
        Err(ArtifactAccessError::UnsupportedMedia) => {
            return (StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported media type").into_response();
        }
    };

    // Open the file and stream it
    let file = match File::open(&access.path).await {
        Ok(file) => file,
        Err(_) => {
            return (StatusCode::GONE, "artifact file gone").into_response();
        }
    };

    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    let content_type = HeaderValue::from_static("image/png");

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .body(body)
        .unwrap()
}
