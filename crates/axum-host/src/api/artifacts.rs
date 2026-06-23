//! `GET /artifacts/:artifact_id` route.
//!
//! Serves PNG artifacts produced by workflow runs.

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::HeaderValue;
use axum::response::IntoResponse;
use reimagine_core::model::ArtifactId;
use tokio::fs::File;
use tokio_util::io::ReaderStream;

use crate::error::{AxumHostError, AxumHostResult};
use crate::state::AxumHostState;

/// `GET /artifacts/:artifact_id` — serve a PNG artifact produced by a
/// workflow run.
///
/// Returns:
/// - 200 OK with `content-type: image/png` for valid PNG artifacts
/// - 404 Not Found for unknown artifact ids or unsafe references
/// - 410 Gone for artifact records whose file no longer exists
/// - 415 Unsupported Media Type for non-PNG artifacts
///
/// All error responses use the standard `{ "error": { "code", "message" } }`
/// envelope.
pub async fn get(
    State(state): State<AxumHostState>,
    Path(artifact_id): Path<String>,
) -> AxumHostResult<impl IntoResponse> {
    let id = ArtifactId::new(artifact_id);
    let access = state.workspace().resolve_artifact(&id)?;

    // Open the file and stream it
    let file = File::open(&access.path)
        .await
        .map_err(|_| AxumHostError::ArtifactFileGone)?;

    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    let content_type = HeaderValue::from_static("image/png");

    let response = axum::response::Response::builder()
        .status(axum::http::StatusCode::OK)
        .header("content-type", content_type)
        .body(body)
        .expect("build artifact response");

    Ok(response)
}
