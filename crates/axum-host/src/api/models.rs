//! `POST /models/download` route for downloading HuggingFace models.
//!
//! Bridges to `WorkspaceHost::services().model_acquisition_service().acquire()`
//! and returns the acquisition report as `ModelDownloadOutput`.
//!
//! This is the Axum HTTP counterpart to MA-03's Tauri IPC command.

use axum::{Json, extract::State};
use reimagine_app_host::dto::ModelDownloadInput;
use reimagine_model_acquisition::{
    AcquireProvider, AllowPatterns, ModelAcquisitionRequest, OverwritePolicy, RepoId, Revision,
    TargetRelativeDir,
};

use crate::error::{AxumHostError, AxumHostResult};
use crate::state::AxumHostState;

/// `POST /models/download`.
///
/// Downloads a HuggingFace model into the workspace's `models` directory.
///
/// # Errors
///
/// - 400: invalid `repo_id` (must be `namespace/name`), invalid `target_relative_dir`, or malformed body
/// - 500: download failed (network, disk, or HuggingFace Hub error)
pub async fn download(
    State(state): State<AxumHostState>,
    Json(body): Json<ModelDownloadInput>,
) -> AxumHostResult<Json<reimagine_app_host::dto::ModelDownloadOutput>> {
    let ModelDownloadInput {
        repo_id,
        revision,
        allow_patterns,
        target_relative_dir,
        overwrite,
    } = body;

    // Parse repo_id
    let repo_id = RepoId::new(&repo_id).ok_or_else(|| AxumHostError::BadRequest {
        message: format!("invalid repo_id: expected `namespace/name` format, got `{repo_id}`"),
    })?;

    let revision = revision.map(Revision::new).unwrap_or_default();
    let allow_patterns = allow_patterns.map(AllowPatterns::new).unwrap_or_default();

    let target_relative_dir = TargetRelativeDir::new(target_relative_dir.into()).map_err(|e| {
        AxumHostError::BadRequest {
            message: format!("invalid target_relative_dir: {e}"),
        }
    })?;

    let overwrite_policy = match overwrite.as_deref() {
        Some("overwrite") => OverwritePolicy::Overwrite,
        Some("fail") => OverwritePolicy::Fail,
        _ => OverwritePolicy::Skip,
    };

    let request = ModelAcquisitionRequest {
        provider: AcquireProvider::HuggingFace,
        repo_id,
        revision,
        allow_patterns,
        target_relative_dir,
        overwrite_policy,
    };

    // Clone the service Arc and call acquire
    let acq = state
        .workspace()
        .services()
        .model_acquisition_service()
        .clone();

    // No progress sink for Axum V1 (synchronous request-response)
    let report = acq.acquire(request, None).await?;

    Ok(Json(reimagine_app_host::dto::ModelDownloadOutput::from(
        report,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
        routing::post,
    };
    use std::sync::Arc;
    use tower::ServiceExt;

    use reimagine_agent::WorkspaceScope;
    use reimagine_app_host::WorkspaceHost;
    use serde_json::json;

    use crate::recorder::RunEventRecorder;
    use crate::state::AxumHostState;

    fn test_state() -> AxumHostState {
        let base = unique_temp_dir("models");
        let host = WorkspaceHost::with_defaults(WorkspaceScope::new("ws-models"), &base);
        AxumHostState::new(Arc::new(host), Arc::new(RunEventRecorder::new()))
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-axum-host-{prefix}-{nonce}"))
    }

    #[tokio::test]
    async fn bad_repo_id_returns_400() {
        let app = Router::new()
            .route("/models/download", post(download))
            .with_state(test_state());

        let body = json!({
            "repoId": "invalid",
            "targetRelativeDir": "test/model",
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/models/download")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn bad_target_relative_dir_returns_400() {
        let app = Router::new()
            .route("/models/download", post(download))
            .with_state(test_state());

        let body = json!({
            "repoId": "test/model",
            "targetRelativeDir": "../escape",
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/models/download")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    #[ignore = "requires network access and HuggingFace token configuration"]
    async fn actual_download_integration() {
        let app = Router::new()
            .route("/models/download", post(download))
            .with_state(test_state());

        let body = json!({
            "repoId": "stabilityai/stable-diffusion-xl-base-1.0",
            "revision": "main",
            "allowPatterns": ["*.json"],
            "targetRelativeDir": "sdxl/base",
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/models/download")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
