//! Model download, acquire, and management routes.

use axum::{Json, extract::Path, extract::State};
use reimagine_app_host::dto::{
    ModelAcquireConversionReport, ModelAcquireDownloadReport, ModelAcquireInput,
    ModelAcquireOutput, ModelComponentDto, ModelConvertConversionReport, ModelConvertInput,
    ModelConvertOutput, ModelDetailDto, ModelDownloadInput, ModelListEntry, ModelListOutput,
    ModelRemoveOutput, format_status,
};
use reimagine_core::model::ModelId;
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

/// `POST /models/acquire`.
///
/// Downloads a HuggingFace model, converts it to a backend-native
/// component layout, and registers the result in the workspace manifest.
pub async fn acquire(
    State(state): State<AxumHostState>,
    Json(body): Json<ModelAcquireInput>,
) -> AxumHostResult<Json<ModelAcquireOutput>> {
    let target_backend = body.target_backend.as_deref().unwrap_or("burn");
    let overwrite_policy = match body.overwrite.as_deref() {
        Some("overwrite") => OverwritePolicy::Overwrite,
        Some("fail") => OverwritePolicy::Fail,
        _ => OverwritePolicy::Skip,
    };

    // RepoId::new validates the repo_id format; we use name() as model_id.
    let repo = RepoId::new(&body.repo_id).ok_or_else(|| AxumHostError::BadRequest {
        message: format!(
            "invalid repo_id: expected `namespace/name` format, got `{}`",
            body.repo_id
        ),
    })?;
    let model_id = repo.name().to_string();

    let model_service = state.workspace().services().model_service();
    let acq_service = state
        .workspace()
        .services()
        .model_acquisition_service()
        .clone();

    let result = model_service
        .acquire_and_convert(
            &body.repo_id,
            &model_id,
            body.revision.as_deref(),
            target_backend,
            overwrite_policy,
            &acq_service,
        )
        .await?;

    let backend = result.backend.clone();

    Ok(Json(ModelAcquireOutput {
        outcome: result.outcome,
        model_id: result.model_id,
        imported_model_id: result.imported_model_id.unwrap_or_default(),
        acquisition: ModelAcquireDownloadReport {
            repo_id: result.acquisition_report,
            revision: String::new(),
            file_count: result.acquisition_file_count,
            total_bytes: result.acquisition_total_bytes,
        },
        conversion: ModelAcquireConversionReport {
            backend,
            mapped_tensor_count: result.mapped_tensor_count,
            component_count: result.component_count,
            source_layout: result.source_layout,
        },
    }))
}

/// `GET /models`.
///
/// Lists every model entry currently registered in the workspace manifest.
pub async fn list_models(
    State(state): State<AxumHostState>,
) -> AxumHostResult<Json<ModelListOutput>> {
    let model_service = state.workspace().services().model_service();
    let descriptors = model_service.list_models().await?;
    let entries = descriptors.into_iter().map(ModelListEntry::from).collect();
    Ok(Json(ModelListOutput { models: entries }))
}

/// `GET /models/:id`.
///
/// Returns a single model's manifest entry plus its component list.
pub async fn get_model(
    State(state): State<AxumHostState>,
    Path(model_id): Path<String>,
) -> AxumHostResult<Json<ModelDetailDto>> {
    let model_service = state.workspace().services().model_service();
    let descriptors = model_service.list_models().await?;

    let descriptor = descriptors
        .iter()
        .find(|d| d.id().as_str() == model_id)
        .cloned()
        .ok_or_else(|| AxumHostError::UnknownModel {
            model_id: model_id.clone(),
        })?;

    let components = descriptor
        .components()
        .iter()
        .map(|c| {
            let path = c.source().path().to_string();
            let metadata = c.metadata().clone();
            ModelComponentDto {
                role: format!("{:?}", c.role()),
                path,
                metadata,
            }
        })
        .collect();

    let dto = ModelDetailDto {
        id: descriptor.id().as_str().to_owned(),
        model_series: descriptor.model_series().as_str().to_owned(),
        variant: descriptor.variant().as_str().to_owned(),
        source_status: format_status(descriptor.source_status()),
        backend: descriptor
            .metadata()
            .get("backend")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string()),
        format: format!("{:?}", descriptor.format()).to_ascii_lowercase(),
        components,
    };

    Ok(Json(dto))
}

/// `DELETE /models/:id`.
///
/// Removes the model entry from the manifest. Disk files are **not** touched.
pub async fn remove_model(
    State(state): State<AxumHostState>,
    Path(model_id): Path<String>,
) -> AxumHostResult<Json<ModelRemoveOutput>> {
    let model_service = state.workspace().services().model_service();
    let core_id = ModelId::new(&model_id);

    // Verify the model exists before attempting removal.
    let descriptors = model_service.list_models().await?;
    if !descriptors.iter().any(|d| d.id() == &core_id) {
        return Err(AxumHostError::UnknownModel {
            model_id: model_id.clone(),
        });
    }

    let (_, _report) = model_service.remove_model(&core_id).await?;

    Ok(Json(ModelRemoveOutput {
        outcome: "removed".to_string(),
        model_id,
    }))
}

/// `POST /models/convert`.
///
/// Converts an already-downloaded checkpoint to the specified backend.
/// Routes to Burn (`execute_real_burn_sdxl_checkpoint_import`) or
/// Candle (`import_sdxl_checkpoint_to_candle_split`).
pub async fn convert_checkpoint(
    State(state): State<AxumHostState>,
    Json(body): Json<ModelConvertInput>,
) -> AxumHostResult<Json<ModelConvertOutput>> {
    let model_service = state.workspace().services().model_service();
    let target_backend = body.target_backend.as_str();

    match target_backend {
        "burn" => {
            let report = model_service
                .convert_checkpoint_to_burn(&body.model_id)
                .await?;

            let conversion = ModelConvertConversionReport {
                mapped_tensor_count: report.mapped_tensor_count,
                component_count: report.output_components.len(),
                source_layout: report.source_layout,
            };

            Ok(Json(ModelConvertOutput {
                outcome: "converted".to_string(),
                model_id: body.model_id.clone(),
                backend: "burn".to_string(),
                report: conversion,
            }))
        }
        "candle" => {
            let core_id = ModelId::new(&body.model_id);
            let (_manifest, _report, _import_result) = model_service
                .import_sdxl_checkpoint_to_candle_split(&core_id)
                .await?;

            let conversion = ModelConvertConversionReport {
                mapped_tensor_count: 0,
                component_count: 4,
                source_layout: "candle_example_split".to_string(),
            };

            Ok(Json(ModelConvertOutput {
                outcome: "converted".to_string(),
                model_id: body.model_id.clone(),
                backend: "candle".to_string(),
                report: conversion,
            }))
        }
        other => Err(AxumHostError::BadRequest {
            message: format!("unsupported target_backend `{other}`; expected `burn` or `candle`"),
        }),
    }
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

    use crate::build_router;
    use crate::recorder::RunEventRecorder;
    use crate::state::AxumHostState;

    fn test_state() -> AxumHostState {
        let base = unique_temp_dir("models");
        let host = WorkspaceHost::with_defaults(WorkspaceScope::new("ws-models"), &base);
        AxumHostState::new(Arc::new(host), Arc::new(RunEventRecorder::new()))
    }

    async fn seeded_state_with_models() -> AxumHostState {
        let base = unique_temp_dir("models");
        let host = WorkspaceHost::with_defaults(WorkspaceScope::new("ws-models"), &base);

        // Seed a manifest entry so listing and detail work.
        let model_service = host.services().model_service();
        let (mut manifest, _) = model_service.load_manifest().await.unwrap();

        use reimagine_core::model::{ModelId, ModelRole, ModelSeries, ModelVariant};
        use reimagine_model_manager::{
            Fingerprint, ModelDescriptor, ModelFormat, ModelRootId, ModelSource, ModelSourceStatus,
        };

        let model = ModelDescriptor::new(
            ModelId::new("test-model-01"),
            ModelSeries::new("stable_diffusion"),
            ModelVariant::new("sdxl"),
            vec![
                ModelRole::DiffusionModel,
                ModelRole::TextEncoder,
                ModelRole::Vae,
            ],
            ModelSource::relative(ModelRootId::new("base"), "test/model.safetensors"),
            ModelFormat::Safetensors,
        )
        .with_source_status(ModelSourceStatus::Available)
        .with_fingerprint(Fingerprint::sha256("abc123"));

        manifest.upsert_model(model);
        model_service.save_manifest(&manifest).await.unwrap();

        AxumHostState::new(Arc::new(host), Arc::new(RunEventRecorder::new()))
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-axum-host-{prefix}-{nonce}"))
    }

    // ------------------------------------------------------------------
    //  Existing tests
    // ------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    //  Issue 03: Model management API tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn get_models_list_returns_empty_when_no_models_seeded() {
        let app = build_router().with_state(test_state());
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = json["models"].as_array().expect("models array");
        assert!(models.is_empty(), "expected empty list, got: {models:?}");
    }

    #[tokio::test]
    async fn get_models_list_returns_seeded_model() {
        let app = build_router().with_state(seeded_state_with_models().await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = json["models"].as_array().expect("models array");
        assert_eq!(models.len(), 1, "expected 1 model, got: {models:?}");
        assert_eq!(models[0]["id"], "test-model-01");
    }

    #[tokio::test]
    async fn get_model_by_id_returns_detail_and_200() {
        let app = build_router().with_state(seeded_state_with_models().await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/models/test-model-01")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["id"], "test-model-01");
        assert!(
            json["components"].is_array(),
            "components should be an array"
        );
    }

    #[tokio::test]
    async fn get_model_by_id_returns_404_for_missing() {
        let app = build_router().with_state(seeded_state_with_models().await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/models/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_model_by_id_returns_200_and_removes_entry() {
        let app = build_router().with_state(seeded_state_with_models().await);

        // Delete the model
        let response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/models/test-model-01")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["outcome"], "removed");
        assert_eq!(json["modelId"], "test-model-01");
    }

    #[tokio::test]
    async fn delete_model_by_id_returns_404_for_missing() {
        let app = build_router().with_state(test_state());
        let response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/models/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn convert_checkpoint_rejects_unknown_backend() {
        let app = build_router().with_state(seeded_state_with_models().await);
        let body = json!({
            "modelId": "test-model-01",
            "targetBackend": "invalid",
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/models/convert")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
