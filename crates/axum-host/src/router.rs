//! Axum router construction for the V1 HTTP surface.
//!
//! Mounts endpoints for health, compute profile, node catalog, workflow
//! execution, run tracking, artifact inspection, and model management.
//! The router is the single source of truth for which paths are mounted;
//! tests and `server.rs` both go through [`build_router`].

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::api::{artifacts, health, nodes, runs, workflows};
#[cfg(feature = "aigc")]
use crate::api::{compute_profile, models};
use crate::state::AxumHostState;

/// Build the V1 HTTP router. The router is stateless — every
/// handler takes the [`AxumHostState`] via the standard `State`
/// extractor.
pub fn build_router() -> Router<AxumHostState> {
    let router = Router::new()
        .route("/health", get(health::get))
        .route("/nodes", get(nodes::list))
        .route("/workflows/open", post(workflows::open))
        .route("/workflows/{id}/run", post(workflows::run))
        .route("/runs/{id}", get(runs::get))
        .route("/runs/{id}/events", get(runs::events))
        .route("/artifacts/{artifact_id}", get(artifacts::get));

    #[cfg(feature = "aigc")]
    let router = router
        .route("/compute-profile", get(compute_profile::get))
        .route("/models/download", post(models::download))
        .route("/models/acquire", post(models::acquire))
        .route("/models", get(models::list_models))
        .route("/models/{id}", get(models::get_model))
        .route("/models/{id}", axum::routing::delete(models::remove_model))
        .route("/models/convert", post(models::convert_checkpoint));

    router.layer(TraceLayer::new_for_http().make_span_with(request_span))
}

/// Create a request span that carries route-level identifiers so
/// workflow/run logs can be correlated across requests.
///
/// The span intentionally omits request bodies, query strings, and
/// headers to avoid leaking prompts, API keys, or backend-private
/// payload keys.
fn request_span(request: &axum::http::Request<axum::body::Body>) -> tracing::Span {
    tracing::info_span!(
        "request",
        method = %request.method(),
        path = %request.uri().path(),
        workflow_id = tracing::field::Empty,
        run_id = tracing::field::Empty,
        correlation_id = tracing::field::Empty,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    use reimagine_app_host::WorkspaceHost;
    use reimagine_app_host::WorkspaceScope;

    use crate::recorder::RunEventRecorder;
    use crate::state::AxumHostState;

    fn test_state() -> AxumHostState {
        let base = unique_temp_dir("router");
        let host = WorkspaceHost::with_defaults(WorkspaceScope::new("ws-router"), &base);
        AxumHostState::new(Arc::new(host), Arc::new(RunEventRecorder::new()))
    }

    #[tokio::test]
    async fn health_route_is_mounted() {
        let app = build_router().with_state(test_state());
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[cfg(feature = "aigc")]
    #[tokio::test]
    async fn compute_profile_route_returns_uninstalled_burn_worker_diagnostics() {
        use axum::body::to_bytes;

        let app = build_router().with_state(test_state());
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/compute-profile")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("collect compute-profile body");
        let json: serde_json::Value =
            serde_json::from_slice(&body).expect("compute-profile body is JSON");

        let backend_profiles = json["backend_profiles"]
            .as_array()
            .expect("backend_profiles is an array");

        let burn = backend_profiles
            .iter()
            .find(|profile| profile["backend"].as_str() == Some("burn"))
            .expect("Burn profile preserves install diagnostics");
        assert_eq!(
            burn["instances"].as_array().map(Vec::len),
            Some(0),
            "an uninstalled Burn worker must not be exposed as executable, got: {json}"
        );
        assert!(
            burn["diagnostics"]
                .as_array()
                .expect("Burn diagnostics are an array")
                .iter()
                .any(|diagnostic| diagnostic["code"].as_str()
                    == Some("APP_HOST/LOCAL_WORKER_NOT_INSTALLED")),
            "the Burn profile must explain how to restore local execution, got: {json}"
        );

        let total_instances = backend_profiles
            .iter()
            .map(|bp| bp["instances"].as_array().map(Vec::len).unwrap_or(0))
            .sum::<usize>();
        assert_eq!(
            total_instances, 0,
            "profile has no available instances without an installed Burn worker, got: {json}"
        );
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-axum-host-{prefix}-{nonce}"))
    }
}
