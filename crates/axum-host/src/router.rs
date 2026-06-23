//! Axum router construction for the V1 HTTP surface.
//!
//! See `docs/architecture/modules/axum-host.md` for the V1 route
//! surface. The router is the single source of truth for which paths
//! are mounted; tests and `server.rs` both go through [`build_router`].

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::api::{artifacts, compute_profile, health, nodes, runs, workflows};
use crate::state::AxumHostState;

/// Build the V1 HTTP router. The router is stateless — every
/// handler takes the [`AxumHostState`] via the standard `State`
/// extractor.
pub fn build_router() -> Router<AxumHostState> {
    Router::new()
        .route("/health", get(health::get))
        .route("/compute-profile", get(compute_profile::get))
        .route("/nodes", get(nodes::list))
        .route("/workflows/open", post(workflows::open))
        .route("/workflows/{id}/run", post(workflows::run))
        .route("/runs/{id}", get(runs::get))
        .route("/runs/{id}/events", get(runs::events))
        .route("/artifacts/{artifact_id}", get(artifacts::get))
        .layer(TraceLayer::new_for_http().make_span_with(request_span))
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

    use reimagine_agent::WorkspaceScope;
    use reimagine_app_host::WorkspaceHost;

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

    #[tokio::test]
    async fn compute_profile_route_returns_cpu_instance() {
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

        // V1 must always include a `candle:cpu` instance, regardless
        // of host hardware. The wire shape mirrors the app-host
        // DTO projection exactly.
        let instances = json["backend_profiles"]
            .as_array()
            .expect("backend_profiles is an array")
            .iter()
            .flat_map(|bp| bp["instances"].as_array().cloned().unwrap_or_default())
            .collect::<Vec<_>>();
        assert!(
            instances.iter().any(|inst| {
                inst["instance"].as_str() == Some("candle:cpu")
                    && inst["status"].as_str() == Some("Available")
            }),
            "expected a `candle:cpu` instance with status=Available, got: {json}"
        );

        // Nested shape: device.kind, device.supported_dtypes, etc.
        let cpu = instances
            .iter()
            .find(|inst| inst["instance"].as_str() == Some("candle:cpu"))
            .expect("candle:cpu present");
        assert_eq!(cpu["device"]["kind"].as_str(), Some("Cpu"));
        assert_eq!(cpu["device"]["label"].as_str(), Some("cpu"));
        assert!(
            cpu["device"]["supported_dtypes"].is_array(),
            "supported_dtypes is a list, got: {}",
            cpu["device"]["supported_dtypes"]
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
