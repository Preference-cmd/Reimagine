//! Axum router construction for the V1 HTTP surface.
//!
//! See `docs/architecture/modules/axum-host.md` for the V1 route
//! surface. The router is the single source of truth for which paths
//! are mounted; tests and `server.rs` both go through [`build_router`].

use axum::Router;
use axum::routing::{get, post};

use crate::routes::{health, nodes, runs, workflows};
use crate::state::AxumHostState;

/// Build the V1 HTTP router. The router is stateless — every
/// handler takes the [`AxumHostState`] via the standard `State`
/// extractor.
pub fn build_router() -> Router<AxumHostState> {
    Router::new()
        .route("/health", get(health::get))
        .route("/nodes", get(nodes::list))
        .route("/workflows/open", post(workflows::open))
        .route("/workflows/{id}/run", post(workflows::run))
        .route("/runs/{id}", get(runs::get))
        .route("/runs/{id}/events", get(runs::events))
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

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-axum-host-{prefix}-{nonce}"))
    }
}
