//! Shared HTTP test fixture for catalog integration tests.
//!
//! Spins up a local [`axum`] server that simulates GitHub's latest‑Release
//! redirect and serves TUF metadata. The server is configured through a
//! shared [`TestCatalogConfig`] and records every request for assertion.
//!
//! # Layout
//!
//! - `GET /latest`               → 302 to `/releases/tag/worker-catalog-v{N}`
//! - `GET /releases/download/{tag}/{file}` → file data (registered ahead of time)

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Configuration for the test catalog server.
#[derive(Clone)]
pub struct TestCatalogConfig {
    /// The concrete tag to redirect to (e.g. "worker-catalog-v12").
    pub redirect_tag: String,
    /// Port to bind. If `0`, the OS assigns one.
    pub port: u16,
}

impl Default for TestCatalogConfig {
    fn default() -> Self {
        Self {
            redirect_tag: "worker-catalog-v12".to_string(),
            port: 0,
        }
    }
}

/// Shared mutable state for the test server.
pub struct AppState {
    config: TestCatalogConfig,
    pub request_count: AtomicUsize,
    files: RwLock<HashMap<String, Vec<u8>>>,
}

async fn latest_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> impl axum::response::IntoResponse {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    let location = format!("/releases/tag/{}", state.config.redirect_tag);
    (
        axum::http::StatusCode::FOUND,
        [(axum::http::header::LOCATION, location)],
    )
}

async fn asset_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl axum::response::IntoResponse {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    let files = state.files.read().unwrap();
    match files.get(path.as_str()) {
        Some(data) => (axum::http::StatusCode::OK, data.clone()),
        None => (axum::http::StatusCode::NOT_FOUND, Vec::new()),
    }
}

/// Start a server on a background task (must be called from within a tokio
/// runtime — e.g. from a `#[tokio::test]`).
pub async fn start_server(config: TestCatalogConfig) -> (String, Arc<AppState>) {
    let state = Arc::new(AppState {
        request_count: AtomicUsize::new(0),
        config: config.clone(),
        files: RwLock::new(HashMap::new()),
    });

    let app = axum::Router::new()
        .route("/releases/download/{*path}", axum::routing::get(asset_handler))
        .route("/{*path}", axum::routing::get(move |state: axum::extract::State<Arc<AppState>>| async move {
            latest_handler(state).await
        }))
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{}", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Yield once to let the server start accepting.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    (base_url, state)
}

/// Register a file in the server's file store at the given request path.
pub fn register_file(state: &AppState, path: &str, data: Vec<u8>) {
    state.files.write().unwrap().insert(path.to_string(), data);
}

/// Get total request count since server start.
pub fn request_count(state: &AppState) -> usize {
    state.request_count.load(Ordering::SeqCst)
}
